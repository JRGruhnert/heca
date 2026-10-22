import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# NOTE: this is needed cause tensorflow has an internal conflict with other packages
from textwrap import dedent

import numpy as np
import torch

from dataclasses import dataclass
from typing import Any
from transformers import AutoProcessor, AutoModelForImageTextToText

from heca.scenes.scene import Scene
from heca.data.entity import Entity
from heca.helper import molmo
from heca.image_encoders.image_encoder import ImageEncoder
from heca.data.data import TDImage

from heca.misc import hardware, logger


class MolmoEncoder(ImageEncoder):
    @dataclass(kw_only=True)
    class Config(ImageEncoder.Config):
        image_size: tuple[int, int] = (256, 256)
        tag: str = "allenai/Molmo2-4B"
        # Per-card budget for the weights, and what may spill to host memory.
        # The weights are 19.4 GB fp32 on disk, i.e. ~9.3 GB in fp16 -- small enough
        # for one 11264 MiB card, which shows torch ~10.5 GiB once the CUDA context
        # is taken, leaving room for the keypoint encoder and the activations of one
        # 256x256 frame. Capping lower is what makes device_map spread the model over
        # every visible card instead, and that spread is what collides when several
        # runs share the node: each one fills all cards evenly, so they all pick the
        # same "emptiest" card.
        # Empty string disables the cap (weights must then fit on the device alone).
        max_gpu_memory: str = "10GiB"
        max_cpu_memory: str = "24GiB"
        # state questions per forward. The batch is the same frame repeated once per
        # entity, so its activations (742 MiB for six) are what does not fit beside
        # the weights on one card; three questions per forward leave ~500 MiB free.
        state_batch_size: int = 3
        # visible cards the weights may go on, comma separated ("" = all of them).
        # With CUDA_VISIBLE_DEVICES="2,3" the answers are 0 and 1, so "0" keeps the
        # model off the card the keypoint encoder was sent to.
        gpu_cards: str = ""

        @property
        def state_batch(self) -> int:
            """Questions per forward, overridable per run (HECA_MOLMO_BATCH)."""
            raw = os.environ.get("HECA_MOLMO_BATCH", "").strip()
            try:
                value = int(raw) if raw else self.state_batch_size
            except ValueError:
                value = self.state_batch_size
            return max(1, value)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.chat_texts: list[str] = []
        self.entity_labels: list[str] = []
        self.entity_letters: list[list[str]] = []
        self.tokens: dict[str, int] = {}
        self.processor = None
        self.model = None
        self.task_str = (
            "Answer the following question about the image with only the single "
            "letter of the correct answer. There can only be one correct answer."
        )

    def load_processor(self):
        """The tokenizer side only: enough to build the questions, and cheap."""
        if self.processor is not None:
            return
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.tag,
            trust_remote_code=True,
        )
        self.tokens = {
            letter: self.processor.tokenizer.encode(letter, add_special_tokens=False)[0]
            for letter in molmo.ALPHABET
        }

    def load_model(self):
        """Load the weights, once, when there is a state to ask about.

        Kept apart from the processor on purpose: fitting the demos asks no state
        questions, so it must not pull 19 GB of weights into the run.
        """
        self.load_processor()
        if self.model is not None:
            return
        # 4.85B parameters: fp32 is 19.4 GB, which no card here holds, and fp16 is
        # 9.7 GB -- still more than is left once the keypoint encoder of the same
        # process is up (~2 GB) on an 11 GB card. So the weights go through a device
        # map with a cap: the overflow lands in host memory instead of an OOM.
        # fp16 rather than bf16 because Turing accelerates fp16 and emulates bf16.
        half = hardware.device.type == "cuda"
        dtype = torch.float16 if half else torch.float32
        device_map: Any = hardware.device
        # the cap is a living number: weights + keypoint encoder + a demo batch's
        # activations have to fit the same card, and how much is left differs by
        # card and by what else is resident, so it is overridable without a code edit
        cap = os.environ.get("HECA_MOLMO_GPU_MEMORY", self.cfg.max_gpu_memory)
        cards = self.cfg.gpu_cards.strip() or os.environ.get(
            "HECA_MOLMO_DEVICES", ""
        ).strip()
        max_memory = None
        if half and cap:
            device_map = "auto"
            indices = (
                [int(part) for part in cards.split(",") if part.strip()]
                if cards
                else list(range(torch.cuda.device_count()))
            )
            max_memory = {index: cap for index in indices}
            max_memory["cpu"] = self.cfg.max_cpu_memory
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.tag,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
        )
        logger.info(
            f"Molmo loaded in {dtype} ({device_map}), map: "
            f"{getattr(self.model, 'hf_device_map', None)}, gpu cap {cap}, "
            f"cards {cards or 'all'}"
        )
        self.model.eval()
        self.split_vocab_lookup()

    def split_vocab_lookup(self):
        """Stop Molmo's word embedding from fusing its two tables on every call.

        Molmo2 carries two embedding tables and its ``wte.forward`` runs
        ``F.embedding(x, torch.cat([self.embedding, self.new_embedding]))``: the
        concatenation allocates a full copy of the vocabulary (~744 MiB in fp16)
        before the lookup, transiently, on every forward. That copy is what tips an
        11 GB card over -- it needs ~100 MiB more than is free -- while the actual
        result is a few MB of rows. Reading the two tables separately gives the
        same vectors without the intermediate tensor.
        """
        wte = getattr(getattr(self.model, "model", None), "transformer", None)
        wte = getattr(wte, "wte", None)
        base = getattr(wte, "embedding", None)
        extra = getattr(wte, "new_embedding", None)
        if base is None or extra is None:
            return

        def forward(ids: torch.Tensor) -> torch.Tensor:
            rows = base.shape[0]
            first = torch.embedding(base, ids.clamp(max=rows - 1))
            tail = ids >= rows
            if not bool(tail.any()):
                return first
            shifted = torch.embedding(extra, (ids - rows).clamp(min=0))
            return torch.where(tail.unsqueeze(-1), shifted, first)

        wte.forward = forward
        logger.info(
            f"Molmo vocabulary split into {base.shape[0]} + {extra.shape[0]} rows "
            f"(no per-call concatenation)"
        )

    @staticmethod
    def as_image(image: TDImage) -> np.ndarray:
        """The image the processor takes: (H, W, 3) uint8 from the tensor form."""
        rgb = image.rgb
        if isinstance(rgb, torch.Tensor):
            return (rgb.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(
                np.uint8
            )
        return np.asarray(rgb, dtype=np.uint8)

    def extract_states(self, image: TDImage) -> dict[str, tuple[int, float]]:
        """The state of every entity that has more than one, keyed by label."""
        if not self.chat_texts:
            return {}
        self.load_model()
        assert self.processor is not None
        assert self.model is not None
        # with a device map the first module may not be the card everything else
        # lives on, so the batch follows the parameters
        device = next(self.model.parameters()).device
        frame = self.as_image(image)
        states: dict[str, tuple[int, float]] = {}

        # one forward per batch of questions: fewer images per forward is what keeps
        # the activations beside the weights on a single card, and entities are
        # independent, so the split costs only wall clock
        step = self.cfg.state_batch
        for begin in range(0, len(self.chat_texts), step):
            texts = self.chat_texts[begin : begin + step]
            inputs = self.processor(
                images=[frame] * len(texts),
                text=texts,
                return_tensors="pt",
                padding=True,
            )
            inputs.pop("token_type_ids", None)
            inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }

            # last valid token position for each sample
            input_lengths = inputs["attention_mask"].sum(dim=1) - 1
            with torch.inference_mode():
                next_logits = self.last_token_logits(inputs, input_lengths)

            for offset, label in enumerate(self.entity_labels[begin : begin + step]):
                letters = self.entity_letters[begin + offset]
                letter_logits = torch.stack(
                    [next_logits[offset][self.tokens[letter]] for letter in letters]
                )
                probs = torch.softmax(letter_logits, dim=0).detach().cpu().numpy()
                state_idx = int(probs.argmax())
                states[label] = (state_idx, float(probs[state_idx]))

            del inputs, next_logits

        return states

    def last_token_logits(
        self,
        inputs: dict[str, Any],
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Logits of each sample's last real token, shape ``[B, vocab]``.

        A plain forward materialises ``[B, seq_len, vocab]`` -- ~550 MB for three
        questions over a 152k vocabulary -- and that tensor is what tips a 10 GiB
        card over the edge beside 9 GB of weights. The state question only reads a
        handful of letter positions of the last token, so the hidden states suffice
        and the output projection runs on one row per sample instead, which is a
        few hundred kB. Falls back to the full logits when the model exposes no
        hidden states or no output embedding.
        """
        out = None
        hidden = None
        try:
            out = self.model(**inputs, output_hidden_states=True)
            hidden = getattr(out, "hidden_states", None)
        except TypeError:
            out = None

        head = None
        if hidden:
            getter = getattr(self.model, "get_output_embeddings", None)
            head = getter() if callable(getter) else None
            if head is None:
                head = getattr(self.model, "lm_head", None)

        if hidden and head is not None:
            last = hidden[-1]
            index = input_lengths.to(last.device)
            rows = last[torch.arange(index.shape[0], device=last.device), index]
            return head(rows.to(next(head.parameters()).device))

        logits = getattr(out, "logits", None)
        if logits is None:
            logits = self.model(**inputs).logits
        index = input_lengths.to(logits.device)
        return logits[torch.arange(index.shape[0], device=logits.device), index]

    def prepare_for_scene(self, cfg: Scene.Config):
        scene = Scene.get(cfg)
        wanted: list[tuple[str, Entity]] = []
        for label, entity in scene.entities.items():
            if entity.cfg.n_states <= 1:
                logger.info(f"{label}: one state only, the state encoder skips it")
                continue
            wanted.append((label, entity))

        self.chat_texts = []
        self.entity_labels = []
        self.entity_letters = []
        if not wanted:
            return
        self.load_processor()

        for label, entity in wanted:
            letters = molmo.ALPHABET[: len(entity.cfg.answers)]
            choices = "\n".join(
                f"{letter}: {state}"
                for letter, state in zip(letters, entity.cfg.answers)
            )
            question_str = dedent(f"""
            {scene.description} 

            Task:
            {self.task_str}  

            Question:
            {entity.cfg.question} 

            Answers:
            {choices}      
            """)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question_str},
                    ],
                }
            ]
            assert self.processor is not None
            self.chat_texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            self.entity_labels.append(label)
            self.entity_letters.append(letters)
