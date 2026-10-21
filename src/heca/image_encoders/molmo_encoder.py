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
        # per-card budget for the weights, and what may spill to host memory; the
        # keypoint encoder of the same process already holds ~2 GB of each card.
        # Empty string disables the cap (weights must then fit on the device alone).
        max_gpu_memory: str = "8000MiB"
        max_cpu_memory: str = "24GiB"

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
        max_memory = None
        if half and self.cfg.max_gpu_memory:
            device_map = "auto"
            max_memory = {
                index: self.cfg.max_gpu_memory
                for index in range(torch.cuda.device_count())
            }
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
            f"{getattr(self.model, 'hf_device_map', None)}"
        )
        self.model.eval()

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
        images = [self.as_image(image)] * len(self.chat_texts)
        inputs = self.processor(
            images=images,
            text=self.chat_texts,
            return_tensors="pt",
            padding=True,
        )

        inputs.pop("token_type_ids", None)
        # with a device map the first module may not be the card everything else
        # lives on, so the batch follows the parameters
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits  # [B, seq_len, vocab_size]

        # last valid token position for each sample
        input_lengths = inputs["attention_mask"].sum(dim=1) - 1

        states: dict[str, tuple[int, float]] = {}
        for i, label in enumerate(self.entity_labels):
            next_logits = logits[i, input_lengths[i]]
            letters = self.entity_letters[i]
            letter_logits = torch.tensor(
                [next_logits[self.tokens[letter]] for letter in letters]
            )
            probs = torch.softmax(letter_logits, dim=0).numpy()
            state_idx = int(probs.argmax())
            states[label] = (state_idx, float(probs[state_idx]))

        return states

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
