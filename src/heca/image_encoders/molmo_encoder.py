import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# NOTE: this is needed cause tensorflow has an internal conflict with other packages
from textwrap import dedent

import numpy as np
import torch

from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForImageTextToText

from heca.scenes.scene import Scene
from heca.data.entity import Entity
from heca.helper import molmo
from heca.image_encoders.image_encoder import ImageEncoder
from heca.data.data import TDImage

from heca.misc import logger


class MolmoEncoder(ImageEncoder):
    @dataclass(kw_only=True)
    class Config(ImageEncoder.Config):
        image_size: tuple[int, int] = (256, 256)
        tag: str = "allenai/Molmo2-4B"

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

    def load_model(self):
        """Load the weights, once, when there is a state to ask about."""
        if self.model is not None:
            return
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.tag,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.tag,
            trust_remote_code=True,
            dtype=torch.float32,
            device_map="auto",
        )
        self.model.eval()
        self.tokens = {
            letter: self.processor.tokenizer.encode(letter, add_special_tokens=False)[0]
            for letter in molmo.ALPHABET
        }

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
            # No entity in this scene has a state to tell apart, so nothing is
            # asked and the weights are never loaded.
            return {}
        self.load_model()
        images = [self.as_image(image)] * len(self.chat_texts)
        inputs = self.processor(
            images=images,
            text=self.chat_texts,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        inputs.pop("token_type_ids", None)

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
            prior = self.priors.get(label)
            state_idx = int(probs.argmax()) if prior is None else prior.update(probs)
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
        self.load_model()

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
            self.chat_texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            self.entity_labels.append(label)
            self.entity_letters.append(letters)
