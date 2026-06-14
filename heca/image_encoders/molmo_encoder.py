from textwrap import dedent

import torch

from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForImageTextToText

from heca.environment.scenes.scene import Scene
from heca.helper import molmo
from heca.image_encoders.image_encoder import ImageEncoder
from heca.misc.td import TDImage


class MolmoEncoder(ImageEncoder):
    @dataclass(kw_only=True)
    class Config(ImageEncoder.Config):
        kp_selection_threshold: float = 0.2
        image_size: tuple[int, int] = (256, 256)
        tag: str = "allenai/Molmo2-4B"

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.tag, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.tag,
            trust_remote_code=True,
            dtype=torch.float32,
            device_map="auto",
        )
        self.model.eval()
        self.chat_texts = []
        self.entity_letters = []
        self.tokens = {}
        for letter in molmo.ALPHABET:
            token = self.processor.tokenizer.encode(letter, add_special_tokens=False)[0]
            self.tokens[letter] = token

        self.task_str = "Answer the following question about the image with only the single letter of the correct answer. There can only be one correct answer."

    def extract_entities(self, image: TDImage) -> torch.Tensor:
        raise NotImplementedError()

    def extract_states(self, image: TDImage, kps_raw_2d: torch.Tensor) -> torch.Tensor:
        images = [image] * len(self.chat_texts)
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

        states = []

        for i in range(len(self.chat_texts)):
            next_logits = logits[i, input_lengths[i]]
            letters = self.entity_letters[i]
            scores = torch.tensor(
                [next_logits[self.tokens[letter]] for letter in letters]
            )
            probs = torch.softmax(scores, dim=0)
            state_idx = probs.argmax().item()
            states.append(state_idx)

        return torch.tensor(states)

    def prepare_for_scene(self, cfg: Scene.Config):
        scene = Scene.get(cfg)
        for entity in scene.entities:
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
            chat_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            self.chat_texts.append(chat_text)
            self.entity_letters.append(letters)
