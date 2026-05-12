import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

MODEL = "allenai/Molmo-7B-D-0924"

processor = AutoProcessor.from_pretrained(
    MODEL,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

image = Image.open("drawer.png").convert("RGB")

prompt = """
Is the drawer open or closed?

Answer with exactly one word.
"""

inputs = processor(
    images=[image],
    text=prompt,
    return_tensors="pt",
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

# logits for NEXT token
logits = outputs.logits[:, -1, :]
