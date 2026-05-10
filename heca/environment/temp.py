import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# --------------------------------------------------
# Load DINOv2
# --------------------------------------------------

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

model.eval().cuda()

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

# --------------------------------------------------
# Embedding extraction
# --------------------------------------------------


@torch.no_grad()
def embed_image(image_path):
    img = Image.open(image_path).convert("RGB")

    x = transform(img).unsqueeze(0).cuda()

    # global embedding
    feat = model(x)

    # normalize
    feat = F.normalize(feat, dim=-1)

    return feat.squeeze(0).cpu()


# --------------------------------------------------
# Build memory
# --------------------------------------------------

memory = {
    "open": [
        embed_image("open1.png"),
        embed_image("open2.png"),
    ],
    "closed": [
        embed_image("closed1.png"),
        embed_image("closed2.png"),
    ],
}

# --------------------------------------------------
# Classification
# --------------------------------------------------


def classify(query_embedding):
    scores = {}

    for cls, embeddings in memory.items():

        sims = [
            F.cosine_similarity(
                query_embedding.unsqueeze(0),
                e.unsqueeze(0),
            ).item()
            for e in embeddings
        ]

        scores[cls] = max(sims)

    pred = max(scores, key=scores.get)

    return pred, scores


# --------------------------------------------------
# Example inference
# --------------------------------------------------

query = embed_image("test.png")

pred, scores = classify(query)

print(pred)
print(scores)
