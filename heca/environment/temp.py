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
    "open": torch.stack(
        [
            embed_image("open1.png"),
            embed_image("open2.png"),
        ],
        dim=0,
    ),  # shape (B_open, D)
    "closed": torch.stack(
        [
            embed_image("closed1.png"),
            embed_image("closed2.png"),
        ],
        dim=0,
    ),  # shape (B_closed, D)
}

# --------------------------------------------------
# Classification
# --------------------------------------------------


def classify(query_embedding):
    """
    query_embedding: torch.Tensor of shape (D,) or (1, D)
    memory: dict[str, torch.Tensor] where each value is (B, D)
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.unsqueeze(0)  # (1, D)
    scores = {}
    for cls, embeddings in memory.items():
        # embeddings: (B, D), query_embedding: (1, D)
        sims = F.cosine_similarity(embeddings, query_embedding, dim=1)  # (B,)
        scores[cls] = sims.max().item()
    # Find the class with the highest score
    best_cls = None
    best_score = float('-inf')
    for cls, score in scores.items():
        if score > best_score:
            best_score = score
            best_cls = cls
    return best_cls, scores


# --------------------------------------------------
# Example inference
# --------------------------------------------------

query = embed_image("test.png")

pred, scores = classify(query)

print(pred)
print(scores)
