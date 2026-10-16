"""Torch models: frozen image backbone, geometry preprocessor, CORAL ordinal head.

The proposed model (Track C) fuses a stitch image crop with wound-relative
geometry and predicts an ordinal 1-5 score through a CORAL head.  Geometry-only
and image-only variants reuse exactly the same head so the ablation is
apples-to-apples.  Everything is sized to fit a 4 GB GTX 1650 Ti: the backbone
is frozen, embeddings are precomputed once, and only a small MLP head is
trained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config
from .geometry import GEOMETRY_FEATURE_COLUMNS

try:  # torchvision is required for the image branch only
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights, resnet18

    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover - environment dependent
    transforms = None  # type: ignore[assignment]
    ResNet18_Weights = None  # type: ignore[assignment]
    resnet18 = None  # type: ignore[assignment]
    _HAS_TORCHVISION = False


# --------------------------------------------------------------------------- #
# Geometry preprocessing
# --------------------------------------------------------------------------- #
class GeometryPreprocessor:
    """Median imputation + standardisation, fitted on the training fold only."""

    def __init__(self, columns: tuple[str, ...] = GEOMETRY_FEATURE_COLUMNS) -> None:
        self.columns = columns
        self._median: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "GeometryPreprocessor":
        values = df.loc[:, list(self.columns)].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            self._median = np.nanmedian(values, axis=0)
        self._median = np.where(np.isfinite(self._median), self._median, 0.0)
        filled = np.where(np.isfinite(values), values, self._median[None, :])
        self._mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        self._std = np.where(std > 1e-8, std, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._median is None:
            raise RuntimeError("GeometryPreprocessor must be fitted before transforming")
        values = df.loc[:, list(self.columns)].to_numpy(dtype=float)
        filled = np.where(np.isfinite(values), values, self._median[None, :])
        return ((filled - self._mean) / self._std).astype(np.float32)

    @property
    def dim(self) -> int:
        return len(self.columns)


# --------------------------------------------------------------------------- #
# Image embeddings (frozen backbone)
# --------------------------------------------------------------------------- #
class ImageEmbedder:
    """Frozen ImageNet backbone that maps a stitch crop to a feature vector."""

    def __init__(self, device: str | None = None, arch: str = "resnet18") -> None:
        if not _HAS_TORCHVISION:
            raise RuntimeError("torchvision is required for the image branch")
        self.arch = arch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pretrained = True
        try:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            self.pretrained = False
            backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone = backbone.to(self.device)
        self.embedding_dim = 512
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((config.EVAL_IMAGE_SIZE, config.EVAL_IMAGE_SIZE)), transforms.ToTensor(), normalize]
        )

    @torch.no_grad()
    def embed_paths(self, paths: list[str], batch_size: int = 16) -> np.ndarray:
        import cv2

        embeddings: list[np.ndarray] = []
        buffer: list[torch.Tensor] = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                image = np.zeros((config.CROP_SIZE, config.CROP_SIZE, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            buffer.append(self.transform(rgb))
            if len(buffer) == batch_size:
                embeddings.append(self._flush(buffer))
                buffer = []
        if buffer:
            embeddings.append(self._flush(buffer))
        if not embeddings:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.concatenate(embeddings, axis=0)

    def _flush(self, buffer: list[torch.Tensor]) -> np.ndarray:
        batch = torch.stack(buffer).to(self.device)
        return self.backbone(batch).detach().cpu().numpy().astype(np.float32)


def precompute_embeddings(
    stitch_ids: list[str],
    crop_paths: list[str],
    out_path: Path = config.MODELS_DIR / "embeddings.npz",
    overwrite: bool = False,
    device: str | None = None,
) -> tuple[np.ndarray, bool]:
    """Return ``(embedding_matrix, pretrained_flag)``, caching to ``out_path``."""
    if out_path.exists() and not overwrite:
        cached = np.load(out_path, allow_pickle=True)
        cached_ids = [str(v) for v in cached["stitch_ids"]]
        if cached_ids == [str(s) for s in stitch_ids]:
            return cached["embeddings"].astype(np.float32), bool(cached["pretrained"])

    embedder = ImageEmbedder(device=device)
    matrix = embedder.embed_paths(crop_paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embeddings=matrix, stitch_ids=np.asarray(stitch_ids, dtype=object), pretrained=embedder.pretrained)
    return matrix, embedder.pretrained


# --------------------------------------------------------------------------- #
# CORAL ordinal head and hybrid network
# --------------------------------------------------------------------------- #
def coral_targets(y: torch.Tensor, n_scores: int = config.N_SCORES) -> torch.Tensor:
    thresholds = torch.arange(config.SCORE_MIN, config.SCORE_MIN + n_scores - 1, device=y.device)
    return (y.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def coral_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, coral_targets(y))


def coral_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Turn K-1 CORAL logits into a probability distribution over scores 1..K."""
    cumulative = torch.sigmoid(logits)
    n_scores = cumulative.shape[1] + 1
    proba = logits.new_zeros((logits.shape[0], n_scores))
    proba[:, 0] = 1.0 - cumulative[:, 0]
    for k in range(1, n_scores - 1):
        proba[:, k] = cumulative[:, k - 1] - cumulative[:, k]
    proba[:, n_scores - 1] = cumulative[:, n_scores - 2]
    proba = proba.clamp_min(1e-6)
    return proba / proba.sum(dim=1, keepdim=True)


class HybridOrdinalNet(nn.Module):
    """Small MLP head over (frozen image embedding, geometry) -> CORAL logits."""

    def __init__(
        self,
        geometry_dim: int,
        image_dim: int = 512,
        hidden: int = 128,
        n_scores: int = config.N_SCORES,
        dropout: float = 0.2,
        use_image: bool = True,
        use_geometry: bool = True,
    ) -> None:
        super().__init__()
        if not use_image and not use_geometry:
            raise ValueError("At least one input branch must be enabled")
        self.use_image = use_image
        self.use_geometry = use_geometry
        in_dim = (image_dim if use_image else 0) + (geometry_dim if use_geometry else 0)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden // 2, 16)),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(max(hidden // 2, 16), n_scores - 1)

    def forward(self, image_features: torch.Tensor | None = None, geometry: torch.Tensor | None = None) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.use_image:
            parts.append(image_features)
        if self.use_geometry:
            parts.append(geometry)
        inputs = [p for p in parts if p is not None]
        if len(inputs) != len(parts):
            raise ValueError("Missing input branch for the configured model")
        x = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=1)
        return self.head(self.trunk(x))

    @torch.no_grad()
    def predict_proba(self, image_features: torch.Tensor | None, geometry: torch.Tensor | None) -> np.ndarray:
        self.eval()
        return coral_probabilities(self.forward(image_features, geometry)).cpu().numpy()


class ResidualHybridNet(nn.Module):
    """Geometry-dominant hybrid: the image contributes a small bounded correction.

    Fuses wound-relative geometry with the frozen image embedding by adding a
    gated image residual (gate initialised small) to a geometry representation,
    then a CORAL head.  Because the image term starts small and is bounded, the
    model can rarely under-perform a geometry-only model by more than the
    correction it learns.
    """

    def __init__(
        self,
        geometry_dim: int,
        image_dim: int = 512,
        hidden: int = 192,
        proj: int = 96,
        dropout: float = 0.25,
        n_scores: int = config.N_SCORES,
    ) -> None:
        super().__init__()
        self.geo_enc = nn.Sequential(
            nn.Linear(geometry_dim, proj),
            nn.ReLU(inplace=True),
            nn.Linear(proj, proj),
            nn.ReLU(inplace=True),
        )
        self.img_enc = nn.Sequential(nn.Linear(image_dim, proj), nn.ReLU(inplace=True))
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.trunk = nn.Sequential(
            nn.Linear(proj, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden // 2, 16)),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(max(hidden // 2, 16), n_scores - 1)

    def forward(self, image_features: torch.Tensor | None = None, geometry: torch.Tensor | None = None) -> torch.Tensor:
        if geometry is None:
            x = self.img_enc(image_features)
        elif image_features is None:
            x = self.geo_enc(geometry)
        else:
            x = self.geo_enc(geometry) + torch.sigmoid(self.gate) * self.img_enc(image_features)
        return self.head(self.trunk(x))

    @torch.no_grad()
    def predict_proba(self, image_features: torch.Tensor | None, geometry: torch.Tensor | None) -> np.ndarray:
        self.eval()
        return coral_probabilities(self.forward(image_features, geometry)).cpu().numpy()


def build_head(
    variant: str,
    geometry_dim: int,
    image_dim: int = 512,
    hidden: int = 128,
    seed: int = config.SEED,
) -> HybridOrdinalNet:
    """Build a ``geometry_only`` / ``image_only`` / ``hybrid`` CORAL network."""
    torch.manual_seed(seed)
    if variant == "geometry_only":
        return HybridOrdinalNet(geometry_dim, image_dim, hidden, use_image=False, use_geometry=True)
    if variant == "image_only":
        return HybridOrdinalNet(geometry_dim, image_dim, hidden, use_image=True, use_geometry=False)
    return ResidualHybridNet(geometry_dim, image_dim=image_dim)
