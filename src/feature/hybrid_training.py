"""Training utilities for the RIPE-style keypoint detector.

The QSED/Von Neumann entropy descriptor is intentionally not learned here.  This
module only trains the detector to rank geometrically repeatable keypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import random

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .entropy_descriptor import detect_default_keypoints


@dataclass(frozen=True)
class HPatchesSplit:
    """Disjoint sequence-level split; all pairs of one sequence stay together."""

    train: tuple[dict, ...]
    validation: tuple[dict, ...]
    test: tuple[dict, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "train": [item["name"] for item in self.train],
            "validation": [item["name"] for item in self.validation],
            "test": [item["name"] for item in self.test],
        }


def split_hpatches_sequences(
    sequences: Sequence[Mapping[str, str]], seed: int = 7
) -> HPatchesSplit:
    """Create a reproducible 70/15/15 split stratified by HPatches type."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {"illumination": [], "viewpoint": []}
    for sequence in sequences:
        item = dict(sequence)
        groups.setdefault(item["type"], []).append(item)

    train: list[dict] = []
    validation: list[dict] = []
    test: list[dict] = []
    for group in groups.values():
        rng.shuffle(group)
        count = len(group)
        train_count = round(0.70 * count)
        validation_count = round(0.15 * count)
        train.extend(group[:train_count])
        validation.extend(group[train_count:train_count + validation_count])
        test.extend(group[train_count + validation_count:])

    result = HPatchesSplit(tuple(train), tuple(validation), tuple(test))
    names = [name for values in result.as_dict().values() for name in values]
    if len(names) != len(set(names)) or len(names) != len(sequences):
        raise ValueError("HPatches split must contain each sequence exactly once.")
    return result


def _resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def _random_homography(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a moderate planar transform that keeps useful image overlap."""
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    center = np.array([width / 2, height / 2], dtype=np.float32)
    angle = np.deg2rad(rng.uniform(-30.0, 30.0))
    scale = rng.uniform(0.80, 1.20)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    target = (source - center) @ rotation.T * scale + center
    target += rng.uniform([-0.10 * width, -0.10 * height], [0.10 * width, 0.10 * height]).astype(np.float32)
    target += rng.uniform(-0.04, 0.04, size=(4, 2)).astype(np.float32) * np.array([width, height], dtype=np.float32)
    return cv2.getPerspectiveTransform(source, target.astype(np.float32))


def _photometric_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = float(rng.uniform(0.70, 1.30))
    beta = float(rng.uniform(-25.0, 25.0))
    augmented = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        augmented = cv2.GaussianBlur(augmented, (3, 3), rng.uniform(0.1, 1.0))
    return augmented


def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(points.astype(np.float32).reshape(-1, 1, 2), homography).reshape(-1, 2)


def _keypoint_map(points: np.ndarray, shape: tuple[int, int], radius: int = 3) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float32)
    height, width = shape
    for x, y in np.rint(points).astype(np.int32):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(result, (int(x), int(y)), radius, 1.0, thickness=-1)
    return result


class SyntheticHomographyDataset(Dataset):
    """Synthetic pairs built only from images in the training sequence split."""

    def __init__(
        self,
        sequences: Iterable[Mapping[str, str]],
        samples_per_epoch: int = 50_000,
        max_side: int = 480,
        max_keypoints: int = 1024,
        seed: int = 7,
    ) -> None:
        self.image_paths = [
            Path(sequence["path"]) / f"{index}.ppm"
            for sequence in sequences
            for index in range(1, 7)
            if (Path(sequence["path"]) / f"{index}.ppm").is_file()
        ]
        if not self.image_paths:
            raise ValueError("Training split contains no HPatches images.")
        self.samples_per_epoch = samples_per_epoch
        self.max_side = max_side
        self.max_keypoints = max_keypoints
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Refresh synthetic transforms while retaining reproducibility."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(self.seed + self.epoch * self.samples_per_epoch + index)
        image = cv2.imread(str(self.image_paths[index % len(self.image_paths)]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(self.image_paths[index % len(self.image_paths)])
        source = _resize_max_side(image, self.max_side)
        height, width = source.shape
        homography = _random_homography(width, height, rng)
        target = cv2.warpPerspective(source, homography, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        source = _photometric_augment(source, rng)
        target = _photometric_augment(target, rng)

        source_xy = np.asarray(
            [(keypoint.x, keypoint.y) for keypoint in detect_default_keypoints(source, self.max_keypoints)],
            dtype=np.float32,
        ).reshape(-1, 2)
        target_xy = _transform_points(source_xy, homography)
        valid = (
            (target_xy[:, 0] >= 3) & (target_xy[:, 0] < width - 3)
            & (target_xy[:, 1] >= 3) & (target_xy[:, 1] < height - 3)
        )
        source_xy, target_xy = source_xy[valid], target_xy[valid]
        return {
            "source": torch.from_numpy(source[None].astype(np.float32) / 255.0),
            "target": torch.from_numpy(target[None].astype(np.float32) / 255.0),
            "source_targets": torch.from_numpy(_keypoint_map(source_xy, source.shape)),
            "target_targets": torch.from_numpy(_keypoint_map(target_xy, target.shape)),
        }


def configure_hybrid_finetuning(model: nn.Module) -> list[nn.Parameter]:
    """Freeze early ImageNet features; train stage 4 and detector-specific layers."""
    for parameter in model.encoder.stage1.parameters():
        parameter.requires_grad = False
    for parameter in model.encoder.stage2.parameters():
        parameter.requires_grad = False
    for parameter in model.encoder.stage3.parameters():
        parameter.requires_grad = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable Hybrid detector parameters remain.")
    return parameters


def detector_supervision_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Class-balanced detector loss for repeatable Sobel candidates."""
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must match, got {logits.shape} and {targets.shape}.")
    positives = targets.sum().clamp_min(1.0)
    negatives = (targets.numel() - positives).clamp_min(1.0)
    positive_weight = (negatives / positives).clamp(max=50.0)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)


def train_detector_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    accumulation_steps: int = 8,
) -> float:
    """Train one epoch; returns the mean symmetric detector-supervision loss."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for step, batch in enumerate(loader, start=1):
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        source_targets = batch["source_targets"].to(device, non_blocking=True)
        target_targets = batch["target_targets"].to(device, non_blocking=True)
        enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=enabled):
            loss = (detector_supervision_loss(model(source), source_targets)
                    + detector_supervision_loss(model(target), target_targets)) / 2.0
            scaled_loss = loss / accumulation_steps
        if scaler is not None and enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if step % accumulation_steps == 0 or step == len(loader):
            if scaler is not None and enabled:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if scaler is not None and enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach())
    return total_loss / max(1, len(loader))
