"""Synchronized rollout evidence recording."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def diagnostic_frames(
    clean_images: list[np.ndarray],
    policy_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    *,
    parameters: dict[str, Any],
    success: bool,
) -> np.ndarray:
    if not clean_images:
        raise ValueError("diagnostic streams contain no frames")
    if not (len(clean_images) == len(policy_images) == len(wrist_images)):
        raise ValueError("diagnostic streams must have identical lengths")
    parameter_text = ", ".join(
        f"{key}={value}"
        for key, value in sorted(parameters.items())
        if key
        in {
            "replan_steps",
            "init_state_index",
            "task_id",
            "task_suite",
        }
    )
    header = f"clean observer | exact policy input | wrist    success={success}"
    frames = []
    for clean, policy, wrist in zip(clean_images, policy_images, wrist_images):
        if clean.shape != policy.shape or clean.shape != wrist.shape:
            raise ValueError("diagnostic frames must share the same shape")
        panel = np.concatenate((clean, policy, wrist), axis=1)
        canvas = Image.new("RGB", (panel.shape[1], panel.shape[0] + 48), "black")
        canvas.paste(Image.fromarray(panel), (0, 48))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), header, fill="white")
        draw.text((6, 25), parameter_text, fill="white")
        frames.append(np.asarray(canvas))
    return np.asarray(frames)


def write_evidence_videos(
    episode_dir: Path,
    clean_images: list[np.ndarray],
    policy_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    *,
    parameters: dict[str, Any],
    success: bool,
) -> dict[str, str]:
    videos = {
        "clean_observer": episode_dir / "clean_observer.mp4",
        "policy_input": episode_dir / "policy_input.mp4",
        "wrist": episode_dir / "wrist.mp4",
        "diagnostic": episode_dir / "diagnostic.mp4",
    }
    iio.imwrite(videos["clean_observer"], np.asarray(clean_images), fps=10)
    iio.imwrite(videos["policy_input"], np.asarray(policy_images), fps=10)
    iio.imwrite(videos["wrist"], np.asarray(wrist_images), fps=10)
    iio.imwrite(
        videos["diagnostic"],
        diagnostic_frames(
            clean_images,
            policy_images,
            wrist_images,
            parameters=parameters,
            success=success,
        ),
        fps=10,
    )
    return {key: str(path) for key, path in videos.items()}
