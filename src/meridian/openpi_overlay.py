from __future__ import annotations

from pathlib import Path

BEGIN = "    # BEGIN MERIDIAN MANAGED CONFIG"
END = "    # END MERIDIAN MANAGED CONFIG"
MARKER = "    #\n    # Fine-tuning Aloha configs.\n"

BLOCK = r"""    # BEGIN MERIDIAN MANAGED CONFIG
    TrainConfig(
        name="meridian_pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id=os.environ.get("MERIDIAN_DATASET_REPO", "meridian/gate0"),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_libero/assets",
                asset_id="physical-intelligence/libero",
            ),
            extra_delta_transform=False,
        ),
        batch_size=int(os.environ.get("MERIDIAN_BATCH_SIZE", "2")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(os.environ.get("MERIDIAN_WARMUP_STEPS", "1")),
            peak_lr=float(os.environ.get("MERIDIAN_PEAK_LR", "5e-5")),
            decay_steps=int(os.environ.get("MERIDIAN_DECAY_STEPS", "1000")),
            decay_lr=float(os.environ.get("MERIDIAN_DECAY_LR", "5e-5")),
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        checkpoint_base_dir=os.environ.get("MERIDIAN_CHECKPOINT_ROOT", "./checkpoints"),
        num_train_steps=int(os.environ.get("MERIDIAN_TRAIN_STEPS", "2")),
        save_interval=int(os.environ.get("MERIDIAN_SAVE_INTERVAL", "1")),
        keep_period=None,
        wandb_enabled=False,
    ),
    # END MERIDIAN MANAGED CONFIG
"""


def install(path: Path) -> bool:
    text = path.read_text()
    if BEGIN in text:
        start = text.index(BEGIN)
        end = text.index(END, start) + len(END)
        text = text[:start] + text[end:].lstrip("\n")
    if MARKER not in text:
        raise ValueError(f"OpenPI config marker not found in {path}")
    if "import os\n" not in text:
        text = text.replace("import logging\n", "import logging\nimport os\n", 1)
    path.write_text(text.replace(MARKER, BLOCK + MARKER, 1))
    return True
