import os
import sys

from lightning.pytorch.cli import LightningCLI  # noqa: E402

from datamodule import ECGDataModule  # noqa: E402
from lit_model import SignalAutoencoderModule  # noqa: E402


def main() -> None:
    LightningCLI(
        SignalAutoencoderModule,
        ECGDataModule,
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
