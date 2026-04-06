import os
import sys
import tempfile

import yaml

from lightning.pytorch.cli import LightningCLI  # noqa: E402

from datamodule import ECGDataModule  # noqa: E402
from lit_model import SignalAutoencoderModule  # noqa: E402


def _patch_config(config: dict, run_name: str) -> None:
    """Inject run_name into ModelCheckpoint dirpath."""
    for cb in config.get("trainer", {}).get("callbacks", []):
        if "ModelCheckpoint" in cb.get("class_path", ""):
            cb.setdefault("init_args", {})["dirpath"] = f"checkpoints/{run_name}"


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    run_name = params.get("signal_autoencoder", {}).get("run_name", "default")

    # Patch the config file in a temp copy so that run_name is injected into
    # the ModelCheckpoint dirpath without relying on
    # the unsupported --trainer.callbacks[N].init_args.* CLI syntax.
    argv = list(sys.argv[1:])
    tmp_path = None
    for i, arg in enumerate(argv):
        config_path = None
        if arg in ("--config", "-c") and i + 1 < len(argv):
            config_path = argv[i + 1]
        elif arg.startswith(("--config=", "-c=")):
            config_path = arg.split("=", 1)[1]

        if config_path is not None:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            _patch_config(config, run_name)
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
            yaml.dump(config, tmp)
            tmp.close()
            tmp_path = tmp.name
            if "=" in arg:
                argv[i] = f"{arg.split('=')[0]}={tmp_path}"
            else:
                argv[i + 1] = tmp_path
            break

    sys.argv[1:] = argv
    try:
        LightningCLI(
            SignalAutoencoderModule,
            ECGDataModule,
            save_config_callback=None,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
