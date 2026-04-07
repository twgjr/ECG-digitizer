import json
from pathlib import Path

import lightning as L


class MetricsJsonCallback(L.Callback):
    """Writes final trainer metrics to a JSON file for DVC tracking."""

    def __init__(self, filepath: str = "metrics/signal_autoencoder.json"):
        self.filepath = Path(filepath)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(metrics, f, indent=2)
