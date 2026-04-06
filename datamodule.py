from torch.utils.data import DataLoader, Subset
import lightning as L

from dataset import make_splits


class ECGDataModule(L.LightningDataModule):
    """LightningDataModule for the ECG signal dataset (Stage 1)."""

    def __init__(
        self,
        csv_path: str = "data/train.csv",
        data_dir: str = "data/train",
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._train: Subset | None = None
        self._val: Subset | None = None
        self._test: Subset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return  # already prepared

        train_sub, val_sub, test_sub = make_splits(
            csv_path=self.hparams.csv_path,
            data_dir=self.hparams.data_dir,
            val_size=self.hparams.val_size,
            test_size=self.hparams.test_size,
            random_state=self.hparams.random_state,
        )
        self._train = train_sub
        self._val = val_sub
        self._test = test_sub

    def _loader(self, subset: Subset, shuffle: bool) -> DataLoader:
        return DataLoader(
            subset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test, shuffle=False)

