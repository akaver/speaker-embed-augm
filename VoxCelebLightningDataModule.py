import pytorch_lightning as pl
from torch.utils.data import DataLoader

from VoxCelebDataset import VoxCelebDataset


class VoxCelebLightningDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()

        self.batch_size = hparams["batch_size"]
        self.num_workers = hparams["num_workers"]

        # assign to use in dataloaders

        self.train_dataset = VoxCelebDataset(hparams, hparams["train_data"])
        self.val_dataset = VoxCelebDataset(hparams, hparams["valid_data"])
        self.test_dataset = VoxCelebDataset(hparams, hparams["test_data"])
        self.enrol_dataset = VoxCelebDataset(hparams, hparams["enrol_data"])

    """
    # Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.
    def prepare_data(self):
        pass

    # There are also data operations you might want to perform on every GPU
    def setup(self, stage = None):
        pass
    """

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True)

    def enrol_dataloader(self):
        return DataLoader(self.enrol_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True)

    """
    def test_dataloader(self):
        pass

    # ow you want to move an arbitrary batch to a device. on single gpu only
    def transfer_batch_to_device(self, batch, device):
        pass

    # alter or apply augmentations to your batch before it is transferred to the device.
    def on_before_batch_transfer(self, batch, dataloader_idx):
        pass

    #  alter or apply augmentations to your batch after it is transferred to the device.
    def on_after_batch_transfer(self, batch, dataloader_idx):
        pass

    #  can be used to clean up the state. It is also called from every process
    def teardown(self, stage = None):
        pass
    """

    def get_label_count(self):
        return self.train_dataset.get_label_count()
