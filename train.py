import sys
import logging
import torch
from hyperpyyaml import load_hyperpyyaml
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from EcapaTdnnLightningModule import EcapaTdnnLightningModule
from VoxCelebLightningDataModule import VoxCelebLightningDataModule
from utils import parse_arguments

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logger.info("Starting...")

    # enable the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparam_str = fin.read()

    if 'yaml' in run_opts:
        for yaml_file in run_opts['yaml']:
            logging.info(f"Loading additional yaml file: {yaml_file[0]}")
            with open(yaml_file[0]) as fin:
                hparam_str = hparam_str + "\n" + fin.read();

    hparams = load_hyperpyyaml(hparam_str, overrides)

    logging.info(f"Params: {hparams}")

    # =============================== Pytorch Lightning ====================================

    data = VoxCelebLightningDataModule(hparams)

    logger.info(f"Speakers found {data.get_label_count()}")

    model = EcapaTdnnLightningModule(hparams, out_neurons=data.get_label_count())

    trainer = pl.Trainer(
        default_root_dir=hparams["data_folder"],
        gpus=-1 if torch.cuda.device_count() > 0 else 0,
        max_epochs=hparams["number_of_epochs"],
        # num_sanity_val_steps=0,
        strategy='ddp',
        # plugins=DDPPlugin(find_unused_parameters=False)
    )

    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == '__main__':
    main()
