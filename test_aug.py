import os
import sys
import torch
import logging
import numpy as np

from EcapaTdnnLightningModule import EcapaTdnnLightningModule
from VoxCelebLightningDataModule import VoxCelebLightningDataModule
from utils import parse_arguments
from hyperpyyaml import load_hyperpyyaml
import ECAPA_TDNN
from tqdm import tqdm
import utils

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
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


if __name__ == "__main__":
    main()
