from ray.util.ray_lightning import RayPlugin

global OPENRIR_FOLDER
OPENRIR_FOLDER = ""
import os
import sys
import logging
import math
from filelock import FileLock
import random
import multiprocessing
from time import sleep, perf_counter as pc
from hyperpyyaml import load_hyperpyyaml

# __import_lightning_begin__
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from torchvision import transforms
# __import_lightning_end__

# __import_tune_begin__
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
# __import_tune_end__

from EcapaTdnnLightningModule import EcapaTdnnLightningModule
from VoxCelebLightningDataModule import VoxCelebLightningDataModule
from utils import parse_arguments

logger = logging.getLogger('App')
logging.basicConfig(level=logging.INFO)


def train_tune_checkpoint(
        hparams,
        checkpoint_dir=None,
        # no really used, we stop after every epoch and let tune decide what to do
        num_epochs=999,
        num_gpus=0
):
    progress_bar = pl.callbacks.progress.TQDMProgressBar(refresh_rate=25)

    data = VoxCelebLightningDataModule(hparams)

    logger.info(f"Speakers found {data.get_label_count()}")

    ray_plugin = RayPlugin(num_workers=1, num_cpus_per_worker=1, use_gpu=True)

    trainer = Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        # gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        num_sanity_val_steps=0,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "ptl/val_loss",
                },
                filename="checkpoint",
                on="validation_end"
            ),
            progress_bar
        ],
        plugins=[ray_plugin]
    )

    if checkpoint_dir:
        model = EcapaTdnnLightningModule.load_from_checkpoint(os.path.join(checkpoint_dir, "checkpoint"),
                                                              hparams=hparams, out_neurons=data.get_label_count())
        logger.info('Lightning loaded from checkpoint')
    else:
        model = EcapaTdnnLightningModule(hparams=hparams, out_neurons=data.get_label_count())
        logger.info('Lightning initialized')

    trainer.fit(model, data)


def tune_pbt(num_samples=15, training_iteration=15, cpus_per_trial=1, gpus_per_trial=0, conf=None):

    def explore(conf):
        # calculate new magnitudes for augmentations
        augmentations = []
        for tfn_name in conf['augmentation_functions']:
            augmentations.append((tfn_name, random.random()))

        conf["augmentations"] = augmentations
        return conf

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        # Models will be considered for perturbation at this interval of time_attr="time_total_s"
        perturbation_interval=1,
        custom_explore_fn=explore,
        log_config=True,
        require_attrs=True,
        quantile_fraction=conf['population_quantile_fraction'] # % of top performes used
    )

    progress_reporter = CLIReporter(
        # overwrite=True,
        parameter_columns=["augmentations"],
        metric_columns=["ptl/val_loss", "training_iteration"]
    )

    # set up the augmentations
    # tuple of augmentation name and its magnitude
    def initial_augmentations(spec):
        augmentations = []
        for tfn_name in conf['augmentation_functions']:
            augmentations.append((tfn_name, random.random()))
        return augmentations

    conf['augmentations'] = tune.sample_from(lambda spec: initial_augmentations(spec))

    analysis = tune.run(
        tune.with_parameters(
            train_tune_checkpoint,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=conf,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=progress_reporter,
        verbose=1,
        name="voxceleb-pbt",
        stop={  # Stop a single trial if one of the conditions are met
            "training_iteration": training_iteration},
        local_dir="./data",
    )

    print("Best hyperparams found were: ", analysis.best_config)
    return analysis


def main():

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logger.info("Starting...")

    global OPENRIR_FOLDER

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

    OPENRIR_FOLDER = hparams["openrir_folder"]

    cpu_count = multiprocessing.cpu_count()
    gpu_count = torch.cuda.device_count()

    print(f"CPUs {cpu_count} GPUs {gpu_count}")

    start_time = pc()

    analysis = tune_pbt(
        num_samples=hparams['population_size'],
        training_iteration=hparams['max_training_iterations'],
        cpus_per_trial=cpu_count/hparams['resources_per_trial_cpu_divider'],
        gpus_per_trial=gpu_count/hparams['resources_per_trial_gpu_divider'],
        conf=hparams)

    analysis.best_config
    analysis.results

    elapsed_time = pc() - start_time

    print(f"Time spent on training: {elapsed_time}")


if __name__ == '__main__':
    main()
