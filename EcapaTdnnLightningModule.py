import logging
from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import ECAPA_TDNN
import augmentations

logger = logging.getLogger(__name__)


class EcapaTdnnLightningModule(pl.LightningModule):
    def __init__(self, hparams, out_neurons):
        super().__init__()

        # naming conflict?
        self._hparams = hparams

        print(self._hparams)

        self.out_neurons = out_neurons

        # embedding size
        lin_neurons = 192

        self.compute_features = ECAPA_TDNN.Fbank(
            n_mels=hparams["n_mels"],
            left_frames=hparams["left_frames"], right_frames=hparams["right_frames"],
            deltas=hparams["deltas"])

        self.mean_var_norm = ECAPA_TDNN.InputNormalization(norm_type="sentence", std_norm=False)

        self.net = ECAPA_TDNN.ECAPA_TDNN(input_size=int(hparams["n_mels"]), lin_neurons=lin_neurons)

        # embedding in, speaker count out
        self.classifier = ECAPA_TDNN.Classifier(lin_neurons, out_neurons=out_neurons)

        self.compute_cost = ECAPA_TDNN.LogSoftmaxWrapper(loss_fn=ECAPA_TDNN.AdditiveAngularMargin(margin=0.2, scale=30))
        # not used
        self.compute_error = ECAPA_TDNN.classification_error
        self.stage = None
        self.n_augment = 1

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str]) -> None:
        logger.info(f"Setup stage {stage}")
        self.stage = stage

    # Use for inference only (separate from training_step)
    def forward(self, wavs):
        # wavs is batch of tensors with audio

        lengths = torch.ones(wavs.shape[0], device=self.device)

        wavs_aug_tot = [wavs]

        self.n_augment = 1

        if self.stage == "fit":
            if "augmentation_functions" in self._hparams:
                for _, augmentation in enumerate(self._hparams['augmentation_functions']):
                    wavs_augmented = getattr(augmentations, augmentation)(wavs, lengths)
                    # wavs_augmented = augmentation(wavs, lengths)

                    # Managing speed change - ie lenght of audio was changed. cut down or pad
                    if wavs_augmented.shape[1] > wavs.shape[1]:
                        wavs_augmented = wavs_augmented[:, 0: wavs.shape[1]]
                    elif wavs_augmented.shape[1] < wavs.shape[1]:
                        zero_sig = torch.zeros_like(wavs)
                        zero_sig[:, 0: wavs_augmented.shape[1]] = wavs_augmented
                        wavs_augmented = zero_sig

                    if "concat_augment" in self._hparams and self._hparams["concat_augment"]:
                        # collect the augmentations
                        wavs_aug_tot.append(wavs_augmented)
                    else:
                        # replace the original audio - ie next augmentation is applied on top of it
                        wavs = wavs_augmented
                        wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)

        lengths = torch.cat([lengths] * self.n_augment)

        # extract features - filterbanks from mfcc
        features = self.compute_features(wavs)

        # normalize
        features_normalized = self.mean_var_norm(features, lengths)

        embedding = self.net(features_normalized)
        prediction = self.classifier(embedding)

        return prediction, embedding

    def get_embeddings(self, wavs):
        _, embeddings = self.forward(wavs)
        return embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.000002)
        return optimizer
        pass

    def model_step(self, batch, batch_idx):
        inputs, labels, ids = batch
        labels_predicted, embedding = self(inputs)  # calls forward

        # multiply the labels by  appended augmentations count
        if self.stage == 'fit':
            labels = torch.cat([labels] * self.n_augment, dim=0)

        loss = self.compute_cost(labels_predicted, labels)

        labels_predicted_squeezed = labels_predicted.squeeze()
        labels_squeezed = labels.squeeze()

        labels_hot_encoded = F.one_hot(labels_squeezed.long(), labels_predicted_squeezed.shape[1])
        acc = accuracy(labels_predicted_squeezed, labels_hot_encoded)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.model_step(batch, batch_idx)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log("ptl/train_loss", loss, sync_dist=True)
        self.log("ptl/train_accuracy", acc, sync_dist=True)

        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self.model_step(batch, batch_idx)
        return {'loss': loss, 'acc': acc}

    def test_step(self, batch, batch_idx):
        # return Union[Tensor, Dict[str, Any], None]
        pass

    # Called at the end of the validation epoch with the outputs of all validation steps.
    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['acc'] for x in val_step_outputs]).mean()

        self.log('avg_val_loss', avg_val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('avg_val_acc', avg_val_acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # info for ray
        self.log("ptl/val_loss", avg_val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("ptl/val_accuracy", avg_val_acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # TODO, implement calculations on test set
        self.log("ptl/EER", avg_val_acc + 1, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("ptl/minDCF", avg_val_acc + 2, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        logger.info(f"avg_val_loss: {avg_val_loss} avg_val_acc: {avg_val_acc}")

        return {'val_loss': avg_val_loss}

    def on_train_start(self) -> None:
        print(f"on_train_start{self.stage}")
        if "augmentations" in self._hparams:
            print(self._hparams["augmentations"])
