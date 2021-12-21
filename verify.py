import sys
# needed in singularity
# sys.path.insert(0, '/opt/project')

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


def compute_embedding_loop(data_loader, model, device):
    """Computes the embeddings of all the unique waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    for batch in tqdm(data_loader, dynamic_ncols=True):
        wavs, labels, ids = batch
        wavs = wavs.to(device)
        embeddings = model.get_embeddings(wavs).unsqueeze(1)
        for i, _id in enumerate(ids):
            embedding_dict[_id] = embeddings[i].detach().clone()

    return embedding_dict


def get_verification_scores(hparams, veri_test, enrol_dict, test_dict, train_dict=None):
    """ Computes positive and negative scores given the verification split.
    """
    scores = []
    positive_scores = []
    negative_scores = []
    full_scores = []

    save_file = os.path.join(hparams["data_folder"], "scores.txt")
    if os.path.exists(save_file):
        s_file = open(save_file, "w")
    else:
        s_file = open(save_file, "x")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # creating cohort for score normalization
    if "score_norm" in hparams:
        train_cohort = torch.stack(list(train_dict.values()))

    for i, line in enumerate(tqdm(veri_test)):
        # Reading verification file (enrol_file test_file label)
        lab_pair = int(line.split(" ")[0].rstrip().split(".")[0].strip())
        enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
        test_id = line.split(" ")[2].rstrip().split(".")[0].strip()

        # logger.info(f"lab_pair {str(lab_pair)} enrol {enrol_id} - test {test_id}")

        enrol = enrol_dict[enrol_id]
        test = test_dict[test_id]

        if "score_norm" in hparams:
            # Getting norm stats for enrol impostors
            enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
            score_e_c = similarity(enrol_rep, train_cohort)

            if "cohort_size" in hparams:
                score_e_c = torch.topk(
                    score_e_c, k=hparams["cohort_size"], dim=0
                )[0]

            mean_e_c = torch.mean(score_e_c, dim=0)
            std_e_c = torch.std(score_e_c, dim=0)

            # Getting norm stats for test impostors
            test_rep = test.repeat(train_cohort.shape[0], 1, 1)
            score_t_c = similarity(test_rep, train_cohort)

            if "cohort_size" in hparams:
                score_t_c = torch.topk(
                    score_t_c, k=hparams["cohort_size"], dim=0
                )[0]

            mean_t_c = torch.mean(score_t_c, dim=0)
            std_t_c = torch.std(score_t_c, dim=0)

        # Compute the score for the given sentence
        score = similarity(enrol, test)[0]

        # Perform score normalization
        if "score_norm" in hparams:
            if hparams["score_norm"] == "z-norm":
                score = (score - mean_e_c) / std_e_c
            elif hparams["score_norm"] == "t-norm":
                score = (score - mean_t_c) / std_t_c
            elif hparams["score_norm"] == "s-norm":
                score_e = (score - mean_e_c) / std_e_c
                score_t = (score - mean_t_c) / std_t_c
                score = 0.5 * (score_e + score_t)

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score))
        scores.append(score)

        full_scores.append((score.item(), lab_pair))

        if lab_pair == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    s_file.close()
    return positive_scores, negative_scores, full_scores


def EER2(scores):
    scores = sorted(scores)  # min->max
    sort_score = np.matrix(scores)
    minIndex = 9223372036854775807
    minDis = 9223372036854775807
    minTh = 9223372036854775807
    alltrue = sort_score.sum(0)[0, 1]
    allfalse = len(scores) - alltrue
    eer = 9223372036854775807
    fa = allfalse
    miss = 0

    for i in range(0, len(scores)):
        # min -> max
        if sort_score[i, 1] == 1:
            miss += 1
        else:
            fa -= 1

        fa_rate = float(fa) / allfalse
        miss_rate = float(miss) / alltrue

        if abs(fa_rate - miss_rate) < minDis:
            minDis = abs(fa_rate - miss_rate)
            eer = max(fa_rate, miss_rate)
            minIndex = i
            minTh = sort_score[i, 0]

    return eer, minTh



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

    data = VoxCelebLightningDataModule(hparams)

    # load the model
    model = EcapaTdnnLightningModule.load_from_checkpoint(hparams["verify_model_checkpoint"], hparams=hparams, out_neurons=data.get_label_count())

    model.to(hparams['device'])

    # do not backpropagate
    model.eval()

    # model.freeze()

    save_enrol_dict = hparams["data_folder"] + "/enrol_dict.pkl"
    save_test_dict = hparams["data_folder"] + "/test_dict.pkl"
    save_train_dict = hparams["data_folder"] + "/train_dict.pkl"
    enrol_dict = {}
    test_dict = {}
    train_dict = {}

    # compute the embeddings

    if not os.path.exists(save_enrol_dict):
        logger.info("Enrol embeddings dict")

        enrol_dict = compute_embedding_loop(data.enrol_dataloader(), model, hparams['device'])
        test_dict = compute_embedding_loop(data.test_dataloader(), model, hparams['device'])

        # Second run (normalization stats are more stable)
        enrol_dict = compute_embedding_loop(data.enrol_dataloader(), model, hparams['device'])
        test_dict = compute_embedding_loop(data.test_dataloader(), model, hparams['device'])

        logger.info("Saving enrol embeddings dict")
        utils.save_pkl(enrol_dict, save_enrol_dict)
    else:
        logger.info("Loading enrol_dict")
        enrol_dict = utils.load_pkl(save_enrol_dict)

    if not os.path.exists(save_test_dict):
        logger.info("Test embeddings dict")
        test_dict = compute_embedding_loop(data.test_dataloader(), model, hparams['device'])
        logger.info("Saving test embeddings dict")
        utils.save_pkl(test_dict, save_test_dict)
    else:
        logger.info("Loading test_dict")
        test_dict = utils.load_pkl(save_test_dict)

    if "score_norm" in hparams:
        if not os.path.exists(save_train_dict):
            logger.info("Train embeddings dict")
            train_dict = compute_embedding_loop(data.train_dataloader(), model, hparams['device'])
            logger.info("Saving train embeddings dict")
            utils.save_pkl(train_dict, save_train_dict)
        else:
            logger.info("Loading train_dict")
            train_dict = utils.load_pkl(save_train_dict)

    logger.info("Computing EER..")
    with open(hparams["verification_file"]) as f:
        veri_test = [line.rstrip() for line in f]
    logger.info(f"Test pairs {len(veri_test)}")

    positive_scores, negative_scores, full_scores = get_verification_scores(hparams, veri_test, enrol_dict, test_dict, train_dict)

    # get rid of all the dicts from memory
    del enrol_dict, test_dict, train_dict

    # eer, th = ECAPA_TDNN.EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    # logger.info("EER(%%)=%f", eer * 100)


    eer2, th = EER2(full_scores)
    logger.info("EER2(%%)=%f", eer2 * 100)

    #min_dcf, th = ECAPA_TDNN.minDCF(
    #    torch.tensor(positive_scores), torch.tensor(negative_scores)
    #)
    #logger.info("minDCF=%f", min_dcf * 100)


if __name__ == "__main__":
    main()
