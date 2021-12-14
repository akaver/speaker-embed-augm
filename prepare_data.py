import sys
import os
import logging
import glob
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
# import torchaudio
from tqdm.contrib import tqdm
import utils
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)


def prepare_voxceleb(hparams):
    """
    Prepares the csv files for the Voxceleb1 or Voxceleb2 datasets.
    Please follow the instructions in the README.md file for
    preparing Voxceleb 1 and 2 datasets (download, convert, etc).
    """

    # validate params we need, set defaults when needed
    if "skip_prep" in hparams and hparams["skip_prep"]:
        logger.info("Skiping dataset preparation by request!")
        return
    if "data_folder" not in hparams:
        sys.exit("data_folder is missing in params")
    data_folder = hparams["data_folder"]

    splits = []
    if "splits" in hparams:
        splits = hparams["splits"]

    if not os.path.exists(data_folder):
        logger.info(f"Data folder not found, creating dir '{data_folder}'")
        os.makedirs(data_folder)
        if not os.path.exists(data_folder):
            logger.error(f"Data folder ({data_folder}) creation failed, returning.")
            return

    logger.info("Creating csv file for the VoxCeleb Dataset..")

    # Split data into train and validation (verification split)
    wav_lst_train, wav_lst_dev = _get_utterance_split_lists(hparams)

    # Creating csv file for training data
    if "train" in splits:
        if "train_data" not in hparams:
            sys.exit("train_data is missing in params")
        prepare_csv(wav_lst_train, hparams["train_data"], hparams)

    if "dev" in splits:
        if "valid_data" not in hparams:
            sys.exit("valid_data is missing in params")
        prepare_csv(wav_lst_dev, hparams["valid_data"], hparams)

    if "test" in splits:
        prepare_csv_enrol_test(hparams)

    return


# Used for verification split
def _get_utterance_split_lists(hparams):
    """
    Tot. number of speakers vox1= 1211.
    Tot. number of speakers vox2= 5994.
    Splits the audio file list into train and dev.
    This function automatically removes verification test files from the training and dev set (if any).
    """

    # validate params we need, set defaults when needed
    if "data_folder_voxceleb1" not in hparams:
        sys.exit("data_folder_voxceleb1 is missing in params")
    if "data_folder_voxceleb2" not in hparams:
        sys.exit("data_folder_voxceleb2 is missing in params")
    data_folder_voxceleb1 = hparams["data_folder_voxceleb1"]
    data_folder_voxceleb2 = hparams["data_folder_voxceleb2"]

    test_pairs_in_file = None
    if "test_pairs_in_file" in hparams:
        test_pairs_in_file = hparams["test_pairs_in_file"]

    voxceleb_file_info = hparams["data_folder"] + "/file_info.pkl"

    if "voxceleb_file_info" in hparams:
        voxceleb_file_info = hparams["voxceleb_file_info"]

    speaker_quantity = None
    if "speaker_quantity" in hparams:
        speaker_quantity = hparams["speaker_quantity"]

    split_ratio = 90
    if "split_ratio" in hparams:
        split_ratio = hparams["split_ratio"]

    train_lst = []
    dev_lst = []

    data_folders = [
        data_folder_voxceleb1 + '/dev/', data_folder_voxceleb1 + '/test/',
        data_folder_voxceleb2 + '/dev/', data_folder_voxceleb2 + '/test/'
    ]

    test_speakers = []
    if test_pairs_in_file is not None:
        logger.info("Loading test file list...")
        test_lst = []
        for line in open(test_pairs_in_file):
            items = line.rstrip("\n").split(" ")
            test_lst.append(items[1])
            test_lst.append(items[2])
        test_lst = set(sorted(test_lst))
        test_speakers = [snt.split("/")[0] for snt in test_lst]
        test_speakers = set(sorted(test_speakers))
        logger.info(f"Unique test speakers (will be excluded from train/dev data): {len(test_speakers)}")

    file_list = []
    # do we already have the info in pickle?
    if not os.path.exists(voxceleb_file_info):
        # get all the files from all the locations
        for data_folder in data_folders:
            path = os.path.join(data_folder, "wav", "**", "*.wav")
            files_in_data_folder = glob.glob(path, recursive=True)
            logger.info(f"{path} contains {len(files_in_data_folder)} wav files")
            file_list.extend(files_in_data_folder)
        # save the file list into pickle
        logger.info(f"Saving file list to {voxceleb_file_info}")
        utils.save_pkl(file_list, voxceleb_file_info)
    else:
        logger.info(f"Loading file list from {voxceleb_file_info}")
        file_list = utils.load_pkl(voxceleb_file_info)

    logger.info(f"Total {len(file_list)} wav audio files")

    audio_files_dict = {}
    for f in file_list:
        spk_id = f.split("/wav/")[1].split("/")[0]
        if spk_id not in test_speakers:
            audio_files_dict.setdefault(spk_id, []).append(f)

    spk_id_list = list(audio_files_dict.keys())
    random.shuffle(spk_id_list)

    if speaker_quantity is not None:
        logger.warning(f"Using only {speaker_quantity} speakers out of {len(spk_id_list)}")
        spk_id_list = random.sample(spk_id_list, speaker_quantity)

    logger.info(f"Unique speakers found (excluding test speakers) {len(spk_id_list)}")

    full_lst = []
    for spk_id in spk_id_list:
        full_lst.extend(audio_files_dict[spk_id])
    logger.info(f"Audio samples {len(full_lst)}")

    test_size_split = 1 - 0.01 * split_ratio
    train_lst, dev_lst = train_test_split(full_lst, test_size=test_size_split, shuffle=True)
    """
    split = int(0.01 * split_ratio[0] * len(spk_id_list))
    for spk_id in spk_id_list[:split]:
        train_lst.extend(audio_files_dict[spk_id])

    for spk_id in spk_id_list[split:]:
        dev_lst.extend(audio_files_dict[spk_id])
    """

    logger.info(f"Train list {len(train_lst)}, dev list {len(dev_lst)}")

    return train_lst, dev_lst


def _get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]

    return chunk_lst


def prepare_csv(wav_lst, csv_file, hparams):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The output csv file

    Returns
    -------
    None
    """

    # validate params we need, set defaults when needed
    sample_rate = 16000
    if "sample_rate" in hparams:
        sample_rate = hparams["sample_rate"]

    sentence_len = 3.0
    if "sentence_len" in hparams:
        sentence_len = hparams["sentence_len"]

    amp_th = 5e-04
    if "amp_th" in hparams:
        amp_th = hparams["amp_th"]

    logger.info(f"Creating csv list: {csv_file}")

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    # For assigning unique ID to each chunk
    my_sep = "--"
    entry = []
    # Processing all the wav files in the list
    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        try:
            [spk_id, sess_id, utt_id] = wav_file.split("/")[-3:]
        except ValueError:
            logger.info(f"Malformed path: {wav_file}")
            continue
        audio_id = my_sep.join([spk_id, sess_id, utt_id.split(".")[0]])

        # Reading the signal (to retrieve duration in seconds)
        signal, f_sample_rate = utils.load_audio(wav_file)
        signal = signal.squeeze(0)

        audio_duration = signal.shape[0] / sample_rate

        uniq_chunks_list = _get_chunks(sentence_len, audio_id, audio_duration)
        for chunk in uniq_chunks_list:
            s, e = chunk.split("_")[-2:]
            start_sample = int(float(s) * sample_rate)
            end_sample = int(float(e) * sample_rate)

            #  Avoid chunks with very small energy
            mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
            if mean_sig < amp_th:
                continue

            # Composition of the csv_line
            csv_line = [
                chunk,
                str(audio_duration),
                wav_file,
                start_sample,
                end_sample,
                spk_id,
            ]
            entry.append(csv_line)

    csv_output = csv_output + entry
    utils.write_csv_file(csv_file, csv_output)

"""
      data_folder_voxceleb1, data_folder_voxceleb2, save_folder,
        test_pairs_in_file, test_pairs_out_file,
        save_enrol_csv, save_test_csv, test_pairs_quantity=None
"""
def prepare_csv_enrol_test(hparams):
    """
    Creates the csv file for test data (useful for verification)

    Arguments
    ---------
    data_folder : str
        Path of the data folders
    save_folder : str
        The directory where to store the csv files.

    Returns
    -------
    None
    """

    # validate params we need, set defaults when needed
    if "test_pairs_in_file" not in hparams:
        sys.exit("test_pairs_in_file is missing in params")
    test_pairs_in_file = hparams["test_pairs_in_file"]

    if "test_pairs_out_file" not in hparams:
        sys.exit("test_pairs_out_file is missing in params")
    test_pairs_out_file = hparams["test_pairs_out_file"]

    sample_rate = 16000
    if "sample_rate" in hparams:
        sample_rate = hparams["sample_rate"]

    test_pairs_quantity = None
    if "test_pairs_quantity" in hparams:
        test_pairs_quantity = hparams["test_pairs_quantity"]


    if "enrol_data" not in hparams:
        sys.exit("enrol_data is missing in params")
    save_enrol_csv = hparams["enrol_data"]

    if "test_data" not in hparams:
        sys.exit("test_data is missing in params")
    save_test_csv = hparams["test_data"]

    data_folder_voxceleb1 = hparams["data_folder_voxceleb1"]
    data_folder_voxceleb2 = hparams["data_folder_voxceleb2"]

    data_folders = [
        data_folder_voxceleb1 + '/dev/', data_folder_voxceleb1 + '/test/',
        data_folder_voxceleb2 + '/dev/', data_folder_voxceleb2 + '/test/'
    ]

    with open(test_pairs_in_file) as fin:
        test_pairs_lines = fin.readlines()
        logger.info(f"Initial test pairs {len(test_pairs_lines)}")

    if test_pairs_quantity is not None and test_pairs_quantity < len(test_pairs_lines):
        test_pairs_lines = random.sample(test_pairs_lines, test_pairs_quantity)

    with open(test_pairs_out_file, "w") as output:
        output.writelines(test_pairs_lines)

    logger.info(f"Final test pairs {len(test_pairs_lines)}")

    csv_output_head = [
        ["ID", "duration", "wav", "start", "stop", "spk_id"]
    ]  # noqa E231

    # extract all the enrol and test ids
    enrol_ids, test_ids = [], []

    # Get unique ids (enrol and test utterances)
    for line in tqdm(test_pairs_lines):
        splits = line.split(" ")
        e_id = splits[1].rstrip().split(".")[0].strip()
        t_id = splits[2].rstrip().split(".")[0].strip()
        enrol_ids.append(e_id)
        test_ids.append(t_id)

    enrol_ids = list(np.unique(np.array(enrol_ids)))
    test_ids = list(np.unique(np.array(test_ids)))

    enrol_csv = []
    test_csv = []
    logger.info("preparing enrol/test csvs")

    for data_folder in data_folders:
        logger.info(f"Using data from {data_folder}")
        # Prepare enrol csv
        for id in tqdm(enrol_ids):
            wav = data_folder + "wav/" + id + ".wav"

            if not os.path.exists(wav):
                continue

            # Reading the signal (to retrieve duration in seconds)
            signal, f_sample_rate = utils.load_audio(wav)

            # Returns a tensor with all the dimensions of input of size 1 removed.
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / sample_rate
            start_sample = 0
            stop_sample = signal.shape[0]
            [spk_id, sess_id, utt_id] = wav.split("/")[-3:]

            csv_line = [
                id,
                audio_duration,
                wav,
                start_sample,
                stop_sample,
                spk_id,
            ]

            enrol_csv.append(csv_line)

        # Prepare test csv
        for id in tqdm(test_ids):
            wav = data_folder + "wav/" + id + ".wav"

            if not os.path.exists(wav):
                continue

            # Reading the signal (to retrieve duration in seconds)
            signal, f_sample_rate = utils.load_audio(wav)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / sample_rate
            start_sample = 0
            stop_sample = signal.shape[0]
            [spk_id, sess_id, utt_id] = wav.split("/")[-3:]

            csv_line = [
                id,
                audio_duration,
                wav,
                start_sample,
                stop_sample,
                spk_id,
            ]

            test_csv.append(csv_line)

    csv_output = csv_output_head + enrol_csv
    csv_file = save_enrol_csv
    # Writing the csv lines
    utils.write_csv_file(csv_file, csv_output)

    csv_output = csv_output_head + test_csv
    csv_file = save_test_csv
    # Writing the csv lines
    utils.write_csv_file(csv_file, csv_output)


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.info("Starting...")

    logging.info(f"Command line args: {sys.argv[1:]}")

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = utils.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparam_str = fin.read()

    if 'yaml' in run_opts:
        for yaml_file in run_opts['yaml'][0]:
            logging.info(f"Loading additional yaml file: {yaml_file}")
            with open(yaml_file) as fin:
                hparam_str = hparam_str + "\n" + fin.read();

    hparams = load_hyperpyyaml(hparam_str, overrides)

    logging.info(f"Params: {hparams}")

    prepare_voxceleb(hparams)


if __name__ == "__main__":
    main()
