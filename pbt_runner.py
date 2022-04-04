import time
import sys
import logging
import os
from subprocess import run

logger = logging.getLogger('RUN')
logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def main():
    logger.info("Starting...")

    # increase max error count allowed on pbt on every restart
    max_error_count = 0
    run_count = 0
    while True:
        logger.info(f'Starting run {run_count}')
        run_params = ['python3', 'pbt.py']
        run_params.extend([f'--max_error_count={max_error_count}'])
        run_params.extend(sys.argv[1:])
        logger.info(f"Run args: {run_params}")
        p = run(run_params)
        logger.info(f'Exit status code: {p.returncode}')
        logger.info(f'Sleeping for 1 seconds...')
        time.sleep(1.0)
        fix_status()
        logger.info(f'Resuming...')
        run_count += 1
        max_error_count += 1
        if max_error_count == 15:
            break


def fix_status():
    file_info = {}
    path = './data/voxceleb-pbt/'
    files = [f for f in os.listdir(path)]
    for f in files:
        if f.startswith("experiment"):
            info = os.stat(path + f)
            file_info[f] = info.st_mtime
    file_info = sorted(file_info.items(), key=lambda item: item[1])
    newest_file = file_info[-1]
    print("Fixing possible errors in: " + newest_file[0])
    file_str = open(path + newest_file[0], 'r').read()
    file_str = file_str.replace("TERMINATED","PENDING")
    file_str = file_str.replace("ERROR","PENDING")
    open(path + newest_file[0], 'w').write(file_str)


if __name__ == '__main__':
    main()
