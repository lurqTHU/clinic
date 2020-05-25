import os
import sys
import datetime
import logging


def setup_logger(name, save_dir, experiment_name,
                 trial_num=0, save_log=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_log:
        now_time = datetime.datetime.now()
        now_time = str(now_time)[:19]
        log_name = experiment_name + '_' + 'trial_{}_'.format(trial_num) \
                   + now_time + '_log.txt'
        log_path = os.path.join(save_dir, log_name)

        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        log_path = None

    return logger, log_path
