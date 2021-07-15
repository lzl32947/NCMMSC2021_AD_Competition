import os
from typing import Dict
from tqdm import tqdm

from util.tools.files_util import set_working_dir, read_config


def clear_log(configs: Dict, clear_limit: int = 20, **kwargs):
    log_dir = kwargs['log_dir'] if 'log_dir' in kwargs.keys() else configs['log_dir']
    sub_directories = os.listdir(log_dir)
    if len(sub_directories) > 0:
        bar = tqdm(range(len(sub_directories)))
        bar.set_description("Removing the unnecessary logs")
        delete_count = 0
        for log_dirs in sub_directories:
            sub_directory = os.path.join(log_dir, log_dirs)
            log_file = os.path.join(sub_directory, "run.log")
            delete_flag = False
            with open(log_file, "r", encoding="utf-8") as login:
                lines = login.readlines(clear_limit)
                if len(lines) < clear_limit:
                    delete_flag = True
            if delete_flag:
                delete_count += 1
                os.remove(log_file)
                os.rmdir(sub_directory)
            bar.set_postfix(deleted=delete_count)
            bar.update(1)
    else:
        print("No logs to be deleted")


if __name__ == '__main__':
    set_working_dir("./..")
    config = read_config(os.path.join("configs", "config.yaml"))
    clear_log(config['log'])
