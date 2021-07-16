import os
from typing import Dict
from tqdm import tqdm

from util.tools.files_util import set_working_dir, read_config


def clear_weight(configs: Dict, **kwargs):
    weight_dir = kwargs['weight_dir'] if 'weight_dir' in kwargs.keys() else configs['weight_dir']
    sub_directories = os.listdir(weight_dir)
    if len(sub_directories) > 0:
        bar = tqdm(range(len(sub_directories)))
        bar.set_description("Removing the unnecessary weights")
        delete_count = 0
        for weight_dirs in sub_directories:
            sub_directory = os.path.join(weight_dir, weight_dirs)
            if len(os.listdir(sub_directory)) == 0:
                os.rmdir(sub_directory)
                delete_count += 1
            else:
                for subs in os.listdir(sub_directory):
                    sub_path = os.path.join(sub_directory, subs)
                    if len(os.listdir(sub_path)) == 0:
                        os.rmdir(sub_path)
                if len(os.listdir(sub_directory)) == 0:
                    os.rmdir(sub_directory)
                    delete_count += 1
            bar.set_postfix(deleted=delete_count)
            bar.update(1)
        bar.close()
    else:
        print("No weights to be deleted")


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
                lines = login.readlines()
                if len(lines) < clear_limit:
                    delete_flag = True
            if delete_flag:
                delete_count += 1
                if os.path.exists(os.path.join(sub_directory, "config.yaml")):
                    os.remove(os.path.join(sub_directory, "config.yaml"))
                os.remove(log_file)
                os.rmdir(sub_directory)
            bar.set_postfix(deleted=delete_count)
            bar.update(1)
        bar.close()
    else:
        print("No logs to be deleted")


if __name__ == '__main__':
    set_working_dir("./..")
    config = read_config(os.path.join("configs", "config.yaml"))
    clear_log(config['log'])
    clear_weight(config['weight'])
