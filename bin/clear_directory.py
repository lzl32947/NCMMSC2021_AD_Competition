import os
from typing import Dict
from tqdm import tqdm

from util.tools.files_util import set_working_dir, read_config


def clear_output(configs: Dict, **kwargs) -> None:
    """
    Clear the weight directory
    :param configs: Dict, the configs
    :param kwargs: Dict, other necessary parameters
    :return: None
    """
    # Unless otherwise given, the weight directory is in the config['output_dir']
    output_dir = kwargs['output_dir'] if 'output_dir' in kwargs.keys() else configs['output_dir']
    sub_directories = os.listdir(output_dir)
    # List the weight directory
    if len(sub_directories) > 0:
        # Generate the tqdm bar
        bar = tqdm(range(len(sub_directories)))
        bar.set_description("Removing the unnecessary outputs")
        delete_count = 0
        # Running into the sub directory
        for output_dirs in sub_directories:
            sub_directory = os.path.join(output_dir, output_dirs)
            if len(os.listdir(sub_directory)) == 0:
                # The dir is empty
                os.rmdir(sub_directory)
                delete_count += 1
            # Update the bar
            bar.set_postfix(deleted=delete_count)
            bar.update(1)
        bar.close()
    else:
        print("No outputs to be deleted")


def clear_image(configs: Dict, **kwargs) -> None:
    """
    Clear the weight directory
    :param configs: Dict, the configs
    :param kwargs: Dict, other necessary parameters
    :return: None
    """
    # Unless otherwise given, the weight directory is in the config['image_dir']
    image_dir = kwargs['image_dir'] if 'image_dir' in kwargs.keys() else configs['image_dir']
    sub_directories = os.listdir(image_dir)
    # List the weight directory
    if len(sub_directories) > 0:
        # Generate the tqdm bar
        bar = tqdm(range(len(sub_directories)))
        bar.set_description("Removing the unnecessary images")
        delete_count = 0
        # Running into the sub directory
        for image_dirs in sub_directories:
            sub_directory = os.path.join(image_dir, image_dirs)
            if len(os.listdir(sub_directory)) == 0:
                # The dir is empty
                os.rmdir(sub_directory)
                delete_count += 1
            # Update the bar
            bar.set_postfix(deleted=delete_count)
            bar.update(1)
        bar.close()
    else:
        print("No images to be deleted")


def clear_weight(configs: Dict, **kwargs) -> None:
    """
    Clear the weight directory
    :param configs: Dict, the configs
    :param kwargs: Dict, other necessary parameters
    :return: None
    """
    # Unless otherwise given, the weight directory is in the config['weight_dir']
    weight_dir = kwargs['weight_dir'] if 'weight_dir' in kwargs.keys() else configs['weight_dir']
    sub_directories = os.listdir(weight_dir)
    # List the weight directory
    if len(sub_directories) > 0:
        # Generate the tqdm bar
        bar = tqdm(range(len(sub_directories)))
        bar.set_description("Removing the unnecessary weights")
        delete_count = 0
        # Running into the sub directory
        for weight_dirs in sub_directories:
            sub_directory = os.path.join(weight_dir, weight_dirs)
            if len(os.listdir(sub_directory)) == 0:
                # The dir is empty
                os.rmdir(sub_directory)
                delete_count += 1
            else:
                # Running into the specific directory
                for subs in os.listdir(sub_directory):
                    sub_path = os.path.join(sub_directory, subs)
                    # The specific directory is empty
                    if len(os.listdir(sub_path)) == 0:
                        # Delete the directory
                        os.rmdir(sub_path)
                # If the directory is all deleted
                if len(os.listdir(sub_directory)) == 0:
                    # Remove the directory
                    os.rmdir(sub_directory)
                    delete_count += 1
            # Update the bar
            bar.set_postfix(deleted=delete_count)
            bar.update(1)
        bar.close()
    else:
        print("No weights to be deleted")


def clear_log(configs: Dict, clear_limit: int = 20, **kwargs) -> None:
    """
    Clear the log directory
    :param configs: Dict, the configs
    :param clear_limit: int, the length of lines, below which the log files will be removed
    :param kwargs: Dict, other necessary parameters
    :return: None
    """
    # Unless otherwise given, the log directory is in the config['log_dir']
    log_dir = kwargs['log_dir'] if 'log_dir' in kwargs.keys() else configs['log_dir']
    # List the log directory
    sub_directories = os.listdir(log_dir)
    if len(sub_directories) > 0:
        # Generate the tqdm bar
        bar = tqdm(range(len(sub_directories)))
        bar.set_description("Removing the unnecessary logs")
        delete_count = 0
        # Running into the sub directory
        for log_dirs in sub_directories:
            sub_directory = os.path.join(log_dir, log_dirs)
            # Delete if empty
            if len(os.listdir(sub_directory)) == 0:
                os.rmdir(sub_directory)
                delete_count += 1
                bar.update(1)
                continue
            # Get the path to the log file
            log_file = os.path.join(sub_directory, "run.log")
            delete_flag = False
            with open(log_file, "r", encoding="utf-8") as login:
                lines = login.readlines()
                # Mark the short files
                if len(lines) < clear_limit:
                    delete_flag = True
            if delete_flag:
                delete_count += 1
                # Remove the configs if exist
                for file in os.listdir(sub_directory):
                    os.remove(os.path.join(sub_directory, file))
                # Remove directory
                os.rmdir(sub_directory)
            # Update the bar
            bar.set_postfix(deleted=delete_count)
            bar.update(1)
        bar.close()
    else:
        print("No logs to be deleted")


if __name__ == '__main__':
    """
    This runnable is to clear the useless files in log directory and weight directory, mainly for early-stop or catch up
    an error.
    """
    set_working_dir("./..")
    config = read_config(os.path.join("configs", "config.yaml"))
    clear_log(config['log'])
    clear_weight(config['weight'])
    clear_image(config['image'])
    clear_output(config['output'])
