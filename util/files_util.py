import os


def create_dir(target: str) -> bool:
    """
    Create the dir and return whether the dir is created.
    :param target: str, the path to dir.
    :return: bool, whether the dir is successfully created.
    """
    if os.path.exists(target):
        return True
    else:
        try:
            #
            os.mkdir(target)
            return True
        except:
            return False


def set_working_dir(target: str):
    """
    Set working dir to target.
    :param target: str, the path to target dir.
    :return: None
    """
    os.chdir(target)
