from configs.types import AudioFeatures
from network.melspec.melspec import MelSpecModel
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init
from util.train_util.trainer_util import prepare_feature, prepare_dataloader, read_weight, get_best_acc_weight, \
    train_general, train_specific_feature

if __name__ == '__main__':
    time_identifier, configs = global_init()
    logger = GlobalLogger().get_logger()
    use_features = prepare_feature(configs['features'])

    total_fold = configs['dataset']['k_fold']
    model = MelSpecModel()
    train_specific_feature(configs, time_identifier, AudioFeatures.MELSPECS, model)