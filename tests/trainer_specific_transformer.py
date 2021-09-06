from configs.types import AudioFeatures
from network.attention.attention_model import AttentionModule
from network.melspec.melspec import MelSpecModel
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init
from util.train_util.data_loader import AldsDataset1D
from util.train_util.trainer_util import prepare_feature, prepare_dataloader, read_weight, get_best_acc_weight, \
    train_general, train_specific_feature

if __name__ == '__main__':
    # Init the global environment
    time_identifier, configs = global_init("config_audio")
    logger = GlobalLogger().get_logger()
    use_features = prepare_feature(configs['features'])
    # Read the fold from config
    total_fold = configs['dataset']['k_fold']
    # Train the general model
    train_specific_feature(configs, time_identifier, AudioFeatures.MFCC, AldsDataset1D, AttentionModule,
                           input_shape=(4, 20, 157), block_num=6, num_heads=4)
