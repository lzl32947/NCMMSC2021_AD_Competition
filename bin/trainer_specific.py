from configs.types import AudioFeatures
from network.melspec.melspec import MelSpecModel
from network.melspec.lstm.lstm import MelSpecModel_lstm
from network.melspec.lstm.rnn import RnnModel
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init
from util.train_util.trainer_util import prepare_feature, prepare_dataloader, read_weight, get_best_acc_weight, \
    train_general, train_specific_feature

sequence_length = 157  # 序列长度，将图像的每一列作为一个序列
input_size = 128  # 输入数据的维度
hidden_size = 300  # 隐藏层的size
num_layers = 2  # 有多少层

num_classes = 3
batch_size = 1
if __name__ == '__main__':
    """
    This is a template for joint-features training
    """
    # Init the global environment
    time_identifier, configs = global_init()
    logger = GlobalLogger().get_logger()
    use_features = prepare_feature(configs['features'])
    # Read the fold from config
    total_fold = configs['dataset']['k_fold']
    # Train the general model
    train_specific_feature(configs, time_identifier, AudioFeatures.MELSPECS, RnnModel)
