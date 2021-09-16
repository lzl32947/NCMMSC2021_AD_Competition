from typing import Tuple

from torch import nn
import torch
import torch.nn.functional as func

from model.base_model import BaseModel
from model.manager import Register, Registers


class DenseModel(nn.Module):
    def __init__(self, input_unit=6240):
        super().__init__()
        self.linear_1 = nn.Linear(input_unit, 64)
        self.dropout = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(64, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = input_tensor.view((batch_size, -1))

        output = self.linear_1(output)
        output = func.relu(output)
        output = self.dropout(output)

        output = self.linear_2(output)
        return output


class ExtractionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(1, 32, (3, 3))
        self.maxpooling_1 = nn.MaxPool2d((2, 4), stride=2)
        self.batchnorm_1 = nn.BatchNorm2d(32)

        self.conv_layer_2 = nn.Conv2d(32, 32, (3, 3))
        self.maxpooling_2 = nn.MaxPool2d((4, 4), stride=2)
        self.batchnorm_2 = nn.BatchNorm2d(32)

        self.conv_layer_3 = nn.Conv2d(32, 32, (3, 3))
        self.maxpooling_3 = nn.MaxPool2d((2, 5), stride=2)
        self.batchnorm_3 = nn.BatchNorm2d(32)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output)
        output = self.maxpooling_1(output)
        output = self.batchnorm_1(output)

        output = self.conv_layer_2(output)
        output = func.relu(output)
        output = self.maxpooling_2(output)
        output = self.batchnorm_2(output)

        output = self.conv_layer_3(output)
        output = func.relu(output)
        output = self.maxpooling_3(output)
        output = self.batchnorm_3(output)

        return output


class ConcatModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(96, 256, (3, 3))

        self.conv_layer_2 = nn.Conv2d(256, 512, (3, 3))
        self.maxpooling_2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_layer_3 = nn.Conv2d(512, 1024, (3, 3))
        self.maxpooling_3 = nn.MaxPool2d((2, 2), stride=2)

        self.linear_1 = nn.Linear(1024, 1024)
        self.dropout_1 = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(1024, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output)

        output = self.conv_layer_2(output)
        output = func.relu(output)
        output = self.maxpooling_2(output)

        output = self.conv_layer_3(output)
        output = func.relu(output)
        output = self.maxpooling_3(output)

        output = output.view((batch_size, -1))
        output = self.linear_1(output)
        output = self.dropout_1(output)
        output = self.linear_2(output)
        return output


class FusionDenseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(32, 256, (3, 3))

        self.conv_layer_2 = nn.Conv2d(256, 512, (3, 3))
        self.maxpooling_2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_layer_3 = nn.Conv2d(512, 1024, (3, 3))
        self.maxpooling_3 = nn.MaxPool2d((2, 2), stride=2)

        self.linear_1 = nn.Linear(1024, 1024)
        self.dropout_1 = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(1024, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output)

        output = self.conv_layer_2(output)
        output = func.relu(output)
        output = self.maxpooling_2(output)

        output = self.conv_layer_3(output)
        output = func.relu(output)
        output = self.maxpooling_3(output)

        output = output.view((batch_size, -1))
        output = self.linear_1(output)
        output = self.dropout_1(output)
        output = self.linear_2(output)
        return output


@Registers.model.register
class SpecificTrainModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor = ExtractionModel()
        self.dense = DenseModel()
        
        self.set_description("Specific 2D Train Model")

    def forward(self, input_tensor: torch.Tensor):
        output = self.extractor(input_tensor)
        output = self.dense(output)
        return output


@Registers.model.register
class SpecificTrainLongModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor = ExtractionModel()
        self.dense = DenseModel()
        self.pool = nn.AdaptiveAvgPool2d((13, 15))
        
        self.set_description("Specific 2D Train Model")

    def forward(self, input_tensor: torch.Tensor):
        output = self.extractor(input_tensor)
        output = self.pool(output)
        output = self.dense(output)
        return output


@Registers.model.register
class SpecificTrainLongLSTMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor = ExtractionModel()
        self.layer_dim = 2
        self.hidden_dim = 32
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.hidden_dim, num_layers=self.layer_dim)
        self.dense = DenseModel(self.hidden_dim * 21)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), stride=(1, 1)),
            nn.ReLU(),

            nn.AvgPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),

        )

        
        self.set_description("Specific 2D Train Model")

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.extractor(input_tensor)
        output = self.conv_layers(output)
        output = output.squeeze(2).permute([2, 0, 1]).contiguous()

        lstm_out, (hn, cn) = self.lstm(output)
        lstm_out = lstm_out.permute([1, 0, 2]).contiguous()
        lstm_out = lstm_out.view(batch_size, -1)
        lstm_out = self.dense(lstm_out)
        return lstm_out


@Registers.model.register
class SpecificTrainLongTransformerEncoderModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor = ExtractionModel()

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dense = DenseModel(21 * 128)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), stride=(1, 1)),
            nn.ReLU(),

            nn.AvgPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),

        )

        
        self.set_description("Specific 2D Train Model")

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.extractor(input_tensor)
        output = self.conv_layers(output)
        output = output.squeeze(2).permute([2, 0, 1]).contiguous()

        lstm_out = self.transformer_encoder(output)
        lstm_out = lstm_out.permute([1, 0, 2]).contiguous()
        lstm_out = lstm_out.view(batch_size, -1)
        lstm_out = self.dense(lstm_out)
        return lstm_out


@Registers.model.register
class MSMJointConcatFineTuneLongModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor_mfcc = ExtractionModel()
        self.extractor_spec = ExtractionModel()
        self.extractor_mel = ExtractionModel()
        self.pool1 = nn.AdaptiveAvgPool2d((13, 15))
        self.pool2 = nn.AdaptiveAvgPool2d((13, 15))
        self.pool3 = nn.AdaptiveAvgPool2d((13, 15))
        self.dense = ConcatModel()
        
        self.set_description("MFCC SPEC MELSPEC Joint 2D Fine-tune Model")

    def forward(self, input_mfcc: torch.Tensor, input_spec: torch.Tensor, input_mel: torch.Tensor):
        output_mfcc = self.extractor_mfcc(input_mfcc)
        output_spec = self.extractor_spec(input_spec)
        output_mel = self.extractor_mel(input_mel)
        output_mfcc = self.pool1(output_mfcc)
        output_spec = self.pool2(output_spec)
        output_mel = self.pool3(output_mel)
        concat_output = torch.cat([output_spec, output_mel, output_mfcc], dim=1)
        output = self.dense(concat_output)
        return output


@Registers.model.register
class MSMJointConcatFineTuneModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor_mfcc = ExtractionModel()
        self.extractor_spec = ExtractionModel()
        self.extractor_mel = ExtractionModel()
        self.dense = ConcatModel()
        
        self.set_description("MFCC SPEC MELSPEC Joint 2D Fine-tune Model")

    def forward(self, input_mfcc: torch.Tensor, input_spec: torch.Tensor, input_mel: torch.Tensor):
        output_mfcc = self.extractor_mfcc(input_mfcc)
        output_spec = self.extractor_spec(input_spec)
        output_mel = self.extractor_mel(input_mel)
        concat_output = torch.cat([output_spec, output_mel, output_mfcc], dim=1)
        output = self.dense(concat_output)
        return output


@Registers.model.register
class MSMJointFusionFineTuneModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor_mfcc = ExtractionModel()
        self.extractor_spec = ExtractionModel()
        self.extractor_mel = ExtractionModel()
        self.fusion_mfcc_spec = Registers.module["IAFFFusion"]((13, 15), 32, 4)
        self.fusion_mfcc_melspec = Registers.module["IAFFFusion"]((13, 15), 32, 4)
        self.fusion_specs = Registers.module["IAFFFusion"]((13, 15), 32, 4)
        self.dense = FusionDenseModel()
        
        self.set_description("MFCC SPEC MELSPEC Joint 2D Fusion Fine-tune Model")

    def forward(self, input_mfcc: torch.Tensor, input_spec: torch.Tensor, input_mel: torch.Tensor):
        output_mfcc = self.extractor_mfcc(input_mfcc)
        output_spec = self.extractor_spec(input_spec)
        output_mel = self.extractor_mel(input_mel)
        fusion_mfcc_spec = self.fusion_mfcc_spec(output_mfcc, output_spec)
        fusion_mfcc_mel = self.fusion_mfcc_spec(output_mfcc, output_mel)
        fusion_specs = self.fusion_mfcc_spec(fusion_mfcc_spec, fusion_mfcc_mel)
        output = self.dense(fusion_specs)
        return output


if __name__ == '__main__':
    import torchinfo

    model = SpecificTrainLongLSTMModel(input_shape=())
    torchinfo.summary(model.cuda(0), (4, 1, 128, 782))
