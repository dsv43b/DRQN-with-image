import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Tuple, List


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoisyLinear(nn.Linear):
    """
    Зашумленные слои для нейросети
    """
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True, train=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.train = train
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        if self.train:
            return F.linear(input, self.weight + self.sigma_weight *\
                            self.epsilon_weight.data, bias)
        else:
            return F.linear(input, self.sigma_weight, self.sigma_bias)


class DQN(nn.Module):
    def __init__(self, inp_shape, out_shape, train=True):
        super(DQN, self).__init__()

        self.hidden_layer = 256
        self.num_layers_lstm = 2
        self.lstm_hidden_dig = 256
        self.lstm_hidden_img = 128

        # 1-я часть с цифровыми данными
        self.linear_layer_1 = nn.Sequential(
            nn.Linear(inp_shape, self.hidden_layer),
            nn.ELU(),
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.ELU()
        )
        self.lstm = nn.LSTM(self.hidden_layer, self.lstm_hidden_dig,
                            self.num_layers_lstm, batch_first=True)
        self.linear_dg = nn.Linear(96 * self.lstm_hidden_dig, 1024)

        # 2-я часть с изображением
        self.conv_layer = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bilstm = nn.LSTM(34 * 47, self.lstm_hidden_img, 1,
                              batch_first=True, bidirectional=True)
        self.linear_im = nn.Linear(128 * 128 * 2, 1024)

        # Создание зашумленных слоев
        self.noisy_layers = [
            NoisyLinear(1536, self.hidden_layer, train=train),
            NoisyLinear(1536, self.hidden_layer, train=train),
            NoisyLinear(self.hidden_layer, out_shape, train=train),
            NoisyLinear(self.hidden_layer, 1, train=train)
        ]

        # Объединение двух сетей и переход в DDQN
        self.fc_adv = nn.Sequential(
            self.noisy_layers[0],
            nn.ELU(),
            self.noisy_layers[2]
        )

        self.fc_val = nn.Sequential(
            self.noisy_layers[1],
            nn.ELU(),
            self.noisy_layers[3]
        )

    def forward(self, digital_data, img, h_c):
        batch_size = digital_data.size(0)
        # Присваивание начальго состояния
        h0_d, c0_d, h0_i, c0_i = h_c

        # Цифровые данные
        digital_data = self.linear_layer_1(digital_data)
        _, (hn_d, cn_d) = self.lstm(digital_data, (h0_d, c0_d))
        out_1 = hn_d.reshape(batch_size, -1)

        # Изображение
        img = self.conv_layer(img)
        img = img.reshape(batch_size, 128, -1)
        out_2, (hn_i, cn_i) = self.bilstm(img, (h0_i, c0_i))
        out_2 = out_2.reshape(batch_size, -1)
        out_2 = F.leaky_relu(self.linear_im(out_2))

        # Объединение выходов
        common_out = torch.hstack((out_1, out_2))

        val = self.fc_val(common_out)
        adv = self.fc_adv(common_out)
        return val + (adv - adv.mean(dim=1, keepdims=True)), \
               (hn_d, cn_d, hn_i, cn_i)

    def update_hidden_state(self, batch_size: int) -> Tuple[torch.tensor]:
        return (
            torch.zeros([self.num_layers_lstm, batch_size, self.lstm_hidden_dig], device=DEVICE),
            torch.zeros([self.num_layers_lstm, batch_size, self.lstm_hidden_dig], device=DEVICE),
            torch.zeros([2 * 1, batch_size, self.lstm_hidden_img], device=DEVICE),
            torch.zeros([2 * 1, batch_size, self.lstm_hidden_img], device=DEVICE)
        )

    def noisy_layers_sigma_snr(self) -> List[torch.tensor]:
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]