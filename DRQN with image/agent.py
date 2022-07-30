import pandas as pd
import torch
import torch.nn as nn
from collections import deque
from datetime import datetime, timedelta, time
import datetime
from colorama import Fore, Back, Style

from environment import TradingEnv
from neural_network import DQN
from buffer import ReplayMemory
from typing import Union, Tuple, List



GAMMA = 0.99
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SYNCHRONZ_MODEL = 1000
SEED = 10
LEARNING_STEP = 2
MIN_BUFFER_SIZE = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Лучшие веса для тестовой модели
PATH_SAVE_MODEL = 'model_weight.pth'
# Лучшие веса для тренировочной модели
PATH_SAVE_MODEL_bs = 'model_weight_bs.pth'


class Agent:
    '''
    Торговый агент

    data: подготовленные данные для обучения
    bufer_size: резмер буфера
    image_archive: преобразованные изображения в тензор pytorch
    test: режим для проверки на тестовых данных
    load_nn_weights: загрузка весов модели из файла
    '''

    def __init__(self, data: pd.DataFrame, bufer_size: int,
                 image_archive: torch.tensor, test: bool = False,
                 load_nn_weights: bool = False):

        self.device = DEVICE
        self.image_archive = image_archive
        self.test = test
        self.best_award = -1e10
        self.load_nn_weights = load_nn_weights

        # Создание торговой среды
        self.env = TradingEnv(data, lot_size=80)
        self.action_space = 3
        self.input_data = data.shape[1] - 1

        # Создание буфера
        self.buffer = ReplayMemory(bufer_size, image_archive)

        # Создание онлайн и таргет нейросети
        self.net = DQN(self.input_data, self.action_space, train=not (test)).to(DEVICE)

        if self.load_nn_weights:
            self.net.load_state_dict(torch.load(PATH_SAVE_MODEL_bs))

        self.target_net = DQN(self.input_data, self.action_space, train=not (test)).to(DEVICE)
        self.target_net.load_state_dict(self.net.state_dict())

        # Функция потерь и оптимизатор
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

        self.mean_reward = deque(maxlen=50)
        self.count_step = 0

    def set_weight(self, W: dict) -> None:
        self.target_net.load_state_dict(W)
        self.net.load_state_dict(W)

    def get_weight(self) -> dict:
        return self.target_net.state_dict()

    def choosing_action(self, state: torch.tensor,
                        image_number: int,
                        hc_init: Tuple[torch.tensor]) -> Tuple[int, torch.tensor]:
        image = self.image_archive[image_number].unsqueeze(0).to(DEVICE)
        state = state.to(DEVICE)
        with torch.no_grad():
            Q_val, hc_init = self.net(state, image, hc_init)
        return torch.argmax(Q_val).item(), hc_init

    @torch.no_grad()
    def calculation_target(self, next_state: torch.tensor,
                           image: torch.tensor, reward: float, done: bool) -> float:
        hc_zero = self.target_net.update_hidden_state(BATCH_SIZE)
        next_step_net_Q = self.net(next_state, image, hc_zero)[0].argmax(1).unsqueeze(-1)
        next_step_target_net_Q = self.target_net(next_state, image, hc_zero)[0].gather(1, next_step_net_Q)
        Q_val = reward + GAMMA * next_step_target_net_Q * (1 - done)
        return Q_val

    def run(self, episode) -> None:

        state, num_img_state = self.env.reset()
        state.unsqueeze_(0)
        episode_reward = 0  # Награда шагов за эпизод
        local_step = 0  # Кол-во шагов за эпизод
        loss_episode = []  # Потери за эпизод
        hc_initial = self.net.update_hidden_state(1)

        while True:
            # Выбор действия
            action, hc_new = self.choosing_action(state, num_img_state, hc_initial)

            # Присваиваю базовым весам новые значения полученные от модели
            hc_initial = hc_new

            (next_state, num_img_nstate), reward, done, _ = self.env.step(action)
            next_state.unsqueeze_(0)
            self.count_step += 1
            local_step += 1
            episode_reward += reward

            # Добавление новых данных в буфер памяти
            self.buffer.add((state, num_img_state), action, reward,
                            (next_state, num_img_nstate), done)

            state = next_state
            num_img_state = num_img_nstate

            if done:
                # Для вычисления среденей награды
                self.mean_reward.append(episode_reward)

                # Получение московского времени
                tz = datetime.timezone(datetime.timedelta(hours=3))
                now = datetime.datetime.now(tz=tz)
                current_time = now.strftime("%H:%M:%S")

                if not self.test:
                    torch.save(self.get_weight(), PATH_SAVE_MODEL_bs)

                # Цвет текста в зависимости обрабатываемых данных
                trade_mode = 'test ' if self.test else 'train'
                color = Fore.GREEN if self.test else Fore.YELLOW

                print(f'{color}{episode + 1:03d}_{trade_mode}{Style.RESET_ALL}| Время: {current_time}| '
                      f'Награда: {episode_reward:6.1f}| Шагов: {self.count_step:06d}| Шагов эпз: {local_step:06d}|'
                      f'Баланс: {float(self.env.balance_history[-1]):.1f}| '
                      f'Кол-во сделок: {self.env.number_open_transactions:4d}')

                # Вывод соотношения сигнал/шум
                if not self.test:
                    for layer_idx, sigma_l2 in enumerate(self.net.noisy_layers_sigma_snr()):
                        print("sigma_snr_layer_%d" % (layer_idx + 1), sigma_l2)
                break

            if len(self.buffer) > MIN_BUFFER_SIZE and \
                    self.count_step % LEARNING_STEP == 0:
                self.optimizer.zero_grad()

                # Получение батча примеров из буфера
                state_t, action_t, reward_t, next_state_t, done_t, \
                image_state, image_nstate = self.buffer.sample(BATCH_SIZE)

                # Вычесление ошибки Q значения
                target_Q = self.calculation_target(next_state_t, image_nstate, reward_t, done_t)
                hc_zero_batch = self.net.update_hidden_state(BATCH_SIZE)
                predict_Q = self.net(state_t, image_state, hc_zero_batch)[0].gather(1, action_t)

                loss = self.loss_fn(predict_Q, target_Q)
                loss_episode.append(loss.item())
                loss.backward()
                self.optimizer.step()

            # Обновление целевой модели
            if self.count_step % SYNCHRONZ_MODEL == 0:
                self.target_net.load_state_dict(self.net.state_dict())