import pandas as pd
import numpy as np
import torch
from typing import Union, Tuple, List


class TradingEnv:
    """
   Эмулятор биржи. При каждом шаге выдает новые данные о стоимости и новое изображение.

   Атрибуты:
       data: подготовленные данные со стоимостью акций.
       start_balance: стартовый баланс.
       time_step: размер временного интервала.
       exchange_commission: комиссия биржи, в процентах.
       lot_size: размер лота.
   """

    def __init__(self, data: pd.DataFrame, start_balance: int = 100_000,
                 time_step: int = 96, exchange_commission: float = 0.1,
                 lot_size: int = 10) -> None:
        self.start_balance = start_balance
        self.time_step = time_step
        self.exchange_commission = exchange_commission
        self.lot_size = lot_size
        self.close_price = data['Close'].to_numpy()
        self.data = data.drop(columns='Full_time').to_numpy()
        self.data_shape = data.shape[0]

    def reset(self) -> Tuple[torch.tensor, int]:
        '''
        Сброс начальных параметров
        '''

        self.number_open_transactions = 0  # Количество открытых сделок
        self.transaction_opening_price = 0  # Цена открытия сделки
        self.action_opening = 0  # Действие на октрытие позиции
        self.balance_history = [self.start_balance]  # История изменения баланса
        self.global_step = 0  # Вычисление номера шага, для вывода следующих данных
        self.balance_at_opening_deal = self.start_balance  # Баланс при открытии сделки
        self.data[:, -2:] = 0  # Обнуление данных о прибыли и открытых сделках
        self.action_history = []  # Список с принятыми действиями
        self.SH_history = []  # Список с изменениями к-нта Шарпа
        self.PR_history = []  # Список с изменениями вознаграждения

        return torch.FloatTensor(
            self.__normalization(self.data[:self.time_step])
        ), self.time_step - 1

    def __profit_calculation(self, price: float) -> float:
        '''
        Расчет прибыли на каждом шаге
        '''

        commission = self.lot_size * self.exchange_commission * \
                     (price + self.transaction_opening_price) / 100
        profit = self.lot_size * self.action_opening * \
                 (price - self.transaction_opening_price) - commission
        return profit

    def __normalization(self, data: np.ndarray) -> np.ndarray:
        '''
        Нормализация данных перед подачей их в модель
        '''

        norm_data = data[:, :-6].copy()
        not_norm_data = data[:, -6:].copy()
        min_n = norm_data.min(axis=0)
        max_n = norm_data.max(axis=0)
        np.seterr(invalid='ignore')
        norm_data = (norm_data - min_n) / (max_n - min_n)
        norm_data = np.nan_to_num(norm_data, nan=1)
        return np.hstack([norm_data, not_norm_data])

    def __Sharp_ratio(self, risk_free: float = 0.0) -> Tuple[np.ndarray, bool]:
        '''
        Расчет коэффициента Шарпа и значение бездействия за последнии 10 шагов
        '''

        if len(self.balance_history) < self.time_step + 1:
            local_start_bln = np.array(self.balance_history[:-1])
            local_bln = np.array(self.balance_history[1:])
        else:
            local_start_bln = np.array(self.balance_history[-self.time_step - 1: -1])
            local_bln = np.array(self.balance_history[-self.time_step:])

        delta_bln = (local_bln - local_start_bln) / local_start_bln

        # Коэффициент Шарпа
        SR = (np.mean(delta_bln) - risk_free) / (np.std(delta_bln) + 1e-6)
        # Бездействие за 10 шагов
        act_10 = np.array(self.action_history[-10:])
        inaction = np.all(act_10 == act_10[-1])

        if np.isinf(SR) or np.isnan(SR):
            SR = 0.5
        return np.clip(SR, -1, 1), inaction

    def step(self, action: int) -> Tuple[Tuple[torch.tensor, int], float, bool, None]:
        '''
        Действия:
        0 - ничего не далать,
        1 - покупка,
        -1 - продажа
        '''

        basic_actions = [-1, 0, 1]
        loc_act = basic_actions[action]
        self.action_history.append(loc_act)

        # Текущая цена закрытия
        current_price = self.close_price[self.global_step + self.time_step - 1]

        # Следующая цена
        next_price = self.close_price[self.global_step + self.time_step]

        # Проверка условия для открытия новой сделки, сделка открывается по
        # current_price цене
        if self.transaction_opening_price == 0 and loc_act != 0:
            self.action_opening = loc_act
            self.transaction_opening_price = current_price
            self.number_open_transactions += 1

        # Проверка условия для закрытия сделки, закрывается по current_price цене
        elif (self.action_opening == 1 and loc_act == -1) or \
                (self.action_opening == -1 and loc_act == 1):
            self.balance_at_opening_deal += self.__profit_calculation(current_price)
            self.action_opening = 0
            self.transaction_opening_price = 0

        # Расчет комиссии и прибыли открытой сделки. Расчет происходит по next_price
        transaction_profit = self.__profit_calculation(next_price)
        self.balance_history.append(
            self.balance_at_opening_deal if self.transaction_opening_price == 0
            else self.balance_at_opening_deal + transaction_profit
        )

        # Вознаграждение(изменение баланса)
        PR = (self.balance_history[-1] - self.balance_history[-2]) / \
             self.balance_history[-2]

        # Вознаграждение(коэффициент Шарпа)
        SR, inaction = self.__Sharp_ratio()

        # Наказание за бездействие
        punishment_inaction = -0.1 if inaction else 0

        self.SH_history.append(SR)
        self.PR_history.append(PR)
        reward = PR * 100 + SR + punishment_inaction

        # В крайнюю колонку добавляя информацию о изменении баланса
        self.data[self.global_step + self.time_step, -1] = \
            (self.balance_history[-1] - self.start_balance) / \
            self.start_balance

        # В предпоседнюю колонку добавляю наличие открытой сделки
        self.data[self.global_step + self.time_step, -2] = self.action_opening

        # Флаг завершения эпизода
        done = True if (self.global_step == self.data.shape[0] - self.time_step - 1
                        or self.balance_history[-1] <= 0) else False

        self.global_step += 1
        next_state = self.__normalization(
            self.data[self.global_step:self.global_step + self.time_step]
        )

        return (torch.FloatTensor(next_state),
                self.global_step + self.time_step - 1), reward, done, None

