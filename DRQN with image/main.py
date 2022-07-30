from agent import Agent, PATH_SAVE_MODEL
import torch
import numpy as np
import random
import common

EPISODES = 200

def main(df_train, df_test):

    agent_train = Agent(df_train, 3000, processed_images_train, load_nn_weights=False)
    agent_test = Agent(df_test, 400, processed_images_test, test=True)
    start_balanse = 100000

    for episode in range(EPISODES):
        agent_train.run(episode)
        weights_train = agent_train.get_weight()
        agent_test.set_weight(weights_train)
        agent_test.run(episode)
        print()

        if agent_test.env.balance_history[-1] > start_balanse:
            start_balanse = agent_test.env.balance_history[-1]
            torch.save(weights_train, PATH_SAVE_MODEL)


if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)

    print('Загрузка данных')
    df_SBER = common.loading_data()
    df_SBER = common.feature_engineering(df_SBER)

    # Удаляю признаки у которых очень высокая корреляция с ценой закрытия Сбера
    df_SBER.drop(columns=['High', 'Low', 'Open', 'EMA', 'Close_MOEX'], inplace=True)
    df_SBER.dropna(inplace=True)
    df_SBER.reset_index(drop=True, inplace=True)
    # Направление открытия сделки
    df_SBER['Actions'] = 0
    # Изменение баланса
    df_SBER['Changing balance'] = 0

    print('Разбиение данных на тестовые и тренировочные')
    # Разбиение данных на тестовые и тренировочные
    df_SBER_test = df_SBER[(df_SBER['Full_time'] > '2021-10-03')] \
        .reset_index(drop=True)

    df_SBER_train = df_SBER[(df_SBER['Full_time'] > '2020-02-06') &
                            (df_SBER['Full_time'] < '2021-08-02')] \
        .reset_index(drop=True)

    common.unarchive()
    processed_images_test = common.image_conversion(df_SBER_test)
    processed_images_train = common.image_conversion(df_SBER_train)

    print('Запуск агента')
    main(df_SBER_train, df_SBER_test)