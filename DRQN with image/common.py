import pandas as pd
import talib
import numpy as np
import zipfile
import torch
import torchvision.transforms as T
from torchvision.io import read_image


def date_preprocessing(df: pd.DataFrame, secondary_columns: bool = True,
                       labl: str = None) -> pd.DataFrame:
    """
    Начальная предобработка данных, приведение времени к формату datetime
    """
    df_c = df.copy()
    df_c.replace({'<TIME>': {0: '000000'}}, inplace=True)
    df_c['Full_time'] = df_c['<DATE>'].astype('str') + df_c['<TIME>'].astype('str')
    df_c['Full_time'] = pd.to_datetime(df_c['Full_time'], format="%Y%m%d%H%M%S")
    df_c.drop(columns=['<DATE>', '<TIME>'], inplace=True)
    df_c.rename(columns={"<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low",
                    "<CLOSE>": "Close", "<VOL>": "Volume"}, inplace=True)

    if secondary_columns:
        return df_c
    else:
        df_c.rename(columns={"Close": f"Close_{labl}"}, inplace=True)
        return df_c[['Full_time', f"Close_{labl}"]]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление новых признаков
    """
    df_c = df.copy()

    close = df_c['Close']
    high = df_c['High']
    low = df_c['Low']

    macd, _, macdhist = talib.MACD(close)
    df_c['MACD'] = macd
    df_c['MACDh'] = macdhist
    df_c['EMA'] = talib.EMA(close, timeperiod=40)
    slowk, slowd = talib.STOCH(high, low, close)
    df_c['STOCHk'] = slowk
    df_c['STOCHd'] = slowd

    df_c['RSI'] = talib.RSI(close)
    df_c['WILLR'] = talib.WILLR(high, low, close)
    df_c['DIFF'] = close.diff()
    df_c['DEA'] = df_c['EMA'].diff()
    df_c['BOP'] = (df_c['Open'] - df_c['Close']) / (df_c['High'] - df_c['Low'])

    df_c['sin_weekday'] = np.sin(2 * np.pi * df_c['Full_time'].dt.weekday / 6)
    df_c['sin_hour'] = np.sin(2 * np.pi * df_c['Full_time'].dt.hour / 23)
    df_c['sin_minute'] = np.sin(2 * np.pi * df_c['Full_time'].dt.minute / 45)
    df_c['sin_days'] = np.sin(2 * np.pi * df_c['Full_time'].dt.day /
                                df_c['Full_time'].dt.days_in_month)
    return df_c


def loading_data():
    # Загрузка данных стоимости акций Сбербанка
    df_SBER = pd.read_csv('SBER_200201_220201.csv', sep=';')

    # Загрузка данных индекса Московской биржи
    df_MOEX = pd.read_csv('IMOEX.csv')

    # Загрузка данных индекса D&J
    df_DJ = pd.read_csv('D&J-IND.csv')

    # Загрузка данных Финансового индекса
    df_MOEXFN = pd.read_csv('MOEXFN.csv')

    # Загрузка данных курса валюты USD-RUB
    df_USD_RUB = pd.read_csv('USD-RUB.csv')

    df_SBER = date_preprocessing(df_SBER)
    df_MOEX = date_preprocessing(df_MOEX, secondary_columns=False, labl='MOEX')
    df_DJ = date_preprocessing(df_DJ, secondary_columns=False, labl='D&J')
    df_MOEXFN = date_preprocessing(df_MOEXFN, secondary_columns=False, labl='MOEXFN')
    df_USD_RUB = date_preprocessing(df_USD_RUB, secondary_columns=False, labl='USD_RUB')

    # Объединяю в единый датафрейм
    for ind in [df_MOEX, df_DJ, df_MOEXFN, df_USD_RUB]:
        df_SBER = df_SBER.merge(ind, how='left', on='Full_time')

    # Заполняю пропуски предыдущими значениями, так как рынки работают в разное время
    df_SBER.fillna(method="bfill", inplace=True)
    return df_SBER

def unarchive(link='C:\\Users\\User\\PycharmProjects\\untitled\\DRQN with image\\SBER_image-20220604T111426Z-001.zip'):
    print('Начата распаковка файла....')
    fantasy_zip = zipfile.ZipFile(link)
    fantasy_zip.extractall('C:\\Users\\User\\PycharmProjects\\untitled\\DRQN with image')
    fantasy_zip.close()
    print('Распаковка файла завершена.')


def custom_crop_volume(image):
    # Для объема
    return T.functional.crop(image, 357, 140, 120, 580)

def custom_crop_candle(image):
    # Для свечей
    return T.functional.crop(image, 70, 140, 290, 580)


preprocess_candle = T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Lambda(custom_crop_candle),
    T.Resize((290, 390)),
    T.Normalize(
        mean=0.9565,
        std=0.1731
    )
])

preprocess_volume = T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Lambda(custom_crop_volume),
    T.Resize((290, 390)),
    T.Normalize(
        mean=0.8322,
        std=0.3393
    )
])


def image_conversion(df: pd.DataFrame, path_to_images: str='SBER_image') -> torch.Tensor:
    date_list = df['Full_time'].astype('str').to_list()
    date_list = sorted(list(map(lambda x: x.replace(':', '_') + '.png', date_list)))
    processed_images = []
    print('Преобразование изображения в тензор начато....')

    for i_name in date_list:
        img = read_image(f'{path_to_images}/{i_name}')
        processed_images.append(
            torch.FloatTensor(
                torch.vstack(
                    (preprocess_volume(img), preprocess_candle(img))
                    )
            )
        )
    print('Преобразование изображений завершено.')
    return processed_images

