import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DROP_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_data(path):
    data = pd.read_csv(path)

    print("\nЭТАП 1. Загрузка и первичная проверка данных")
    print(f"Размер датасета: {data.shape[0]} строк, {data.shape[1]} колонки")
    print(f"Количество пропущенных значений: {data.isna().sum().sum()}")
    print(f"Количество полных дубликатов строк: {data.duplicated().sum()}")
    print(
        "Диапазон target rankings: "
        f"от {data[TARGET_COLUMN].min():.0f} до {data[TARGET_COLUMN].max():.0f}"
    )

    return data


def prepare_features(data):
    y = data[TARGET_COLUMN]

    print("\nЭТАП 2. Очистка и подготовка признаков")
    print("Удаляем Unnamed: 0, потому что это технический индекс из CSV.")
    print("Удаляем PLAYER, потому что имя игрока является идентификатором, а не статистикой.")
    print("Отделяем rankings как target, чтобы не было утечки данных.")

    x = data.drop(columns=DROP_COLUMNS)
    print(f"Удаленные колонки: {DROP_COLUMNS}")
    print(f"Размер матрицы признаков после очистки: {x.shape[0]} строк, {x.shape[1]} колонок")

    return x, y


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Размер обучающей выборки:", x_train.shape)
    print("Размер тестовой выборки:", x_test.shape)

    return x_train, x_test, y_train, y_test


def get_numeric_features(x_train):
    numeric_features = x_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print("Числовые признаки:", numeric_features)
    return numeric_features
