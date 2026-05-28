import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import TARGET_COLUMN


def plot_target_distribution(data):
    plt.figure(figsize=(8, 5))
    sns.histplot(data[TARGET_COLUMN], bins=35)
    plt.title('Распределение рейтингов')
    plt.show()


def plot_feature_importance(weights):
    top_positive = weights.sort_values(ascending=False).head(10)
    top_negative = weights.sort_values(ascending=True).head(10)
    selected_weights = pd.concat([top_positive, top_negative]).sort_values()
    colors = np.where(selected_weights >= 0, '#1DB954', '#FF4B4B')

    selected_weights.plot(kind='barh', figsize=(10, 7), color=colors)
    plt.axvline(0, color='white', linewidth=1)
    plt.title("Топ положительных и отрицательных признаков Ridge")
    plt.xlabel('Коэффициент после StandardScaler')
    plt.show()


def plot_train_errors(errors):
    plt.figure(figsize=(8, 5))
    sns.histplot(np.sqrt(errors), bins=25, kde=True, color='#1DB954')
    plt.axvline(np.sqrt(errors).mean(), color='white', linestyle='--', label='Средняя ошибка')
    plt.title('Насколько модель ошибается в рейтинге игроков')
    plt.xlabel('Ошибка в пунктах рейтинга')
    plt.ylabel('Количество игроков')
    plt.legend()
    plt.show()


def plot_model_comparison(results):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=results, x='RMSE', y='Модель', color='#1DB954')
    plt.title('Сравнение моделей по RMSE')
    plt.xlabel('RMSE: меньше — лучше')
    plt.ylabel('')
    plt.xlim(0, results['RMSE'].max() + 0.5)
    plt.show()
