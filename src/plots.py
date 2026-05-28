import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import target


def plot_target_distribution(data):
    plt.figure(figsize=(8, 5))
    sns.histplot(data[target], bins=35)
    plt.title('Распределение рейтингов')
    plt.show()


def plot_correlation_matrix(data):
    numeric_data = data.select_dtypes(include='number')
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        cmap='viridis',
        center=0,
        linewidths=0.2,
        cbar_kws={'label': 'Корреляция'},
    )
    plt.title('Корреляционная матрица числовых признаков')
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.show()


def plot_feature_vs_target(data, feature):
    if feature not in data.columns:
        print(f"Колонка {feature} не найдена, график не построен.")
        return

    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=data,
        x=feature,
        y=target,
        scatter_kws={'alpha': 0.45, 's': 24, 'color': '#1DB954'},
        line_kws={'color': '#FF4B4B', 'linewidth': 2},
    )
    plt.title(f'{feature} vs {target}')
    plt.xlabel(feature)
    plt.ylabel('Рейтинг NBA 2K')
    plt.show()


def plot_rating_boxplot(data):
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=data, x=target, color='#1DB954')
    plt.title('Boxplot рейтингов игроков')
    plt.xlabel('Рейтинг NBA 2K')
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
