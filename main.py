import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings('ignore')

# настройки стиля графиков
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.style.use('dark_background')
sns.set_theme(
    style='darkgrid',
    rc={
        'axes.facecolor': '#191414',
        'figure.facecolor': '#191414',
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
    },
)


data = pd.read_csv('data/nba_rankings_2014-2020.csv')

print("\nЭТАП 1. Загрузка и проверка данных")
print(f"Размер датасета: {data.shape[0]} строк, {data.shape[1]} колонки")
print(f"Количество пропусков: {data.isna().sum().sum()}")
print(f"Количество дубликатов: {data.duplicated().sum()}")
print(f"Диапазон rankings: от {data['rankings'].min():.0f} до {data['rankings'].max():.0f}")


plt.figure(figsize=(8, 5))
sns.histplot(data['rankings'], bins=35, color='#1DB954')
plt.title('Распределение рейтингов')
plt.xlabel('Рейтинг')
plt.ylabel('Количество игроков')
plt.show()


# Тепловая карта средних статов по группам рейтинга
stats_for_heatmap = [
    'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK',
    'TOV', 'FG%', '3P%', 'FT%', 'DD2', 'TD3', '+/-'
]

rating_group = pd.cut(data['rankings'], bins=5)
heatmap_data = data.groupby(rating_group)[stats_for_heatmap].mean()

# Нормализуем колонки, чтобы признаки с разными масштабами нормально сравнивались цветом
heatmap_data_scaled = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()

plt.figure(figsize=(11, 6))
sns.heatmap(
    heatmap_data_scaled.T,
    cmap='viridis',
    center=0,
    linewidths=0.5,
    annot=True,
    fmt='.1f',
    cbar_kws={'label': 'Насколько стат выше/ниже среднего'},
)
plt.title('Тепловая карта статистики по группам рейтинга')
plt.xlabel('Статистика игрока')
plt.ylabel('Группа рейтинга')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.regplot(
    data=data,
    x='PTS',
    y='rankings',
    scatter_kws={'alpha': 0.45, 's': 24, 'color': '#1DB954'},
    line_kws={'color': '#FF4B4B', 'linewidth': 2},
)
plt.title('Связь очков за игру и рейтинга')
plt.xlabel('PTS: очки за игру')
plt.ylabel('Рейтинг')
plt.show()


plt.figure(figsize=(7, 4))
sns.boxplot(data=data, x='rankings', color='#1DB954')
plt.title('Boxplot рейтингов игроков')
plt.xlabel('Рейтинг')
plt.show()


y = data['rankings']

print("\nЭТАП 2. Очистка и подготовка признаков")
print("Удаляем Unnamed: 0, потому что это технический индекс из CSV.")
print("Удаляем PLAYER, потому что имя игрока является идентификатором.")
print("Отделяем rankings как target, чтобы не было утечки данных.")

# дропаем колонки которые не нужны
drop_cols = ['Unnamed: 0', 'PLAYER', 'rankings']
X = data.drop(columns=drop_cols)

print(f"Удаленные колонки: {drop_cols}")
print(f"Размер X после очистки: {X.shape[0]} строк, {X.shape[1]} колонок")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)


# категориальные (текстовые) признаки
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = ['TEAM', 'SEASON']

print("Числовые признаки:", numeric_features)


X_train_num = X_train[numeric_features]
X_test_num = X_test[numeric_features]


model = Ridge()
model.fit(X_train_num, y_train)

y_pred = model.predict(X_test_num)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


y_pred_baseline = np.full_like(y_test, y_train.mean())
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)
X_test_scaled = scaler.transform(X_test_num)

model_scaled = Ridge()
model_scaled.fit(X_train_scaled, y_train)

weights_scaled = pd.Series(model_scaled.coef_, index=numeric_features)

# топ 10 положительных и топ 10 отрицательных признаков 
top_positive = weights_scaled.sort_values(ascending=False).head(10)
top_negative = weights_scaled.sort_values(ascending=True).head(10)
selected_weights = pd.concat([top_positive, top_negative]).sort_values()
colors = np.where(selected_weights >= 0, '#1DB954', '#FF4B4B')

plt.figure(figsize=(10, 7))
selected_weights.plot(kind='barh', color=colors)
plt.axvline(0, color='white', linewidth=1)
plt.title('Топ положительных и отрицательных признаков')
plt.xlabel('Коэффициент после')
plt.show()


print("\nИНТЕРПРЕТАЦИЯ ПРИЗНАКОВ")
print("\nТоп-10 положительных признаков:")
print(top_positive.to_string())
print("\nТоп-10 отрицательных признаков:")
print(top_negative.to_string())


param_grid = {'alpha': np.logspace(-3, 3, 20)}

# cv=5 означает 5-фолдовую кросс-валидацию
grid_search = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
)
grid_search.fit(X_train_scaled, y_train)

print(f"\nЛучшее значение alpha: {grid_search.best_params_['alpha']}")
print(f"Лучший RMSE на кросс-валидации: {-grid_search.best_score_:.2f}")


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=grid_search.best_params_['alpha'])),
])

pipeline.fit(X_train, y_train)

y_pred_pipe = pipeline.predict(X_test)
rmse_pipe = np.sqrt(mean_squared_error(y_test, y_pred_pipe))


pipeline_lasso = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Lasso(alpha=0.1)),
])

pipeline_lasso.fit(X_train, y_train)

zero_weights = np.sum(pipeline_lasso.named_steps['model'].coef_ == 0)
total_weights = len(pipeline_lasso.named_steps['model'].coef_)
print(f"Lasso занулил {zero_weights} признаков из {total_weights}!")

rmse_lasso = np.sqrt(mean_squared_error(y_test, pipeline_lasso.predict(X_test)))


y_train_pred = pipeline.predict(X_train)
errors = (y_train - y_train_pred) ** 2

plt.figure(figsize=(8, 5))
sns.histplot(np.sqrt(errors), bins=25, kde=True, color='#1DB954')
plt.axvline(np.sqrt(errors).mean(), color='white', linestyle='--', label='Средняя ошибка')
plt.title('Насколько модель ошибается в рейтинге игроков')
plt.xlabel('Ошибка в пунктах рейтинга')
plt.ylabel('Количество игроков')
plt.legend()
plt.show()


# удаляем выбросы
threshold = np.quantile(errors, 0.95)
mask = errors < threshold

X_train_clean = X_train[mask]
y_train_clean = y_train[mask]

pipeline.fit(X_train_clean, y_train_clean)
y_pred_clean = pipeline.predict(X_test)
rmse_clean = np.sqrt(mean_squared_error(y_test, y_pred_clean))


results = pd.DataFrame({
    'Модель': [
        'Baseline (среднее)',
        'Ridge',
        'Ridge + TEAM/SEASON',
        'Lasso + TEAM/SEASON',
        'Ridge после удаления выбросов',
    ],
    'RMSE': [
        rmse_baseline,
        rmse,
        rmse_pipe,
        rmse_lasso,
        rmse_clean,
    ],
}).sort_values('RMSE')

best_model = results.iloc[0]

print("\nИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
print(results.to_string(index=False, formatters={'RMSE': '{:.2f}'.format}))
print(
    f"\nЛучшая модель: {best_model['Модель']} "
    f"с RMSE = {best_model['RMSE']:.2f}"
)
print(
    f"Модель ошибается примерно на "
    f"{best_model['RMSE'] / rmse_baseline * 100:.2f}% "
    "от ошибки бейзлайна."
)
