import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

#!target = 'rankings'

# %config InlineBackend.figure_format = 'retina' #! спросить почему не работает
plt.style.use('dark_background')
sns.set_theme(style='darkgrid', rc={'axes.facecolor': '#191414', 'figure.facecolor': '#191414', 'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data/nba_rankings_2014-2020.csv')


plt.figure(figsize=(8, 5))
sns.histplot(data['rankings'], bins=35)
plt.title('Распределение рейтингов')
plt.show()


y = data['rankings']

# Удаляем таргет и текстовые идентификаторы
drop_cols = ["Unnamed: 0", "PLAYER", "rankings"]
X = data.drop(columns=drop_cols)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)


numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Числовые признаки:", numeric_features)

X_train_num = X_train[numeric_features]
X_test_num = X_test[numeric_features]


# Создаем объект модели
model = Ridge()

# Обучаем модель (подбираем веса w и w0)
model.fit(X_train_num, y_train)


y_pred = model.predict(X_test_num)
y_pred[:5]


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE на тесте: {rmse:.2f}")


y_pred_baseline = np.full_like(y_test, y_train.mean())
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
print(f"RMSE бейзлайна (просто среднее): {rmse_baseline:.2f}") 


weights = pd.Series(model.coef_, index=numeric_features)
weights.sort_values(ascending=False)


scaler = StandardScaler()
# Находим средние и дисперсии на train и сразу применяем к train
X_train_scaled = scaler.fit_transform(X_train_num)
# На test мы ТОЛЬКО применяем (не пересчитываем средние!)
X_test_scaled = scaler.transform(X_test_num)

# Обучаем новую модель на отмасштабированных данных
model_scaled = Ridge()
model_scaled.fit(X_train_scaled, y_train)

weights_scaled = pd.Series(model_scaled.coef_, index=numeric_features)
weights_scaled.sort_values(ascending=False).plot(kind='bar', figsize=(10, 5), color='#1DB954')
plt.title("Истинная важность признаков (после StandardScaler)")
plt.show()


# Задаем сетку гиперпараметров
param_grid = {'alpha': np.logspace(-3, 3, 20)}

# cv=5 означает 5-фолдовую кросс-валидацию
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

print(f"Лучшее значение alpha: {grid_search.best_params_['alpha']}")
print(f"Лучший RMSE на кросс-валидации: {-grid_search.best_score_:.2f}")


categorical_features = ["TEAM", "SEASON"]

# Создаем трансформер: к числам применяем StandardScaler, к категориям — OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Собираем всё в Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=grid_search.best_params_['alpha']))
])


# Обучаем ВЕСЬ пайплайн одной строчкой!
pipeline.fit(X_train, y_train)

# Предсказываем
y_pred_pipe = pipeline.predict(X_test)
rmse_pipe = np.sqrt(mean_squared_error(y_test, y_pred_pipe))
print(f"RMSE с категориальными признаками: {rmse_pipe:.2f}")


pipeline_lasso = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Lasso(alpha=0.1)) # alpha=0.1 для Лассо обычно достаточно сильно режет
])

pipeline_lasso.fit(X_train, y_train)

# Считаем количество нулевых весов
zero_weights = np.sum(pipeline_lasso.named_steps['model'].coef_ == 0)
total_weights = len(pipeline_lasso.named_steps['model'].coef_)

print(f"Lasso занулил {zero_weights} признаков из {total_weights}!")
rmse_lasso = np.sqrt(mean_squared_error(y_test, pipeline_lasso.predict(X_test)))
print(f"RMSE Lasso: {rmse_lasso:.2f}")


# Получаем предсказания на обучающей выборке
y_train_pred = pipeline.predict(X_train)
errors = (y_train - y_train_pred) ** 2

plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=20)
plt.title('Распределение квадратов ошибок на обучающей выборке')
plt.show()


# Берем 95-й квантиль ошибки
threshold = np.quantile(errors, 0.95)
mask = errors < threshold # Маска "хороших" треков

# Оставляем только те треки, где ошибка была не слишком большой
X_train_clean = X_train[mask]
y_train_clean = y_train[mask]

# Переобучаем
pipeline.fit(X_train_clean, y_train_clean)
y_pred_clean = pipeline.predict(X_test)
rmse_clean = np.sqrt(mean_squared_error(y_test, y_pred_clean))
print(f"RMSE после удаления выбросов: {rmse_clean:.2f}")