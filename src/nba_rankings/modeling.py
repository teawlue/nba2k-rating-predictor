import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_baseline(y_train, y_test):
    y_pred_baseline = np.full_like(y_test, y_train.mean())
    return calculate_rmse(y_test, y_pred_baseline)


def train_ridge_numeric(x_train, x_test, y_train, y_test, numeric_features):
    x_train_num = x_train[numeric_features]
    x_test_num = x_test[numeric_features]

    model = Ridge()
    model.fit(x_train_num, y_train)

    y_pred = model.predict(x_test_num)
    rmse = calculate_rmse(y_test, y_pred)

    return model, rmse, x_train_num, x_test_num


def train_scaled_ridge(x_train_num, x_test_num, y_train, numeric_features):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_num)
    x_test_scaled = scaler.transform(x_test_num)

    model_scaled = Ridge()
    model_scaled.fit(x_train_scaled, y_train)

    weights_scaled = pd.Series(model_scaled.coef_, index=numeric_features)
    return model_scaled, weights_scaled, x_train_scaled, x_test_scaled


def tune_ridge_alpha(x_train_scaled, y_train):
    param_grid = {'alpha': np.logspace(-3, 3, 20)}

    # cv=5 означает 5-фолдовую кросс-валидацию
    grid_search = GridSearchCV(
        Ridge(),
        param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
    )
    grid_search.fit(x_train_scaled, y_train)

    print(f"Лучшее значение alpha: {grid_search.best_params_['alpha']}")
    print(f"Лучший RMSE на кросс-валидации: {-grid_search.best_score_:.2f}")

    return grid_search


def build_preprocessor(numeric_features):
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ]
    )


def train_ridge_pipeline(x_train, x_test, y_train, y_test, preprocessor, alpha):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=alpha)),
    ])

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    rmse = calculate_rmse(y_test, y_pred)

    return pipeline, rmse


def train_lasso_pipeline(x_train, x_test, y_train, y_test, preprocessor):
    pipeline_lasso = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Lasso(alpha=0.1)),
    ])

    pipeline_lasso.fit(x_train, y_train)

    zero_weights = np.sum(pipeline_lasso.named_steps['model'].coef_ == 0)
    total_weights = len(pipeline_lasso.named_steps['model'].coef_)
    print(f"Lasso занулил {zero_weights} признаков из {total_weights}!")

    rmse = calculate_rmse(y_test, pipeline_lasso.predict(x_test))

    return pipeline_lasso, rmse


def train_without_extreme_errors(pipeline, x_train, x_test, y_train, y_test):
    y_train_pred = pipeline.predict(x_train)
    errors = (y_train - y_train_pred) ** 2

    threshold = np.quantile(errors, 0.95)
    mask = errors < threshold

    x_train_clean = x_train[mask]
    y_train_clean = y_train[mask]

    pipeline.fit(x_train_clean, y_train_clean)
    y_pred_clean = pipeline.predict(x_test)
    rmse_clean = calculate_rmse(y_test, y_pred_clean)

    return rmse_clean, errors
