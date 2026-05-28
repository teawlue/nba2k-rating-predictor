from .config import DATA_PATH, setup_environment
from .data import get_numeric_features, load_data, prepare_features, split_data
from .modeling import (
    build_preprocessor,
    train_baseline,
    train_lasso_pipeline,
    train_ridge_numeric,
    train_ridge_pipeline,
    train_scaled_ridge,
    train_without_extreme_errors,
    tune_ridge_alpha,
)
from .plots import (
    plot_feature_importance,
    plot_model_comparison,
    plot_target_distribution,
    plot_train_errors,
)
from .reporting import (
    build_results,
    print_feature_interpretation,
    print_final_report,
)


def run_project():
    setup_environment()

    data = load_data(DATA_PATH)
    plot_target_distribution(data)

    x, y = prepare_features(data)
    x_train, x_test, y_train, y_test = split_data(x, y)
    numeric_features = get_numeric_features(x_train)

    rmse_baseline = train_baseline(y_train, y_test)
    _, rmse_ridge, x_train_num, x_test_num = train_ridge_numeric(
        x_train,
        x_test,
        y_train,
        y_test,
        numeric_features,
    )

    _, weights_scaled, x_train_scaled, _ = train_scaled_ridge(
        x_train_num,
        x_test_num,
        y_train,
        numeric_features,
    )
    plot_feature_importance(weights_scaled)
    print_feature_interpretation(weights_scaled)

    grid_search = tune_ridge_alpha(x_train_scaled, y_train)

    preprocessor = build_preprocessor(numeric_features)
    pipeline, rmse_pipe = train_ridge_pipeline(
        x_train,
        x_test,
        y_train,
        y_test,
        preprocessor,
        grid_search.best_params_['alpha'],
    )
    _, rmse_lasso = train_lasso_pipeline(
        x_train,
        x_test,
        y_train,
        y_test,
        preprocessor,
    )

    rmse_clean, errors = train_without_extreme_errors(
        pipeline,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    plot_train_errors(errors)

    results = build_results(
        rmse_baseline,
        rmse_ridge,
        rmse_pipe,
        rmse_lasso,
        rmse_clean,
    )
    print_final_report(results, rmse_baseline)
    plot_model_comparison(results)
