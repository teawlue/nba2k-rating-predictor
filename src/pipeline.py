from .config import data_path, setup_environment
from .data import (
    get_numeric_features,
    load_data,
    prepare_features,
    split_data,
)
from .modeling import (
    train_baseline,
    train_ridge_numeric,
    train_ridge_with_alpha,
    train_scaled_ridge,
    train_without_extreme_errors,
    tune_ridge_alpha,
)
from .plots import (
    plot_correlation_matrix,
    plot_feature_importance,
    plot_feature_vs_target,
    plot_model_comparison,
    plot_rating_boxplot,
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

    data = load_data(data_path)

    plot_target_distribution(data)
    plot_correlation_matrix(data)
    plot_feature_vs_target(data, 'PTS')
    plot_rating_boxplot(data)

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

    _, weights_scaled, x_train_scaled, x_test_scaled = train_scaled_ridge(
        x_train_num,
        x_test_num,
        y_train,
        numeric_features,
    )
    plot_feature_importance(weights_scaled)
    print_feature_interpretation(weights_scaled)

    grid_search = tune_ridge_alpha(x_train_scaled, y_train)

    best_ridge, rmse_best_ridge = train_ridge_with_alpha(
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
        grid_search.best_params_['alpha'],
    )

    rmse_clean, errors = train_without_extreme_errors(
        best_ridge,
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
    )
    plot_train_errors(errors)

    results = build_results(
        rmse_baseline,
        rmse_ridge,
        rmse_best_ridge,
        rmse_clean,
    )
    print_final_report(results, rmse_baseline)
    plot_model_comparison(results)
