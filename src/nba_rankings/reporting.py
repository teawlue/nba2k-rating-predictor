import pandas as pd


def build_results(rmse_baseline, rmse_ridge, rmse_pipe, rmse_lasso, rmse_clean):
    return pd.DataFrame({
        'Модель': [
            'Baseline (среднее)',
            'Ridge',
            'Ridge + TEAM/SEASON',
            'Lasso + TEAM/SEASON',
            'Ridge после удаления выбросов',
        ],
        'RMSE': [
            rmse_baseline,
            rmse_ridge,
            rmse_pipe,
            rmse_lasso,
            rmse_clean,
        ],
    }).sort_values('RMSE')


def print_final_report(results, rmse_baseline):
    best_model = results.iloc[0]

    print("\nИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print(results.to_string(index=False, formatters={'RMSE': '{:.2f}'.format}))
    print(
        f"\nЛучшая модель: {best_model['Модель']} "
        f"с RMSE = {best_model['RMSE']:.2f}"
    )
    print(
        f"Модель ошибается в среднем примерно на {best_model['RMSE']:.2f} / "
        f"{rmse_baseline:.2f} * 100% = {best_model['RMSE'] / rmse_baseline * 100:.2f}% "
        "процентов от ошибки базовой модели, которая просто предсказывает среднее."
    )
