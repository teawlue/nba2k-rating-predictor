import pandas as pd


FEATURE_LABELS = {
    'AGE': 'возраст',
    'GP': 'игры',
    'W': 'победы',
    'L': 'поражения',
    'MIN': 'минуты',
    'PTS': 'очки',
    'FGM': 'попадания с игры',
    'FGA': 'броски с игры',
    'FG%': 'процент с игры',
    '3PM': 'попадания трехочковых',
    '3PA': 'попытки трехочковых',
    '3P%': 'процент трехочковых',
    'FTM': 'попадания штрафных',
    'FTA': 'попытки штрафных',
    'FT%': 'процент штрафных',
    'OREB': 'подборы в нападении',
    'DREB': 'подборы в защите',
    'REB': 'подборы',
    'AST': 'передачи',
    'TOV': 'потери',
    'STL': 'перехваты',
    'BLK': 'блок-шоты',
    'PF': 'фолы',
    'FP': 'fantasy points',
    'DD2': 'дабл-даблы',
    'TD3': 'трипл-даблы',
    '+/-': 'плюс-минус',
}


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


def format_feature_name(feature):
    label = FEATURE_LABELS.get(feature)
    if label is None:
        return feature
    return f"{feature} ({label})"


def build_feature_interpretation(weights, top_n=10):
    top_positive = weights.sort_values(ascending=False).head(top_n)
    top_negative = weights.sort_values(ascending=True).head(top_n)
    strongest = weights.abs().sort_values(ascending=False).head(top_n).index
    strongest_labels = [format_feature_name(feature) for feature in strongest]

    return top_positive, top_negative, strongest_labels


def print_feature_interpretation(weights, top_n=10):
    top_positive, top_negative, strongest_labels = build_feature_interpretation(
        weights,
        top_n,
    )

    print("\nИНТЕРПРЕТАЦИЯ ПРИЗНАКОВ")

    print(f"\nТоп-{top_n} положительных признаков:")
    for feature, value in top_positive.items():
        print(f"{format_feature_name(feature):35s} {value:8.3f}")

    print(f"\nТоп-{top_n} отрицательных признаков:")
    for feature, value in top_negative.items():
        print(f"{format_feature_name(feature):35s} {value:8.3f}")

    print(
        "\nСильнее всего рейтинг связан с признаками: "
        f"{', '.join(strongest_labels)}."
    )


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
