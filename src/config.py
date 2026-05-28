import warnings

import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = 'data/nba_rankings_2014-2020.csv'
TARGET_COLUMN = 'rankings'
DROP_COLUMNS = ['Unnamed: 0', 'PLAYER', TARGET_COLUMN]
CATEGORICAL_FEATURES = ['TEAM', 'SEASON']
RANDOM_STATE = 42
TEST_SIZE = 0.2


def setup_environment():
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
    warnings.filterwarnings('ignore')
