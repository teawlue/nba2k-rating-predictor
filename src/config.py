import warnings

import matplotlib.pyplot as plt
import seaborn as sns


data_path = 'data/nba_rankings_2014-2020.csv'
target = 'rankings'
drop_cols = ['Unnamed: 0', 'PLAYER', 'rankings', 'SEASON', 'TEAM']
random_state = 42
test_size = 0.2


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
