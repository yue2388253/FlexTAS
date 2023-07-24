import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import plot_results

from src.lib.execute import execute_from_command_line


def plot_training_rewards(dirname: str):
    plot_results([dirname], None, 'timesteps', 'training')
    plt.show()


if __name__ == '__main__':
    execute_from_command_line(plot_training_rewards)
