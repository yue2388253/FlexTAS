import matplotlib.pyplot as plt
import os.path
from stable_baselines3.common.results_plotter import plot_results

from src.lib.execute import execute_from_command_line


def plot_training_rewards(dirname: str):
    plot_results([dirname], None, 'timesteps', next((s for s in dirname.split(r'/') if '100' in s), None))
    filename = os.path.join(os.path.dirname(dirname), f"training_reward.png")
    print(f"saving the figure to {filename}")
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    execute_from_command_line(plot_training_rewards)
