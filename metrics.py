import matplotlib.pyplot as plt
from pathlib import Path
from data_dumps import Dump
import numpy as np
from utils import parse_args, get_output_directory_experiment
from properties import ID, NAME, ANNOTATIONS

def plot_metrics(output_directories: Path, configuration_list: dict = None):
    plt.figure(figsize=(12, 8))
    colors =  ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for exp_idx, output_directory in enumerate(output_directories):
        color = colors[exp_idx]
        dumps_list = sorted(list(output_directory.glob("*.json")))
        full_data = [Dump.load_json(dump) for dump in dumps_list]
        epochs = [0] + [d['epoch']+1 for d in full_data]
        # np.interp(epochs, epochs, epochs)

        train_losses = np.array([d['training_loss'] for d in full_data])
        train_losses = train_losses.flatten()
        epochs_step = np.arange(len(train_losses))*epochs[-1]/len(train_losses)
        val_losses = [d['validation_loss'] for d in full_data]
        if configuration_list is None:
            configuration = full_data[0]['configuration']
        else:
            configuration = configuration_list[exp_idx]
        name = ""
        for key in [ID, NAME, ANNOTATIONS]:
            name += f"{configuration[key]} "
        plt.plot(epochs_step, train_losses, "-o", color=color, label=f"{name} train loss")
        plt.plot(epochs[1:], val_losses, "-o", color=color, label=f"{name} valid loss")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == '__main__':
    args = parse_args(help="Plot metrics")
    exp_dir = [get_output_directory_experiment(exp) for exp in args.exp_list]
    plot_metrics(exp_dir)