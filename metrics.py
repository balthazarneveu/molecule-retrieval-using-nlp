import matplotlib.pyplot as plt
from pathlib import Path
from data_dumps import Dump
import numpy as np


def plot_metrics(output_directory: Path, configuration: dict = None):
    dumps_list = sorted(list(output_directory.glob("*.json")))
    full_data = [Dump.load_json(dump) for dump in dumps_list]
    epochs = [d['epoch'] for d in full_data]
    # np.interp(epochs, epochs, epochs)

    train_losses = np.array([d['training_loss'] for d in full_data])
    train_losses = train_losses.flatten()
    epochs_step = np.interp(np.arange(len(train_losses))/epochs[-1], epochs, epochs)
    val_losses = [d['validation_loss'] for d in full_data]
    if configuration is None:
        configuration = full_data[0]['configuration']
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_step, train_losses, "-o", label="train loss")
    plt.plot(epochs, val_losses, "-o", label="valid loss")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot_metrics(Path("__output/0000_check-pipeline-BERT-GCN"))
