import matplotlib.pyplot as plt
from pathlib import Path
from data_dumps import Dump
import numpy as np
from utils import get_default_parser, get_output_directory_experiment
from properties import ID, NAME, ANNOTATIONS
from texttable import Texttable
import latextable


def plot_metrics(results: dict) -> None:
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for exp_idx, (exp_id, res) in enumerate(results.items()):
        color = colors[exp_idx]
        plt.plot(res["timeline_steps"], res["train_losses"], "-.", alpha=0.5, color=color,
                 label=f"{res['name']} train loss")
        plt.plot(res["epochs"][1:], res["val_losses"], "-o", color=color,
                 label=f"{res['name']} valid loss")
    plt.legend()
    plt.grid()
    plt.show()


def get_results(output_directories: Path, configuration_list: dict = None) -> dict:
    results = {}
    for exp_idx, output_directory in enumerate(output_directories):
        dumps_list = sorted(list(output_directory.glob("*.json")))
        full_data = [Dump.load_json(dump) for dump in dumps_list]
        epochs = [0] + [d['epoch']+1 for d in full_data]
        train_losses = np.array([d['training_loss'] for d in full_data])
        train_losses = train_losses.flatten()
        timeline_steps = (1+np.arange(len(train_losses)))*epochs[-1]/(len(train_losses))
        val_losses = [d['validation_loss'] for d in full_data]
        if configuration_list is None:
            configuration = full_data[0]['configuration']
        else:
            configuration = configuration_list[exp_idx]
        name = ""
        for key in [ID, NAME, ANNOTATIONS]:
            name += f"{configuration[key]} "
        results[configuration[ID]] = {
            "name": name,
            "epochs": epochs,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "timeline_steps": timeline_steps,
            "configuration": configuration
        }
        # TODO: store this in a file
    return results


def format_hyper_params(hyper_params: dict) -> str:
    return "\n".join([f"{key}={value}" for key, value in hyper_params.items()])


def get_table(results: dict, caption="Impact of feature reduction", table_label="input_features_reduction"):
    table = Texttable(max_width=150)
    table.set_deco(Texttable.HEADER)
    header = ["Experience\nID",  "Validation Loss", "Epochs", "Model", "Details", "Hyper params"]
    table_content = []
    for exp_id, res in results.items():
        table_content.append([
            exp_id,
            np.array(res["val_losses"]).min(),
            res["epochs"][-1],
            res["configuration"][NAME],
            "\n".join(res["configuration"][ANNOTATIONS].split(" - ")),
            format_hyper_params(res["configuration"]["optimizer"]),
           
        ])
    table.add_rows([
        header,
        *table_content
    ])
    print(table.draw())
    print(latextable.draw_latex(
        table,
        caption=caption,
        label=f"table:{table_label.replace(' ', '_')}"))


if __name__ == '__main__':
    parser = get_default_parser(help="Plot metrics")
    parser.add_argument("-t", "--table", action="store_true", help="Print table")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot Curves")
    args = parser.parse_args()
    exp_dir = [get_output_directory_experiment(exp) for exp in args.exp_list]

    results = get_results(exp_dir, configuration_list=None)
    if args.table:
        get_table(results)
    if args.plot:
        plot_metrics(results)
