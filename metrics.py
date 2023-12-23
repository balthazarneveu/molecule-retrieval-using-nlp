import matplotlib.pyplot as plt
from pathlib import Path
from data_dumps import Dump
import numpy as np
from utils import get_default_parser, get_output_directory_experiment
from properties import ID, NAME, ANNOTATIONS, BATCH_SIZE, MODEL_SIZE, TRAIN
from texttable import Texttable
import latextable
import logging


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
        assert len(full_data) > 0, f"No data found in {output_directory}"
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


def fetch_kaggle_scores(competition_name="altegrad-2023-data-challenge"):
    import kaggle
    kaggle.api.authenticate()
    submissions = kaggle.api.competition_submissions(competition_name)
    scores = {}
    for sub in submissions:
        if sub.ref is None:
            continue
        try:
            exp_id = int(sub.description.lower().split("exp_")[1].split(" ")[0])
            if exp_id in scores.keys():
                if float(sub.publicScore) < scores[exp_id]["score"]:
                    logging.warning(f"Found a better score for {exp_id} {sub.ref} \n{sub.description}")
                    continue
            scores[exp_id] = {
                "ref": sub.ref,
                "date": sub.date,
                "comment": sub.description,
                "score": float(sub.publicScore)
            }
        except Exception as e:
            logging.warning(f"Failed to parse submission {sub.ref} \n{sub.description} with error {e}")
    return scores


def get_table(results: dict, kaggle_results={},
              caption="Model performances",
              table_label="input_features_reduction"):
    table = Texttable(max_width=150)
    table.set_deco(Texttable.HEADER)
    header = [
        "Experience\nID",  "Score [%]", "Validation Loss", "Epochs", "Model Name",  "Model\nSize [M]", "Batch\nsize",
        "Hyper params",  "Details"]
    table_content = []
    for exp_id, res in results.items():
        kaggle_res = kaggle_results.get(exp_id, {"score": "N/A"})
        kaggle_score = kaggle_res.get("score", "N/A")
        table_content.append([
            exp_id,
            kaggle_score,
            np.array(res["val_losses"]).min(),
            res["epochs"][-1],
            res["configuration"][NAME],
            res["configuration"].get(MODEL_SIZE, 0)*1E-6,
            res["configuration"].get(BATCH_SIZE, {TRAIN: "N/A"})[TRAIN],
            format_hyper_params(res["configuration"]["optimizer"]),
            "\n".join(res["configuration"][ANNOTATIONS].split(" - ")),

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
    parser.add_argument("-nok", "--disable-kaggle", action="store_true", help="Disable Kaggle fetching scores")
    args = parser.parse_args()
    exp_dir = [get_output_directory_experiment(exp) for exp in args.exp_list]

    results = get_results(exp_dir, configuration_list=None)
    if args.table:
        kaggle_results = {} if args.disable_kaggle else fetch_kaggle_scores()
        get_table(results, kaggle_results=kaggle_results)
    if args.plot:
        plot_metrics(results)
