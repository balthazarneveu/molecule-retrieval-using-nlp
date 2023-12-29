import kaggle
from pathlib import Path
import json
from __kaggle_login import kaggle_users
import argparse
import sys
import subprocess


def get_git_branch_name():
    try:
        branch_name = subprocess.check_output(["git", "branch", "--show-current"]).strip().decode()
        return branch_name
    except subprocess.CalledProcessError:
        return "Error: Could not determine the Git branch name."


def prepare_notebook(
    output_nb_path: Path,
    exp: int,
    branch: str,
    template_nb_path: Path = "remote_training_template.ipynb"
):
    with open(template_nb_path) as f:
        template_nb = f.readlines()
        for line_idx, li in enumerate(template_nb):
            if "!!!exp!!!" in li:
                template_nb[line_idx] = template_nb[line_idx].replace("!!!exp!!!", f"{exp}")
            if "!!!branch!!!" in li:
                template_nb[line_idx] = template_nb[line_idx].replace("!!!branch!!!", f"\'{branch}\'")
        template_nb = "".join(template_nb)
    with open(output_nb_path, "w") as w:
        w.write(template_nb)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=str, help="Kaggle user", choices=list(kaggle_users.keys()))
    parser.add_argument("-e", "--exp", type=int, required=True, help="Experiment id")
    parser.add_argument("--branch", type=str, help="Git branch name", default=get_git_branch_name())
    parser.add_argument("-p", "--push", action="store_true", help="Push")
    args = parser.parse_args(argv)
    kaggle_user = kaggle_users[args.user]
    uname_kaggle = kaggle_user["username"]
    kaggle.api._load_config(kaggle_user)
    kernel_root = Path(f"__nb_{uname_kaggle}")
    kernel_root.mkdir(exist_ok=True, parents=True)
    exp = args.exp
    kernel_path = kernel_root/f"{exp:04d}"
    kernel_path.mkdir(exist_ok=True, parents=True)
    branch = args.branch
    print(branch)

    nb_id = "train-molecule-nlp"
    config = {
        "id": f"{kaggle_user['username']}/{nb_id}",
        # "id_no" : 3,
        "title": nb_id.lower(),
        "code_file": f"{nb_id}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["balthazarneveu/altegrad23-processed-distilbert-base-uncased"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    prepare_notebook((kernel_path/nb_id).with_suffix(".ipynb"), exp, branch)
    assert (kernel_path/nb_id).with_suffix(".ipynb").exists()
    with open(kernel_path/"kernel-metadata.json", "w") as f:
        json.dump(config, f, indent=4)

    if args.push:
        kaggle.api.kernels_push_cli(str(kernel_path))


if __name__ == '__main__':
    main(sys.argv[1:])
