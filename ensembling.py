from properties import OUT_DIR
import pandas as pd
from typing import List
from pathlib import Path


def ensemble_models_average(submissions_list: List[Path], output: Path) -> None:
    """Ensemble models by averaging their predictions"""
    dfs = [pd.read_csv(sub, index_col='ID') for sub in submissions_list]
    avg_df = pd.concat(dfs).groupby(level=0).mean()
    avg_df.reset_index(inplace=True)
    avg_df.to_csv(output, index=False)


if __name__ == "__main__":
    submissions_list = [
        OUT_DIR/"9008_BERT-biggerGCN"/"submission.csv",
        OUT_DIR/"9009_BERT - FatGCN"/"submission_original.csv",
        OUT_DIR/"9010_BERT - FatGCN"/"submission.csv",
        OUT_DIR/"9011_BERT-biggerGCN"/"submission.csv",
        OUT_DIR/"0611_LoraSciBERT-GCN - FatGCN Pretrained on 573/submission.csv",
        OUT_DIR/"9075_BERT - FatGCN/submission.csv",
        OUT_DIR/"9077_BERT - FatGCN/submission.csv",
    ]
    ensemble_models_average(
        submissions_list,
        OUT_DIR/"submission_mean_of_dataframes_611_9008_9009_9010_9011_9075_9077.csv"
    )
