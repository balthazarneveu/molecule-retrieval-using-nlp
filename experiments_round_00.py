from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
    MAX_STEP_PER_EPOCH
)
from pathlib import Path
import torch
from baseline_model import Model as BaselineModel


def get_baseline_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    assert exp >= 0 and exp <= 99, "baseline exp must be <= 99"
    if exp == 0:
        configuration[NAME] = 'check-pipeline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Check pipeline'
        configuration[MAX_STEP_PER_EPOCH] = 5
        model = BaselineModel(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                              nhid=8, graph_hidden_channels=8)
    if exp == 1:
        configuration[NB_EPOCHS] = 5
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers'
        model = BaselineModel(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                              nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 2:
        configuration[NB_EPOCHS] = 5
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - 5 epochs'
        model = BaselineModel(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                              nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 3:
        configuration[NB_EPOCHS] = 30
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - 30 epochs'
        model = BaselineModel(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                              nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 4:
        configuration[BATCH_SIZE] = (32, 64, 64)  # Collab
        configuration[NB_EPOCHS] = 30
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - Collab'
        model = BaselineModel(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                              nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 5:
        configuration[BATCH_SIZE] = (96, 64, 64)  # Collab
        configuration[NB_EPOCHS] = 60
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - Collab'
        model = BaselineModel(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                              nhid=600, graph_hidden_channels=600)  # nout = bert model hidden dim
    if exp == 6:
        configuration[BATCH_SIZE] = (64, 64, 64)  # Collab
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-5
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.05
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - Collab - Restart exp 5'
        model = BaselineModel(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                              nhid=600, graph_hidden_channels=600)  # nout = bert model hidden dim
        if backup_root is not None:
            pretrained_model_path = backup_root/'0005_Baseline-BERT-GCN/model_0009.pt'
        elif root_dir is not None:
            pretrained_model_path = root_dir/'__output'/'0005_Baseline-BERT-GCN/model_0009.pt'
        else:
            raise ValueError("No root_dir or backup_root provided")
        assert pretrained_model_path.exists(), f"Pretrained model not found at {pretrained_model_path}"
        model.load_state_dict(
            torch.load(pretrained_model_path, map_location='cpu')['model_state_dict'])
    if exp == 7:
        configuration[BATCH_SIZE] = (64, 64, 64)    # Kaggle
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-6
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - Kaggle'
        model = BaselineModel(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                              nhid=600, graph_hidden_channels=600)  # nout = bert model hidden dim
    if exp == 8:
        configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-6
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - RTX2060'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 9:
        configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-3
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.3
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - RTX2060'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 10:
        configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.3
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - RTX2060'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)
    if exp == 11:
        configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-5
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.3
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - RTX2060'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)
    if exp == 12:
        configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-5
        configuration[OPTIMIZER][WEIGHT_DECAY] = 1.
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - RTX2060'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)
    return configuration, model
