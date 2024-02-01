from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
    MAX_STEP_PER_EPOCH, BETAS, OPTIMIZER_STATE_DICT, SCHEDULER, SCHEDULER_CONFIGURATION

)
from pathlib import Path
import torch
from baseline_model import Model as BaselineModel
from utils import reload_model_and_optimizer_state


def get_baseline_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """
    Baseline = trainable BERT + base GCN
    https://github.com/balthazarneveu/molecule-retrieval-using-nlp/issues/12
    """
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
    if exp == 13:
        configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-6
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.2
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - RTX2060'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 14:
        configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-5
        configuration[OPTIMIZER][WEIGHT_DECAY] = 1.  # OOPS!
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers-resume-8'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)
        if backup_root is not None:
            pretrained_model_path = backup_root/'0008_Baseline-BERT-GCN/model_0038.pt'
        elif root_dir is not None:
            pretrained_model_path = root_dir/'__output'/'0008_Baseline-BERT-GCN/model_0038.pt'
        else:
            raise ValueError("No root_dir or backup_root provided")
        assert pretrained_model_path.exists(), f"Pretrained model not found at {pretrained_model_path}"
        model.load_state_dict(
            torch.load(pretrained_model_path, map_location='cpu')['model_state_dict'])
    if exp == 15:
        configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 2e-6
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - resume 8'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
        if backup_root is not None:
            pretrained_model_path = backup_root/'0008_Baseline-BERT-GCN/model_0058.pt'
        elif root_dir is not None:
            pretrained_model_path = root_dir/'__output'/'0008_Baseline-BERT-GCN/model_0058.pt'
        else:
            raise ValueError("No root_dir or backup_root provided")
        assert pretrained_model_path.exists(), f"Pretrained model not found at {pretrained_model_path}"
        model.load_state_dict(
            torch.load(pretrained_model_path, map_location='cpu')['model_state_dict'])
    if exp == 16:
        configuration[BATCH_SIZE] = (64, 64, 64)    # Kaggle
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 2e-6
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 17:
        configuration[BATCH_SIZE] = (64, 64, 64)    # Kaggle
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-6
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim

    if exp == 18: #baseline experiece to try with the new loss
        configuration[BATCH_SIZE] = (32, 32, 32)    
        configuration[NB_EPOCHS] = 80
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)
    if exp == 19: #18 with more epochs and bigger batch size
        configuration[BATCH_SIZE] = (64, 32, 32)    
        configuration[NB_EPOCHS] = 150
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)
    if exp == 20: # with more epochs and bigger batch size
        configuration[BATCH_SIZE] = (128, 32, 32)    
        configuration[NB_EPOCHS] = 150
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)
    f exp == 21: # with more epochs and bigger batch size
        configuration[BATCH_SIZE] = (164, 32, 32)    
        configuration[NB_EPOCHS] = 180
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers'
        model = BaselineModel(
            model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
            nhid=300, graph_hidden_channels=300)

    if exp >= 30 and exp < 60:
        model, configuration = get_best_pretrained_model(configuration, root_dir, backup_root)
        configuration[NB_EPOCHS] = 16
        if exp >= 30 and exp < 40:
            milestone_index = 30
            configuration[BATCH_SIZE] = (8, 8, 8)    # T500
        elif exp >= 40 and exp < 50:
            milestone_index = 40
            configuration[BATCH_SIZE] = (16, 16, 16)    # T500
        elif exp >= 50:
            milestone_index = 50
            configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        delta_index = exp - milestone_index
        configuration[OPTIMIZER][LEARNING_RATE] = [
            1e-9, 1e-8, 1e-7, 1e-6, 2e-6][delta_index]
        configuration[OPTIMIZER].pop(BETAS)
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.
    if exp >= 60:
        configuration[NAME] = 'Base-' + 'HP-search'
        model = BaselineModel(model_name=configuration[TOKENIZER_NAME], num_node_features=300,
                              nout=768, nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
        configuration[NB_EPOCHS] = 120
        configuration[OPTIMIZER] = {
            LEARNING_RATE: 2e-6,
            WEIGHT_DECAY: 0.3,
            BETAS: [0.9, 0.999]  # Default ADAM parameters
        }
        batch_size_val = None
        if exp == 60:
            configuration[NB_EPOCHS] = 150
            batch_size = 64  # 96 or 128 Does not work even on Kaggle.
            configuration[OPTIMIZER] = {
                LEARNING_RATE: 2e-6,
                WEIGHT_DECAY: 0.3,
            }
        if exp == 61:
            configuration[OPTIMIZER] = {
                LEARNING_RATE: 2e-6,
                WEIGHT_DECAY: 0.3,
            }
            batch_size = 32  # suitable for RTX 2060
        if exp == 62:
            configuration[OPTIMIZER] = {
                LEARNING_RATE: 1e-6,
                WEIGHT_DECAY: 0.3,
            }
            batch_size = 32  # suitable for RTX 2060
        if exp == 63:
            configuration[OPTIMIZER] = {
                LEARNING_RATE: 5e-7,
                WEIGHT_DECAY: 0.3,
            }
            batch_size = 32  # suitable for RTX 2060
        if exp == 64:
            configuration[OPTIMIZER] = {
                LEARNING_RATE: 2e-6,
                WEIGHT_DECAY: 0.5,
            }
            batch_size = 32  # suitable for RTX 2060
        if exp == 65:
            configuration[NB_EPOCHS] = 70
            batch_size = 64
            configuration[OPTIMIZER] = {
                LEARNING_RATE: 7e-6,
                WEIGHT_DECAY: 0.3,
            }
        if exp == 66:
            configuration[NB_EPOCHS] = 70
            batch_size = 32
            configuration[OPTIMIZER] = {
                LEARNING_RATE: 7e-6,
                WEIGHT_DECAY: 0.3,
            }
            model, configuration = reload_model_and_optimizer_state(
                65, backup_root=backup_root, configuration=configuration, model=model)
        if exp == 68: # BEST PERFORMING EXPERIMENT
            configuration[NB_EPOCHS] = 120
            batch_size = 128
            configuration[OPTIMIZER] = {
                LEARNING_RATE: 7e-6,
                WEIGHT_DECAY: 0.3, # Copy of 65
            }
        if exp == 69:
            configuration[NAME] = 'Base-' + 'LR-sched'
            configuration[NB_EPOCHS] = 150
            batch_size = 128
            configuration[OPTIMIZER] = {
                LEARNING_RATE: 3e-4,
                WEIGHT_DECAY: 0.3,
            }
            configuration[SCHEDULER] = "ReduceLROnPlateau"
            configuration[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
        if batch_size_val is None:
            batch_size_val = batch_size
        configuration[BATCH_SIZE] = (batch_size, batch_size_val, batch_size_val)
    return model, configuration


def get_best_pretrained_model(configuration, root_dir: Path = None, backup_root: Path = None):
    configuration[BATCH_SIZE] = (32, 32, 32)    # T500
    configuration[NB_EPOCHS] = 60
    configuration[OPTIMIZER][LEARNING_RATE] = 2e-6
    configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
    configuration[NAME] = 'Base-' + 'HP-search'
    configuration[ANNOTATIONS] = 'Baseline - provided by organizers - resume 15'
    model = BaselineModel(
        model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
        nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if backup_root is not None:
        pretrained_model_path = backup_root/'0015_Baseline-BERT-GCN/model_0039.pt'
    elif root_dir is not None:
        pretrained_model_path = root_dir/'__output'/'0015_Baseline-BERT-GCN/model_0039.pt'
    else:
        raise ValueError("No root_dir or backup_root provided")
    assert pretrained_model_path.exists(), f"Pretrained model not found at {pretrained_model_path}"
    model.load_state_dict(
        torch.load(pretrained_model_path, map_location='cpu')['model_state_dict'])
    return model, configuration
