from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
)
from pathlib import Path
from multimodal_model import MultimodalModel
from language_model import TextEncoder, Adapter
from graph_model import BasicGraphEncoder, BigGraphEncoder


def get_round_1_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """Freeze BERT and train adapter + base GCN"""
    if exp in [100, 101, 102, 103, 104, 105]:
        # Base GCN - Frozen BERT
        configuration[NAME] = 'FBERT-GCN'
        configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True)
        model = MultimodalModel(graph_encoder, text_encoder)
        configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        if exp == 100:
            configuration[OPTIMIZER][LEARNING_RATE] = 5e-5
            configuration[OPTIMIZER][WEIGHT_DECAY] = 1.  # OOPS!!!!!!!!!!!!!!!!
        if exp == 101:
            configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
            configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01
        if exp == 102:
            configuration[OPTIMIZER][LEARNING_RATE] = 5e-4
            configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01

        if exp == 103:
            configuration[OPTIMIZER][LEARNING_RATE] = 1e-3
            configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01

        if exp == 104:
            configuration[OPTIMIZER][LEARNING_RATE] = 1e-5
            configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01

        if exp == 105:
            configuration[OPTIMIZER][LEARNING_RATE] = 5e-5
            configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01

    if exp in [106, 110, 112]:
        # Frozen BERT or SciBERT + ADAPTER
        # Base GCN
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01
        configuration[BATCH_SIZE] = (32, 32, 32)
        if exp == 106:  # Frozen BERT + ADAPTER
            configuration[NAME] = 'FBERT-ADAPT-GCN'
            configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        if exp in [110, 112]:  # Frozen Sci BERT + ADAPTER
            configuration[TOKENIZER_NAME] = "allenai/scibert_scivocab_uncased"
            configuration[NAME] = 'F-SciBERT-ADAPT-GCN'
            configuration[ANNOTATIONS] = 'Frozen SciBERT - Trainable GCN'
            if exp == 110:
                configuration[BATCH_SIZE] = (8, 8, 8)
        shared_out_size = 256
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=shared_out_size,
                                          nhid=300, graph_hidden_channels=300)
        text_adapter = Adapter(text_embedding_size=768, out_size=shared_out_size)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True, adapter=text_adapter)
        model = MultimodalModel(graph_encoder, text_encoder)

    if exp in [111, 113, 114]:
        # Bigger GCN
        configuration["GCN-architecture"] = {
            "depth": 5,
            "GCN-FC-hidden-size": 512,
            "GCN-hidden-size": 256,
            "GNN-out-size": 768,
        }

        configuration[BATCH_SIZE] = (32, 32, 32)
        if exp == 111:
            configuration[BATCH_SIZE] = (8, 8, 8)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01
        if exp in [111, 113]:
            configuration[TOKENIZER_NAME] = "allenai/scibert_scivocab_uncased"
            configuration[NAME] = 'F-SciBERT-ADAPT-Bigger-GCN'
            configuration[ANNOTATIONS] = 'Frozen SciBERT - Trainable bigger GCN'
        if exp in [114]:
            configuration[NAME] = 'FBERT-ADAPT-Bigger-GCN'
            configuration[ANNOTATIONS] = 'Frozen BERT - Trainable bigger GCN'
        shared_out_size = 512
        graph_encoder = BigGraphEncoder(num_node_features=300, nout=shared_out_size,
                                        nhid=256, graph_hidden_channels=512)
        text_adapter = Adapter(text_embedding_size=768, out_size=shared_out_size)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True, adapter=text_adapter)
        model = MultimodalModel(graph_encoder, text_encoder)
    return model, configuration
