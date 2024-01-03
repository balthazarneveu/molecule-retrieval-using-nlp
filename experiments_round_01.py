from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
)
from pathlib import Path
from multimodal_model import MultimodalModel
from language_model import TextEncoder, Adapter
from graph_model import BasicGraphEncoder, BigGraphEncoder


def get_round_1_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """Freeze BERT and train adapter + base GCN"""
    if exp == 100:
        configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-5
        configuration[OPTIMIZER][WEIGHT_DECAY] = 1.  # OOPS!!!!!!!!!!!!!!!!
        configuration[NAME] = 'FBERT-GCN'
        configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True)
        model = MultimodalModel(graph_encoder, text_encoder)
    if exp == 101:
        configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01
        configuration[NAME] = 'FBERT-GCN'
        configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True)
        model = MultimodalModel(graph_encoder, text_encoder)
    if exp == 102:
        configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01
        configuration[NAME] = 'FBERT-GCN'
        configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True)
        model = MultimodalModel(graph_encoder, text_encoder)
    if exp == 103:
        configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-3
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01
        configuration[NAME] = 'FBERT-GCN'
        configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True)
        model = MultimodalModel(graph_encoder, text_encoder)
    if exp == 104:
        configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-5
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01
        configuration[NAME] = 'FBERT-GCN'
        configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True)
        model = MultimodalModel(graph_encoder, text_encoder)
    if exp == 105:
        configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-5
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01
        configuration[NAME] = 'FBERT-GCN'
        configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True)
        model = MultimodalModel(graph_encoder, text_encoder)
    if exp == 106:
        configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01
        configuration[NAME] = 'FBERT-ADAPT-GCN'
        configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        shared_out_size = 256
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=shared_out_size,
                                          nhid=300, graph_hidden_channels=300)
        text_adapter = Adapter(text_embedding_size=768, out_size=shared_out_size)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True, adapter=text_adapter)
        model = MultimodalModel(graph_encoder, text_encoder)

    if exp == 110 or exp == 112:
        configuration[TOKENIZER_NAME] = "allenai/scibert_scivocab_uncased"
        configuration[BATCH_SIZE] = (8, 8, 8)
        if exp == 112:
            configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01
        configuration[NAME] = 'F-SciBERT-ADAPT-GCN'
        configuration[ANNOTATIONS] = 'Frozen SciBERT - Trainable GCN'
        shared_out_size = 256
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=shared_out_size,
                                          nhid=300, graph_hidden_channels=300)
        text_adapter = Adapter(text_embedding_size=768, out_size=shared_out_size)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True, adapter=text_adapter)
        model = MultimodalModel(graph_encoder, text_encoder)
    if exp == 111 or exp == 113:
        configuration["GCN-architecture"] = {
            "depth": 5,
            "GCN-FC-hidden-size": 512,
            "GCN-hidden-size": 256,
            "GNN-out-size": 768,
        }
        configuration[TOKENIZER_NAME] = "allenai/scibert_scivocab_uncased"
        configuration[BATCH_SIZE] = (8, 8, 8)
        if exp == 113:
            configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 1e-4
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.01
        configuration[NAME] = 'F-SciBERT-ADAPT-Bigger-GCN'
        configuration[ANNOTATIONS] = 'Frozen SciBERT - Trainable bigger GCN'
        shared_out_size = 512
        graph_encoder = BigGraphEncoder(num_node_features=300, nout=shared_out_size,
                                        nhid=256, graph_hidden_channels=512)
        text_adapter = Adapter(text_embedding_size=768, out_size=shared_out_size)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True, adapter=text_adapter)
        model = MultimodalModel(graph_encoder, text_encoder)
    return model, configuration
