from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
)
from pathlib import Path
from multimodal_model import MultimodalModel
from language_model import TextEncoder
from graph_model import BasicGraphEncoder
from peft import TaskType


def get_round_5_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """Use LoRA to drastically reduce the number of parameters of the model

    https://github.com/balthazarneveu/molecule-retrieval-using-nlp/issues/19
    """
    assert exp >= 500 and exp <= 599, "round 5 between 500 and 599"
    if exp == 500:
        configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        # configuration[BATCH_SIZE] = (16, 16, 16)    # T500
        configuration[NB_EPOCHS] = 150
        configuration[OPTIMIZER][LEARNING_RATE] = 7e-6
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'LoraBERT-GCN'
        configuration[ANNOTATIONS] = 'Trainable Lora Dsitil BERT - base GCN'
        configuration[TOKENIZER_NAME] = "distilbert-base-uncased"
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        if "distil" in configuration[TOKENIZER_NAME]:
            target_modules = {
                "q_lin",
                "k_lin",
                "v_lin",
                "out_lin",
            }
        else:
            target_modules = {
                "query",
                "key",
                "value",
                "dense",
            }
        lora_dict = dict(
            task_type=TaskType.CAUSAL_LM,
            # task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            target_modules=target_modules,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        # lora_dict = {}
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=False, lora=lora_dict)
        model = MultimodalModel(graph_encoder, text_encoder)
    return model, configuration
