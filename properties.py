from pathlib import Path
NB_EPOCHS = "nb_epochs"
BATCH_SIZE = "batch_size"
MODEL_SIZE = "model_size"
LEARNING_RATE = "lr"
TOKENIZER_NAME = "tokenizer_model_name"
NAME = "name"
ANNOTATIONS = "annotations"
WEIGHT_DECAY = "weight_decay"
BETAS = "betas"
OPTIMIZER = "optimizer"
TRAIN, VALIDATION, TEST = "train", "validation", "test"
DEVICE = "device"
ID = "id"
PLATFORM = "platform"
MAX_STEP_PER_EPOCH = "max_step_per_epoch"
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / '__data'
OUT_DIR = ROOT_DIR/'__output'
OUT_DIR.mkdir(exist_ok=True, parents=True)
