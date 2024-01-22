from properties import (
    SCHEDULER, SCHEDULER_CONFIGURATION,
    DISTILBERT, BIG_GCN
)
from pathlib import Path
from experiments_generic import generic_experiment, custom_lr
from functools import partial


def get_round_6_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """"""
    assert exp >= 600 and exp < 699, f"Experiment {exp} is not in the round 90"
    configuration["max_step_per_epoch"] = 5
    if exp == 600:
        lr = 6e-4
        model, configuration = generic_experiment(
            configuration,
            llm=DISTILBERT, graph=BIG_GCN,
            n=200,
            b=32, lr=lr, wd=1e-1,
            lora=False, quantization=None
        )
        configuration[SCHEDULER] = "LambdaLR"
        lr_lambda = partial(
            custom_lr, warmup=40,
            lr_init=lr, lr_min=1e-5, lr_tmp=1e-4,
            period_oscillation=20,
            periods_dampen=5
        )
        configuration[SCHEDULER_CONFIGURATION] = dict(lr_lambda=lr_lambda)
    print(configuration)
    return model, configuration
