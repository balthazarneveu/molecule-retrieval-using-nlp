# molecule-retrieval-using-nlp
MVA23 ALTEGRAD data challenge on Molecule retrieval using natural language analyzis.


## Authors
- Program: [MVA Master's degree](https://www.master-mva.com/) class on language and graph learning [ALTEGRAD](https://www.master-mva.com/cours/cat-advanced-learning-for-text-and-graph-data-altegrad/). ENS Paris-Saclay.
- Authors
    - [Balthazar Neveu](https://github.com/balthazarneveu)
    - [Basile Terver](https://github.com/Basile-Terv)
    - [Lea Khalil](https://github.com/lea-khalil)



-----

| ![](/report/figures/logo.png) |
|:-----:|
| :medal_military: [Kaggle](https://www.kaggle.com/competitions/altegrad-2023-data-challenge) |
| :trophy: [Weights and biases](https://wandb.ai/molecule-nlp-altegrad-23) |

------
# :gear: Setup
Everything can be trained locally


To use :parrot: LORA properly on BERT model in this project, you need to use a custom version of PEFT.

```
cd ~
git clone https://github.com/huggingface/peft
cd peft
pip install -e .

echo 'diff --git a/src/peft/tuners/tuners_utils.py b/src/peft/tuners/tuners_utils.py
index d9aa7cb..a5cba30 100644
--- a/src/peft/tuners/tuners_utils.py
+++ b/src/peft/tuners/tuners_utils.py
@@ -158,6 +158,7 @@ class BaseTuner(nn.Module, ABC):
         return self.active_adapter
 
     def forward(self, *args: Any, **kwargs: Any):
+        kwargs.pop("labels", None)
         return self.model.forward(*args, **kwargs)
 
     @abstractmethod' | git apply
cd ~
```

## :satellite: Remote setup
Supported platforms for training:
- Google Colab
- Kaggle

#### :key: Secrets
Set secrets in Colab or Kaggle: 
- `gitrepo` : `balthazarneveu/molecule-retrieval-using-nlp.git`
- `uname_git`: github user name
- `github_token`: github token to this repo
- `wandb_api_key`: weights and biases API key
- `kaggle_uname`: https://www.kaggle.com/settings/account > Create new token
- `kaggle_token`: same as above


#### :rocket: Kaggle

:warning: Phone needs to be confirmed to acess GPU and have internet acess on Kaggle


Create a [__kaggle_logins.py](/__kaggle_login.py) file.
```python
kaggle_users = {
    "user1": {
        "username": "user1_kaggle_name",
        "key": "user1_kaggle_key"
    },
    "user2": {
        "username": "user2_kaggle_name",
        "key": "user2_kaggle_key"
    },
}
```

Run `python remote_training.py -u user1 -e X`
This will create a dedicated folder for training a specific experiment with a dedicated notebook.
- use **`-p`** (`--push`) will upload/push the notebook and run it.
- use **`-d`** (`--download`) to download the training results and save it to disk)
Please note that the first time, you'll need to manually edit the notebook under kaggle web page to allow secrets.


#### :scroll: Notebook
- Setup a drive folder named `molecules-nlp` where you can store results checkpoints
- Use [training_notebook.ipynb](/training_notebook.ipynb) to launch the right training, ready for Colab & Kaggle.


# :toolbox: Experimenting
## :jigsaw: Training 
Add a new experiment `X` to [experiments](/experiments.py) defined by a given number.
Commit your file, we need to track results.
```shell
python train.py -e X
```



## :triangular_ruler: Metrics, evaluation
#### Check metrics


```
python3 metrics.py -e 2 3 4 -t -p
```
- `-p` to plot training curves
- `-t` to display a result table :bulb: `-nok` will disable Kaggle score retrieval

#### :chart_with_upwards_trend: Weights and Biases
[wandb.ai/molecule-nlp-altegrad-23](https://wandb.ai/molecule-nlp-altegrad-23)

#### :chart_with_downwards_trend: Tensorboard
```bash
tensorboard --logdir __tensorboard_logs
```
#### :rocket: Evaluation and submission
Evaluation of various models
```shell
python evaluation.py -e X Y Z
```
This will generate a submission.csv that can be submitted to the competition.
Command line will be provided to push results to Kaggle directly.


```shell
kaggle competitions submit -c altegrad-2023-data-challenge -f __output/X/submission.csv -m "exp_X commit"
```


## Related papers review
#### Text2Mol: 
- Text encoder = SciBERT
- Linear projection (adapter) + **layer normalization** (why?)
- Graph encoder GCN d=3, input tokens = Morgan Fingerprint -> Mol2Vec radius 0 & 1.
- Temperature ($exp (\tau)$)
- Add some negatives samples. Unclear.
- Transformer text decoder to focus on various parts of the graph.



### Saving some time
:fire: LORA SciBERT + :fire: FatGCN - `batch_size=8`
```
Device 1 [NVIDIA T500]                         PCIe GEN 3@ 4x RX: 20.51 MiB/s TX: 7.812 MiB/s
 GPU 1635MHz MEM 5000MHz TEMP  72°C FAN N/A% POW N/A W
 GPU[||||||||||||||||||||||||||||||||||||| 90%] MEM[||||||||||||||||||||||||||2.951Gi/4.000Gi]
```

:snowflake: LORA SciBERT + :fire: FatGCN  `batch_size=8`
```
Device 1 [NVIDIA T500]                         PCIe GEN 3@ 4x RX: 24.41 MiB/s TX: 7.812 MiB/s
 GPU 1455MHz MEM 5000MHz TEMP  75°C FAN N/A% POW N/A W
 GPU[||||||||||||||||||||||||||||||||||||| 91%] MEM[|||||||||                 0.913Gi/4.000Gi]
```

:snowflake: LORA SciBERT + :fire: FatGCN  `batch_size=32`
```
Device 1 [NVIDIA T500]                         PCIe GEN 3@ 4x RX: 19.53 MiB/s TX: 6.836 MiB/s
 GPU 1710MHz MEM 5000MHz TEMP  66°C FAN N/A% POW N/A W
 GPU[|||||||||||||||||||||||||||||||||||||100%] MEM[|||||||||||||             1.290Gi/4.000Gi]
```


:snowflake: LORA SciBERT + :fire: FatGCN  `batch_size=128`
```
Device 1 [NVIDIA T500]                         PCIe GEN 3@ 4x RX: 24.41 MiB/s TX: 8.789 MiB/s
 GPU 1725MHz MEM 5000MHz TEMP  79°C FAN N/A% POW N/A W
 GPU[|||||||||||||||||||||||||||||||||||||100%] MEM[||||||||||||||||||||||||||2.974Gi/4.000Gi]
```