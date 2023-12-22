# molecule-retrieval-using-nlp
MVA23 ALTEGRAD data challenge on Molecule retrieval using natural language analyzis.


## Authors
- Program: [MVA Master's degree](https://www.master-mva.com/) class on language and graph learning [ALTEGRAD](https://www.master-mva.com/cours/cat-advanced-learning-for-text-and-graph-data-altegrad/). ENS Paris-Saclay.
- Authors
    - [Balthazar Neveu](https://github.com/balthazarneveu)
- [Kaggle](https://www.kaggle.com/competitions/altegrad-2023-data-challenge)


## Training
Add a new experiment `X` to [experiments](/experiments.py) defined by a given number.
Commit your file, we need to track results.
```shell
python train.py -e X
```

## Metrics, evaluation
#### Check metrics


```
python3 metrics.py -e 2 3 4 -t -p
```
- `-p` to plot training curves
- `-t` to display a result table :bulb: `-nok` will disable Kaggle score retrieval


#### Evaluation and submission
Evaluation of various models
```shell
python evaluation.py -e X Y Z
```
This will generate a submission.csv that can be submitted to the competition.
Command line will be provided to push results to Kaggle directly.


```shell
kaggle competitions submit -c altegrad-2023-data-challenge -f __output/X/submission.csv -m "exp_X commit"
```