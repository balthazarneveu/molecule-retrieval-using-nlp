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

Evaluation of a model
```shell
python evaluation.py -e X
```
This will generate a submission.csv