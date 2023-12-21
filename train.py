from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, BETAS, OPTIMIZER,
    TRAIN, VALIDATION, TEST, ID, MAX_STEP_PER_EPOCH
)
import logging
from data_dumps import Dump
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from model import Model
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
from pathlib import Path
from typing import Tuple
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / '__data'


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_experience(exp: int) -> Tuple[torch.nn.Module, dict]:
    configuration = {
        NB_EPOCHS: 5,
        # batch_size : 32
        BATCH_SIZE: (16, 8, 8),  # To fit Nvidia T500 4Gb RAM
        OPTIMIZER: {
            LEARNING_RATE: 2e-5,
            WEIGHT_DECAY: 0.01,
            BETAS: (0.9, 0.999)
        },
        TOKENIZER_NAME: 'distilbert-base-uncased',
        MAX_STEP_PER_EPOCH: None
    }
    if exp == 0:
        configuration[NAME] = 'check-pipeline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Check pipeline'
        configuration[MAX_STEP_PER_EPOCH] = 100
        model = Model(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                      nhid=8, graph_hidden_channels=8)
    if exp == 1:
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers'
        model = Model(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                      nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    configuration[ID] = exp
    configuration[BATCH_SIZE] = {
        TRAIN: configuration[BATCH_SIZE][0],
        VALIDATION: configuration[BATCH_SIZE][1],
        TEST: configuration[BATCH_SIZE][2]
    }
    return model, configuration


def contrastive_loss(v1, v2):
    CE = torch.nn.CrossEntropyLoss()
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


def get_tokenizer(configuration):
    tokenizer_model_name = configuration[TOKENIZER_NAME]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    return tokenizer


def get_output_directory(configuration: dict, root_dir: Path = ROOT_DIR):
    output_directory = root_dir/'__output'/f"{configuration[ID]:04d}_{configuration[NAME]}"
    return output_directory


def prepare_experience(exp: int, root_dir=ROOT_DIR) -> dict:
    model, configuration = get_experience(exp)
    output_directory = get_output_directory(configuration, root_dir=root_dir)
    tokenizer = get_tokenizer(configuration)
    device = get_device()
    return model, configuration, output_directory, tokenizer, device


def train_experience(exp: int, root_dir=ROOT_DIR) -> None:
    model, configuration, output_directory, tokenizer, device = prepare_experience(exp, root_dir=root_dir)
    if output_directory.exists():
        logging.warning(f"Experience {exp} already trained")
        return
    output_directory.mkdir(exist_ok=True, parents=True)
    training(model, output_directory, configuration, tokenizer, device)


def training(model, output_directory, configuration, tokenizer, device):
    gt = np.load(DATA_DIR/"token_embedding_dict.npy", allow_pickle=True)[()]

    nb_epochs = configuration[NB_EPOCHS]
    batch_size = configuration[BATCH_SIZE]
    val_dataset = GraphTextDataset(root=DATA_DIR, gt=gt, split=VALIDATION[:3], tokenizer=tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size[VALIDATION], shuffle=True)
    train_dataset = GraphTextDataset(root=DATA_DIR, gt=gt, split=TRAIN, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size[TRAIN], shuffle=True)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), **configuration[OPTIMIZER])

    epoch = 0
    loss = 0
    losses = []
    count_iter = 0
    time1 = time.time()
    printEvery = 50
    printEvery = 1
    best_validation_loss = 1000000
    max_count = configuration[MAX_STEP_PER_EPOCH]

    for epoch in range(nb_epochs):
        print('-----EPOCH{}-----'.format(epoch+1))
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            if max_count is not None and batch_idx > max_count:
                break
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch

            x_graph, x_text = model(graph_batch.to(device),
                                    input_ids.to(device),
                                    attention_mask.to(device))
            current_loss = contrastive_loss(x_graph, x_text)
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            loss += current_loss.item()

            count_iter += 1
            if count_iter % printEvery == 0:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                                       time2 - time1, loss/printEvery))
                losses.append(loss)
                loss = 0
        model.eval()
        val_loss = 0
        for batch in val_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            x_graph, x_text = model(graph_batch.to(device),
                                    input_ids.to(device),
                                    attention_mask.to(device))
            current_loss = contrastive_loss(x_graph, x_text)
            val_loss += current_loss.item()
        best_validation_loss = min(best_validation_loss, val_loss)
        print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ',
              str(val_loss/len(val_loader)))
        metrics_dict = {
            'epoch': epoch,
            'validation_accuracy': val_loss,
            'optimizer_state_dict': optimizer.state_dict(),
            'configuration': configuration,
            'loss': loss,
        }
        Dump.save_json(metrics_dict, output_directory/'metrics__{epoch:04d}.json')

        if best_validation_loss == val_loss:
            print('validation loss improved saving checkpoint...')

            save_path = os.path.join(output_directory, f'model_{epoch:04d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_accuracy': val_loss,
                'configuration': configuration,
                'loss': loss,
            }, save_path)
            print('checkpoint saved to: {}'.format(save_path))


def evaluate_experience(exp: int, root_dir=ROOT_DIR) -> None:
    # model, configuration = get_experience(exp)
    # output_directory = get_output_directory(configuration, root_dir=root_dir)
    # tokenizer = get_tokenizer(configuration)
    model, configuration, output_directory, tokenizer, device = prepare_experience(exp, root_dir=root_dir)
    evaluation(model, output_directory, configuration, tokenizer, device)


def evaluation(model, model_path, configuration, tokenizer, device):
    batch_size = configuration[BATCH_SIZE][TEST]
    print('loading best model...')
    best_model_path = sorted(list(model_path.glob("*.pt")))
    assert len(best_model_path) > 0, "No model checkpoint found at {model_path}"
    best_model_path = best_model_path[-1]
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()
    gt = np.load(DATA_DIR/"token_embedding_dict.npy", allow_pickle=True)[()]
    test_cids_dataset = GraphDataset(root=DATA_DIR, gt=gt, split='test_cids')
    test_text_dataset = TextDataset(
        file_path=DATA_DIR/'test_text.txt', tokenizer=tokenizer)

    idx_to_cid = test_cids_dataset.get_idx_to_cid()

    test_loader = DataLoader(
        test_cids_dataset, batch_size=batch_size, shuffle=False)

    graph_embeddings = []
    for batch in test_loader:
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())

    test_text_loader = TorchDataLoader(
        test_text_dataset, batch_size=batch_size, shuffle=False)
    text_embeddings = []
    for batch in test_text_loader:
        for output in text_model(batch['input_ids'].to(device),
                                 attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())

    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col != 'ID']]
    solution.to_csv(model_path/'submission.csv', index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train classification models on Abide dataset - compare performances")
    parser.add_argument("-d", "--device", type=str,
                        choices=["cpu", "cuda"], default=str(get_device()), help="Training device")
    parser.add_argument("-e", "--exp-list", nargs="+", type=int, default=[1], help="List of experiments to run")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    for exp in args.exp_list:
        train_experience(exp)
        evaluate_experience(exp)
