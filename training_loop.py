from properties import (
    NB_EPOCHS, BATCH_SIZE, OPTIMIZER,
    TRAIN, VALIDATION, DATA_DIR, MAX_STEP_PER_EPOCH
)
from data_dumps import Dump
from torch_geometric.data import DataLoader
from dataloader import GraphTextDataset
import numpy as np
import torch
from torch import optim
import time
import os
from pathlib import Path
from transformers import PreTrainedTokenizer
from loss import contrastive_loss
from tqdm import tqdm
import logging


def train(model, optimizer, count_iter, epoch, train_loader, max_count=None, print_freq=50, device='cuda'):
    logging.info('-----EPOCH{}-----'.format(epoch+1))
    model.train()
    losses = []
    time1 = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train"):
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
        loss = current_loss.item()

        count_iter += 1
        if count_iter % print_freq == 0:
            time2 = time.time()
            logging.info("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(
                count_iter,
                time2 - time1, loss/print_freq))
        losses.append(loss)
    return model, losses


def eval(model, val_loader, device='cuda'):
    model.eval()
    val_loss = 0
    for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
        # if batch_idx < 300:
        #     continue
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

    return val_loss/len(val_loader)


def training(
    model: torch.nn.Module,
    output_directory: Path, configuration: dict, tokenizer: PreTrainedTokenizer,
    device: str,
    print_freq: int = 50,
    backup_folder: Path = None
):
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
    all_losses = []
    count_iter = 0

    best_validation_loss = 1000000
    max_count = configuration[MAX_STEP_PER_EPOCH]

    for epoch in range(nb_epochs):
        torch.cuda.empty_cache()
        model, epoch_losses = train(model, optimizer, count_iter, epoch, train_loader,
                                    max_count=max_count, print_freq=print_freq, device=device)
        all_losses.extend(epoch_losses)
        val_loss = eval(model, val_loader, device=device)
        best_validation_loss = min(best_validation_loss, val_loss)
        print(f'-----EPOCH {epoch+1} ----- done.   ' +
              f'Validation loss:  {val_loss:.3e} - BEST : {best_validation_loss:.3e}')
        metrics_dict = {
            'epoch': epoch,
            'validation_loss': val_loss,
            'configuration': configuration,
            'training_loss': epoch_losses,
        }

        metric_file_name = f'metrics__{epoch:04d}.json'
        metric_files_list = [output_directory/metric_file_name]
        if backup_folder is not None:
            metric_files_list.append(backup_folder/metric_file_name)
        for metric_file_path in metric_files_list:
            Dump.save_json(metrics_dict, metric_file_path)
        if best_validation_loss == val_loss:
            print('validation loss improved saving checkpoint...')
            model_file_name = f'model_{epoch:04d}.pt'
            save_path = os.path.join(output_directory, model_file_name)
            save_path_list = [save_path]
            if backup_folder is not None:
                save_path_list.append(os.path.join(backup_folder, model_file_name))
            for save_path in save_path_list:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'validation_accuracy': val_loss,
                    'configuration': configuration,
                    'loss': loss,
                }, save_path)
            print('checkpoint saved to: {}'.format(save_path))
