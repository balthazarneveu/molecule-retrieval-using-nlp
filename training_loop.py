from properties import (
    NB_EPOCHS, BATCH_SIZE, OPTIMIZER,
    TRAIN, VALIDATION, DATA_DIR, MAX_STEP_PER_EPOCH, ROOT_DIR
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
from torch.utils.tensorboard import SummaryWriter
from loss import contrastive_loss
from tqdm import tqdm
import logging
from typing import Optional
from validation import eval
import wandb
import shutil

def train(
        model, optimizer, count_iter, epoch, train_loader,
        max_count: Optional[int] = None,
        print_freq: Optional[int] = 50, device: Optional[str] = 'cuda',
        writer: Optional[SummaryWriter] = None
):
    logging.info('-----EPOCH{}-----'.format(epoch+1))
    model.train()
    losses = []
    time1 = time.time()
    avg_loss = 0.
    total_batches = len(train_loader)
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
        avg_loss += current_loss.item()
        count_iter += 1
        if count_iter % print_freq == 0:
            avg_loss = avg_loss/print_freq
            global_step = epoch * total_batches + batch_idx
            writer.add_scalar('Loss', current_loss.item(), global_step)
            time2 = time.time()
            logging.info("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(
                count_iter,
                time2 - time1, avg_loss))
            avg_loss = 0.
        losses.append(loss)
    return model, losses


def training(
    model: torch.nn.Module,
    output_directory: Path, configuration: dict, tokenizer: PreTrainedTokenizer,
    device: str,
    print_freq: int = 50,
    backup_folder: Path = None,
    tensorboard_root: Path = ROOT_DIR / '__tensorboard_logs',
    wandb_flag: bool = True
):
    tensorboard_root.mkdir(exist_ok=True, parents=True)
    tensorboard_dir = tensorboard_root / output_directory.name
    if backup_folder is not None:
        backup_tensorboard_dir = backup_folder.parent / tensorboard_root.name / backup_folder.name
        backup_tensorboard_dir.mkdir(exist_ok=True, parents=True)
    writer_tra = SummaryWriter(log_dir=tensorboard_dir/"train")
    writer_val = SummaryWriter(log_dir=tensorboard_dir/"val")
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
    last_checkpoint = []
    for epoch in range(nb_epochs):
        torch.cuda.empty_cache()
        model, epoch_losses = train(model, optimizer, count_iter, epoch, train_loader,
                                    max_count=max_count, print_freq=print_freq, device=device,
                                    writer=writer_tra)
        all_losses.extend(epoch_losses)
        val_loss, lrap_score = eval(model, val_loader, device=device, max_count=max_count, score=True)
        best_validation_loss = min(best_validation_loss, val_loss)
        print(f'-----EPOCH {epoch+1} ----- done.   ' +
              f'Validation loss:  {val_loss:.3e} - BEST : {best_validation_loss:.3e} | lrap_score: {lrap_score:.3e}')
        metrics_dict = {
            'epoch': epoch,
            'validation_loss': val_loss,
            'configuration': configuration,
            'training_loss': epoch_losses,
            'lrap_score': lrap_score,
        }
        writer_val.add_scalar('Loss', val_loss, (epoch+1) * len(train_loader) + len(train_loader))
        writer_val.add_scalar('Score', lrap_score, (epoch+1) * len(train_loader) + len(train_loader))
        if wandb_flag:
            wandb.log({"Validation Loss": val_loss, "Score": lrap_score})
        metric_file_name = f'metrics__{epoch:04d}.json'
        metric_files_list = [output_directory/metric_file_name]
        if backup_folder is not None:
            metric_files_list.append(backup_folder/metric_file_name)
        for metric_file_path in metric_files_list:
            Dump.save_json(metrics_dict, metric_file_path)
        if best_validation_loss == val_loss:
            print('validation loss improved saving checkpoint...')
            model_file_name = f'model_{epoch:04d}.pt'
            save_path = output_directory/model_file_name
            save_path_list = [save_path]
            if backup_folder is not None:
                save_path_list.append(os.path.join(backup_folder, model_file_name))
                shutil.copytree(tensorboard_dir, backup_tensorboard_dir, dirs_exist_ok=True)
            for last_check in last_checkpoint:
                if Path(last_check).exists():
                    logging.warning(f"Removing last checkpoint {last_check}")
                    os.remove(last_check)
            last_checkpoint = []
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
                last_checkpoint.append(save_path)
    writer_tra.close()
    writer_val.close()
