from properties import (
    NB_EPOCHS, BATCH_SIZE, OPTIMIZER, SCHEDULER, SCHEDULER_CONFIGURATION,
    TRAIN, VALIDATION, DATA_DIR, MAX_STEP_PER_EPOCH, ROOT_DIR,
    TOKENIZER_NAME, LOSS, LOSS_BINARY_CROSSENTROPY, LOSS_CROSSENTROPY, LOSS_TEMPERED_CROSSENTROPY
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, LambdaLR
from loss import contrastive_loss, tempered_contrastive_loss, binary_classifier_contrastive_loss
from tqdm import tqdm
import logging
from typing import Optional, Tuple
from validation import eval
import wandb
import shutil


def train(
        model, optimizer, count_iter, epoch, train_loader,
        max_count: Optional[int] = None,
        print_freq: Optional[int] = 50,
        device: Optional[str] = 'cuda',
        writer: Optional[SummaryWriter] = None,
        scheduler=None,
        scaler: torch.cuda.amp.GradScaler = None,
        loss_type=LOSS_CROSSENTROPY
) -> Tuple[torch.nn.Module, list]:
    """
    Trains the model for one epoch and returns the trained model and a list of losses for each batch.

    Args:
        model (torch.nn.Module): model to be trained.
        optimizer (torch.optim.Optimizer): optimizer for training.
        count_iter (int): current iteration count.
        epoch (int): current epoch count.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        max_count (int, optional): maximum number of batches to train on. If None, trains on all batches.
        Defaults to None.
        print_freq (int, optional): frequency of printing training status. Defaults to 50.
        device (str, optional): device to train on. Defaults to 'cuda'.
        writer (SummaryWriter, optional): TensorBoard writer. Defaults to None.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): learning rate scheduler.
        Update performed only for CosineAnnealingWarmRestarts
        https://wandb.ai/wandb_fc/tips/reports/How-to-Properly-Use-PyTorch-s-CosineAnnealingWarmRestarts-Scheduler--VmlldzoyMTA3MjM2

    Returns:
        Tuple[torch.nn.Module, list]: The trained model and a list of losses for each batch.
    """
    loss_type_table = loss_type.split(",")
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
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler.is_enabled()):
            x_graph, x_text = model(graph_batch.to(device),
                                    input_ids.to(device),
                                    attention_mask.to(device))
            if x_text.dtype == torch.float16:
                x_graph = x_graph.half()
            current_loss = 0
            if LOSS_TEMPERED_CROSSENTROPY in loss_type_table:
                assert hasattr(model, "temperature"), "Model has no temperature attribute."
                current_loss += tempered_contrastive_loss(x_graph, x_text, model.temperature)
                if count_iter == 0 and batch_idx == 0:
                    print("tempered contrastive loss")
                logging.debug(f"temp: {model.temperature.item():.4f}")
            if LOSS_CROSSENTROPY in loss_type_table:
                if count_iter == 0 and batch_idx == 0:
                    print("cross entropy")
                current_loss += contrastive_loss(x_graph, x_text)
            if LOSS_BINARY_CROSSENTROPY in loss_type_table:
                if count_iter == 0 and batch_idx == 0:
                    print("binary contrastive loss")
                current_loss += binary_classifier_contrastive_loss(x_graph, x_text)
        if torch.isnan(current_loss):
            print("WARNING NaN in loss")
            continue
        scaler.scale(current_loss).backward()  # current_loss.backward()
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None and isinstance(scheduler, CosineAnnealingWarmRestarts) or isinstance(scheduler, LambdaLR):
            scheduler.step(epoch + batch_idx / total_batches)  # update learning rate inside of an epoch

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
    output_directory: Path,
    configuration: dict,
    tokenizer: PreTrainedTokenizer,
    device: str,
    optimizer_state_dict: dict = None,
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
    val_dataset = GraphTextDataset(root=DATA_DIR, gt=gt, split=VALIDATION[:3], tokenizer=tokenizer,
                                   specific_name=configuration[TOKENIZER_NAME])
    val_loader = DataLoader(val_dataset, batch_size=batch_size[VALIDATION], shuffle=True)
    train_dataset = GraphTextDataset(root=DATA_DIR, gt=gt, split=TRAIN, tokenizer=tokenizer,
                                     specific_name=configuration[TOKENIZER_NAME])
    train_loader = DataLoader(train_dataset, batch_size=batch_size[TRAIN], shuffle=True)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), **configuration[OPTIMIZER])
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    epoch = 0
    loss = 0
    all_losses = []
    count_iter = 0
    best_accuracy = 0
    best_validation_loss = 1000000
    max_count = configuration[MAX_STEP_PER_EPOCH]
    last_checkpoint = []
    scheduler = None
    if configuration.get(SCHEDULER, False):
        scheduler_config = configuration[SCHEDULER_CONFIGURATION]
        if configuration[SCHEDULER] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, **scheduler_config)
        elif configuration[SCHEDULER] == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(optimizer, **scheduler_config)
        elif configuration[SCHEDULER] == "LambdaLR":
            scheduler = LambdaLR(optimizer, **scheduler_config)
        else:
            raise NameError(f"Scheduler {configuration[SCHEDULER]} not implemented")
    use_amp = configuration.get("use_amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(nb_epochs):
        if "cuda" in device:
            torch.cuda.empty_cache()
        epoch_losses = [0]
        model, epoch_losses = train(model, optimizer, count_iter, epoch, train_loader,
                                    max_count=max_count, print_freq=print_freq, device=device,
                                    writer=writer_tra, scheduler=scheduler, scaler=scaler,
                                    loss_type=configuration.get(LOSS, LOSS_CROSSENTROPY))
        all_losses.extend(epoch_losses)
        model.eval()
        if "cuda" in device:
            torch.cuda.empty_cache()
        with torch.no_grad():
            val_loss, lrap_score = eval(model, val_loader, device=device, max_count=max_count, score=True)
            if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(lrap_score)
        best_validation_loss = min(best_validation_loss, val_loss)
        best_accuracy = max(best_accuracy, lrap_score)
        print(f'-----EPOCH {epoch+1} ----- done.   ' +
              f'LRAP {lrap_score:.3%} | best {best_accuracy:.3%}')
        print(f"Validation loss:  {val_loss:.3e} - best : {best_validation_loss:.3e}")
        if configuration.get(SCHEDULER, False):
            if configuration[SCHEDULER] == "LambdaLR":
                # print("REMOVE lr_lambda from configuration!!!")
                configuration[SCHEDULER_CONFIGURATION].pop("lr_lambda", None)
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
            wandb.log({"Validation Loss": val_loss, "Score": lrap_score,
                      "Learning Rate": optimizer.param_groups[0]['lr'],
                       "Temperature": model.temperature.item() if hasattr(model, "temperature") else None})
        metric_file_name = f'metrics__{epoch:04d}.json'
        metric_files_list = [output_directory/metric_file_name]
        if backup_folder is not None:
            metric_files_list.append(backup_folder/metric_file_name)
        for metric_file_path in metric_files_list:
            Dump.save_json(metrics_dict, metric_file_path)
        if best_accuracy == lrap_score:
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
