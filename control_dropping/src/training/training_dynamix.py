## Initialization:: Use env stbl3 or raylib
import pickle
import os
CONTROL_DROP_DIR = os.environ["CONTROL_DROP_DIR"]

import argparse
import sys

from math import inf, radians, degrees
from stable_baselines3 import PPO, A2C
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box
from typing import Dict, List, Optional, Tuple, Any

import logging

# Torch
import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import Adam
from torchmetrics import Accuracy

## Dataloader for Model Training
from stable_baselines3.common.utils import obs_as_tensor
from torch.utils.data import DataLoader, random_split

### Training Modules
from training.distributed import setup, cleanup, run_distributed
from training.data import get_joint_dataset

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
from tqdm import tqdm


## Model for predictions
from control_dropping_rpal.RL.Networks.ExtractorNetworks import (
    DYNAMIX_OUTPUT_SIZES_DICT,
    TemporalObjectTactileEncoder_Additive,
    DynamixModel,
    CritiqModel,
)

from control_dropping_rpal.RL.control_dropping_env import BerrettHandGym, T_buffer

BATCH_SIZE = 128
DATA_SAVE_PATH = os.path.join(CONTROL_DROP_DIR, "Data_Collection")
MODEL_PATH = os.path.join(
    CONTROL_DROP_DIR,
    "control_dropping/src/RL/Training/Checkpoints/TransformerFeatureEncoder/Expert_rl_5000_steps.zip",
)
PATH_DYNAMIX = os.path.join(
    CONTROL_DROP_DIR, "Data_Collection/Time_Dependent_Samples_4/"
)
PATH_CRITIQ = os.path.join(
    CONTROL_DROP_DIR, "Data_Collection/Action_Pred_Time_Dependent_Samples_4/"
)

GAMMA = 0.5

state_space = {
    "palm_tactile": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            24,
        ),
    ),  # Value
    "finger_1_tactile": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            24,
        ),
    ),  # Value
    "finger_2_tactile": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            24,
        ),
    ),  # Value
    "finger_3_tactile": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            24,
        ),
    ),  # Value
    # 'tactile_pos': Box(low= -inf, high= inf, shape=(378, )), # Position
    "finger_1_location": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            2 + 34 * 3,
        ),
    ),  # Joint pos [Theta_1, Theta_2] + [xyz*34]
    "finger_2_location": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            2 + 34 * 3,
        ),
    ),
    "finger_3_location": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            2 + 34 * 3,
        ),
    ),
    "palm_location": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            2 + 24 * 3,
        ),
    ),
    "obj_location": Box(low=-inf, high=inf, shape=(T_buffer, 7, 6)),  # Position
    "obj_velocity": Box(
        low=-inf, high=inf, shape=(T_buffer, 7, 6)
    ),  # Value, Concat with angular velocity
    "state_attrib": Box(
        low=-inf, high=inf, shape=(45,)
    ),  # Ball Cnt, Progress, Prev.Actions, hand_cfg, hand_trq (44)
}

# Params
NUM_LAYERS_TRANSFORMER = 12
NUM_RESIDUALS = 3
EPOCHS = 1000
VEC_ENCODING_SIZE = 512

MODEL_ARGS = {
    "vec_encoding_size": VEC_ENCODING_SIZE,
    "num_residuals": NUM_RESIDUALS,
    "num_tsf_layer": NUM_LAYERS_TRANSFORMER,
    "use_mask": True,
    "dropout_prob": 0.01,
    "embed_dim_low": VEC_ENCODING_SIZE,
    "T_buffer": T_buffer,
    "state_space": state_space,
}

# TODO:
# Try Including/Excluding Finger Values (2)
# Diff between loss (Mean/Sum) ()
# Use huggingface model
# Model Predictive Control: Accuracy (1)
#
# key_losses_dynamix = {
#     "palm_tactile": lambda y_pred, y_target: 1
#     - F.cosine_similarity(y_pred, y_target).mean(),
#     "finger_1_tactile": lambda y_pred, y_target: 1
#     - F.cosine_similarity(y_pred, y_target).mean(),
#     "finger_2_tactile": lambda y_pred, y_target: 1
#     - F.cosine_similarity(y_pred, y_target).mean(),
#     "finger_3_tactile": lambda y_pred, y_target: 1
#     - F.cosine_similarity(y_pred, y_target).mean(),
#     "palm_location": lambda y_pred, y_target: torch.zeros_like(
#         F.mse_loss(y_pred, y_target)
#     ),  # F.mse_loss(y_pred, y_target),
#     "finger_1_location": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
#     "finger_2_location": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
#     "finger_3_location": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
#     "obj_location": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
#     "obj_count": lambda y_pred, y_target: F.cross_entropy(y_pred, y_target),
#     "reward": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
# }

# key_losses_critiq = {
#     "action": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
#     "reward": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
# }

key_losses_dynamix = {
    "palm_tactile": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
    "finger_1_tactile": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
    "finger_2_tactile": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
    "finger_3_tactile": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
    "palm_location": lambda y_pred, y_target: torch.zeros_like(
        F.mse_loss(y_pred, y_target)
    ),  # F.mse_loss(y_pred, y_target),
    "finger_1_location": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
    "finger_2_location": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
    "finger_3_location": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
    "obj_location": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
    "obj_count": lambda y_pred, y_target: F.cross_entropy(y_pred, y_target),
    "reward": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
}

key_losses_critiq = {
    "action": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
    "reward": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
}


def get_args():
    parser = argparse.ArgumentParser(description='Joint Model Training')
    parser.add_argument('--name', type=str, required=True, help='Name for the experiment')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes for distributed training')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'linear', 'constant'], help='Learning rate scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine annealing')
    parser.add_argument('--clip_grad_norm', type=float, default=None, help='grad norm clip (try 2.0)')
    args = parser.parse_args()
    setattr(args, "distributed", min(args.world_size, torch.cuda.device_count()) > 1)
    return args

def get_optimizer_and_scheduler(models, args):
    params = list(models[0].parameters()) + list(models[1].parameters())
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

    if args.lr_scheduler == 'constant':
        return optimizer, None

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)

    if args.lr_scheduler == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
    elif args.lr_scheduler == 'linear':
        main_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs - args.warmup_epochs)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])

    return optimizer, scheduler

def make_dynamix_and_predictor(
    model_args: Dict[str, Any]
) -> Tuple[DynamixModel, CritiqModel]:
    # Extract common arguments
    embed_dim_high = model_args.get("embed_dim_high", 1024)
    embed_dim_low = model_args.get("embed_dim_low", 256)
    device = model_args.get("device", "cuda")
    dropout_prob = model_args.get("dropout_prob", 0.05)
    num_tsf_layer = model_args.get("num_tsf_layer", 4)
    num_residual_blocks = model_args.get("num_residual_blocks", 4)
    vec_encoding_size = model_args.get("vec_encoding_size", 8)
    use_mask = model_args.get("use_mask", False)

    # Create the shared encoder
    encoder = TemporalObjectTactileEncoder_Additive(
        observation_space=model_args.get("state_space"),
        vec_encoding_size=vec_encoding_size,
        t_dim_size=model_args.get("T_buffer"),
        load_pretrain=False,
        num_tsf_layer=num_tsf_layer,
        use_mask=use_mask,
    )

    # Create DynamixModel
    dynamix_model = DynamixModel(
        embed_dim_high=embed_dim_high,
        embed_dim_low=embed_dim_low,
        device=device,
        dropout_prob=dropout_prob,
        num_tsf_layer=num_tsf_layer,
        num_residual_blocks=num_residual_blocks,
        vec_encoding_size=vec_encoding_size,
        use_mask=use_mask,
        encoder=encoder,
    )

    # Create CritiqModel
    critiq_model = CritiqModel(
        embed_dim_high=embed_dim_high,
        embed_dim_low=embed_dim_low,
        device=device,
        dropout_prob=dropout_prob,
        num_tsf_layer=num_tsf_layer,
        num_residual_blocks=num_residual_blocks,
        vec_encoding_size=vec_encoding_size,
        use_mask=use_mask,
        encoder=encoder,
    )

    return dynamix_model, critiq_model    

class DynamixCritiqLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            dynamix_preds,
            dynamix_target,
            critiq_preds,
            critiq_target
    ):
        dynamix_loss = th.stack(
            [
                key_losses_dynamix[key](dynamix_preds[key], dynamix_target[key])
                for key in dynamix_target.keys() 
                if key in dynamix_preds and key in dynamix_target
            ]
        ).sum()

        critiq_loss = th.stack(
            [
                key_losses_critiq[key](critiq_preds[key], critiq_target[key])
                for key in critiq_target.keys() 
                if key in critiq_preds and key in critiq_target
            ]
        ).sum()

        return {
            "dynamix_loss": dynamix_loss, "critiq_loss": critiq_loss
        }
    

class DynamixCritiqValidation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            dynamix_preds,
            dynamix_target,
            critiq_preds,
            critiq_target
    ):
        dynamix_loss = th.stack(
            [
                key_losses_critiq[key](dynamix_preds[key], dynamix_target[key])
                for key in critiq_target.keys() 
                if key in dynamix_preds and key in dynamix_target
            ]
        ).sum()

        critiq_loss = th.stack(
            [
                key_losses_critiq[key](critiq_preds[key], critiq_target[key])
                for key in critiq_target.keys() 
                if key in critiq_preds and key in critiq_target
            ]
        ).sum()

        correct = th.argmax(dynamix_target["obj_count"], dim=1) == th.argmax(
            dynamix_preds["obj_count"], dim=1
        )
        obj_count_accuracy = correct.int().sum() / correct.size(0)
        print("[DBG] obj cnt acc:", obj_count_accuracy, correct.shape)

        return {
            "dynamix_loss": dynamix_loss, "critiq_loss": critiq_loss, "obj_count_accuracy": obj_count_accuracy
        }

def train_one_epoch(models, train_loader, optimizer, criterion, device, epoch, writer):
    dynamix_model, critiq_model = models
    dynamix_model.train()
    critiq_model.train()
    
    total_loss = 0
    loss_info = {}

    for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
        dynamix_data, dynamix_target = batch['dynamix']
        critiq_state, critiq_n_state, critiq_target = batch['critiq']

        dynamix_data = {k: v.to(device) for k, v in dynamix_data.items()}
        dynamix_target = {k: v.to(device) for k, v in dynamix_target.items()}
        critiq_state = {k: v.to(device) for k, v in critiq_state.items()}
        critiq_n_state = {k: v.to(device) for k, v in critiq_n_state.items()}
        critiq_target = {k: v.to(device) for k, v in critiq_target.items()}

        optimizer.zero_grad()

        dynamix_pred = dynamix_model(dynamix_data)
        critiq_pred = critiq_model(critiq_state, critiq_n_state)

        loss = criterion(dynamix_pred, dynamix_target, critiq_pred, critiq_target)
        if isinstance(loss, dict):
            for loss_key, loss_value in loss.items():
                if loss_key not in loss_info:
                    loss_info[loss_key] = 0
                loss_info[loss_key] += loss_value.item()
            loss = sum(loss.values())
        
        loss.backward()

        if args.clip_grad_norm:
            clip_grad_norm_(dynamix_model.parameters(), args.clip_grad_norm)
            clip_grad_norm_(critiq_model.parameters(), args.clip_grad_norm)

        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train/total', avg_loss, epoch)
    
    if loss_info:
        for k, v in loss_info.items():
            avg_loss_component = v / len(train_loader)
            writer.add_scalar(f'Loss/train/{k}', avg_loss_component, epoch)
    
    return avg_loss, loss_info

def validate(models, val_loader, criterion, device, epoch, writer):
    dynamix_model, critiq_model = models
    dynamix_model.eval()
    critiq_model.eval()
    
    total_loss = 0
    loss_info = {}

    with th.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            dynamix_data, dynamix_target = batch['dynamix']
            critiq_state, critiq_n_state, critiq_target = batch['critiq']

            dynamix_data = {k: v.to(device) for k, v in dynamix_data.items()}
            dynamix_target = {k: v.to(device) for k, v in dynamix_target.items()}
            critiq_state = {k: v.to(device) for k, v in critiq_state.items()}
            critiq_n_state = {k: v.to(device) for k, v in critiq_n_state.items()}
            critiq_target = {k: v.to(device) for k, v in critiq_target.items()}

            dynamix_pred = dynamix_model(dynamix_data, is_val=True)
            critiq_pred = critiq_model(critiq_state, critiq_n_state, is_val=True)

            loss = criterion(dynamix_pred, dynamix_target, critiq_pred, critiq_target)
            if isinstance(loss, dict):
                for loss_key, loss_value in loss.items():
                    if loss_key not in loss_info:
                        loss_info[loss_key] = 0
                    loss_info[loss_key] += loss_value.item()
                loss = sum(loss.values())
            
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    writer.add_scalar('Loss/val/total', avg_loss, epoch)
    
    if loss_info:
        for k, v in loss_info.items():
            avg_loss_component = v / len(val_loader)
            writer.add_scalar(f'Loss/val/{k}', avg_loss_component, epoch)
    
    return avg_loss, loss_info

def train(rank, world_size, args):
    if args.distributed:
        setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    dynamix_model, critiq_model = make_dynamix_and_predictor(MODEL_ARGS)

    if args.distributed:
        dynamix_model = DDP(dynamix_model.to(device), device_ids=[rank])
        critiq_model = DDP(critiq_model.to(device), device_ids=[rank])
    else:
        dynamix_model = dynamix_model.to(device)
        critiq_model = critiq_model.to(device)
    
    models = (dynamix_model, critiq_model)
    optimizer, scheduler = get_optimizer_and_scheduler(models, args)
    
    train_dataset, val_dataset = get_joint_dataset(PATH_DYNAMIX, PATH_CRITIQ)
    
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, sampler=val_sampler)
    
    

    train_criterion = DynamixCritiqLoss()
    val_criterion = DynamixCritiqValidation()

    log_dir = os.path.join('./logs', args.name)
    if os.path.exists(os.path.join(log_dir)):
        experiment_cnt = len([i for i in os.listdir("./logs") if i.startswith(args.name)])
        log_dir = os.path.join('./logs', f"{args.name}_{experiment_cnt+1}")
        logging.warn(f"{args.name} already exists, new logging directory: {log_dir}")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard')) if rank == 0 else None

    
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)

    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        train_loss, train_info = train_one_epoch(models, train_loader, optimizer, train_criterion, device, epoch, writer)
        val_loss, val_info = validate(models, val_loader, val_criterion, device, epoch, writer)
        
        train_log = "\n\t".join([f"Train {k}: {v / len(train_loader):.4f}" for k, v in train_info.items()])
        val_log = "\n\t".join([f"Val {k}: {v / len(val_loader):.4f}" for k, v in val_info.items()])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if rank == 0:
                torch.save({
                    'dynamix_model': dynamix_model.state_dict(),
                    'critiq_model': critiq_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(log_dir, 'checkpoints', f'best_model_epoch_{epoch}.pth'))
        
        if rank == 0:
            logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            logging.info(train_log)
            logging.info(val_log)
            logging.info("-"*25)
        
        if scheduler:
            scheduler.step()
            if rank == 0:
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)




    if args.distributed:
        cleanup()


if __name__ == '__main__':
    args = get_args()
    if args.distributed:
        run_distributed(train, args.world_size, args)
    else:
        train(0, 1, args)