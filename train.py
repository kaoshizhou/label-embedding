import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['MASTER_ADDR'] = 'localhost'

import logging

import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import HfArgumentParser

from src.config import Config
from src.dataset import IPC_DATA
from src.logger import Logger
from src.trainer import Trainer
from src.utils import *

from tqdm import tqdm

if __name__ == '__main__':
    logger = Logger()
    parser = HfArgumentParser(Config)
    config, = parser.parse_args_into_dataclasses()

    use_gpu = config.use_gpu and torch.cuda.is_available()
    if not use_gpu and config.distribute:
        raise RuntimeError('no gpu can use when distributed training')

    # wandb init
    if config.local_rank == -1 or config.local_rank == 0:
        wandb.init()

    logging.info(f'start training!!!! local_rank={config.local_rank}')
    set_seed(config.seed)

    # distributed training setting
    if config.distribute and config.local_rank != -1:
        assert torch.cuda.device_count() > config.local_rank
        torch.cuda.set_device(config.local_rank)
        device = torch.device('cuda', index=config.local_rank)
        world_size = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        dist.init_process_group(backend=config.backend, init_method='env://',
                                world_size=world_size, rank=config.local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1
    
    train_set = IPC_DATA(config, split='train')
    valid_set = IPC_DATA(config, split='valid')

    # prepare data sampler
    if config.distribute and config.local_rank != -1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=config.local_rank, seed=config.seed)
    else:
        train_sampler = None

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=(train_sampler is None),
                              num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn, sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn)
    if config.local_rank == -1 or config.local_rank == 0:
        logging.info(f'datasets loaded, train: {len(train_set)}, valid: {len(valid_set)}')


    trainer = Trainer(config, device, world_size)
    trainer.train(train_loader, valid_loader, train_sampler)