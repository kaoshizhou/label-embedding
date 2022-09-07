import logging
import os
from pickletools import optimize

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.models.bert import BertConfig, BertModel, BertTokenizer
from transformers.optimization import AdamW, get_constant_schedule_with_warmup
import wandb

from src.dataset import IPC_LABEL
from src.loss import IPC_LOSS

from .config import Config
from .model import IPC_Classification_Model


class Trainer:
    def __init__(self, config: Config, device=None, world_size=None):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.label = IPC_LABEL(config.label_name_file, config.label_definition_file)
        if config.model_name_or_path:
            self.bert = BertModel.from_pretrained(config.model_name_or_path)
        elif config.config_name:
            bert_config = BertConfig.from_pretrained(config.config_name)
            self.bert = BertModel(bert_config)
        else:
            logging.error("set eithor `config` or `model_name_or_path`")
        self.add_token_and_definition()
        self.bert.resize_token_embeddings(len(self.tokenizer))

        self.model = IPC_Classification_Model(config=config, encoder=self.bert, tokenizer=self.tokenizer)
        self.model.to(self.device)
        

    def add_token_and_definition(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.config.tokenizer_name)
        label_without_definition = self.label.label_without_definition
        self.tokenizer.add_tokens(label_without_definition)
        self.label.add_definition(dict(zip(label_without_definition, label_without_definition)))
        
    def train(self, train_loader: DataLoader, valid_loader: DataLoader, train_sampler: DistributedSampler):
        num_train = len(train_loader.dataset)
        num_valid = len(valid_loader.dataset)
        total_batch_size = self.config.batch_size * self.world_size
        total_steps = (num_train // total_batch_size) * self.config.epochs if num_train % total_batch_size == 0 \
            else (num_train // total_batch_size + 1) * self.config.epochs
        last_epoch = -1
        finished_epoch = 0

        self.optimizer = AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.loss_fn = IPC_LOSS(self.config)
        self.loss_fn.to(self.device)

        # {'model_state_dict': obj, 'optimizer_state_dict': obj, 'epoch': epoch, 'step': step}
        if self.config.resume_dir:
            checkpoint = torch.load(os.path.join(self.config.resume_dir, self.config.params))
            last_epoch = checkpoint['step'] - 1
            finished_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.config.distribute:
            self.model = DDP(self.model, device_ids=[self.config.local_rank], broadcast_buffers=False)
        
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_ratio*total_steps,
                                                           last_epoch=last_epoch)

        if self.config.local_rank == -1 or self.config.local_rank == 0:
            logging.info('-----------start training--------')
        step = 0
        for epoch in range(finished_epoch + 1, finished_epoch + 1 + self.config.epochs):
            if self.config.local_rank == 0 or self.config.local_rank == -1:
                logging.info(f'--------epoch {epoch}---------')
            self.model.train()
            with torch.enable_grad(), tqdm(total=total_steps // self.config.epochs) as progress_bar:
                for _, batch in enumerate(train_loader):
                    step += 1
                    sentence = batch['sentence']
                    label = batch['label']
                    label_text = [self.label[l] for l in label]
                    batch_size = len(sentence) // (self.config.num_negative_labels + 1)

                    sentence_dict = self.tokenizer(sentence, padding='longest', truncation=True, return_tensors='pt').to(self.device)
                    label_dict = self.tokenizer(label_text, padding='longest', truncation=True, return_tensors='pt').to(self.device)
                    sentence_embedding = self.model(sentence_dict)
                    label_embedding = self.model(label_dict)
                    loss = self.loss_fn(sentence_embedding, label_embedding, batch_size)
                    loss_val = loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.config.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()

                    progress_bar.update(1)
                    progress_bar.set_postfix(epoch=epoch, loss=loss_val)
                    if self.config.local_rank == 0 or self.config.local_rank == -1:
                        wandb.log({'training_loss': loss_val}, step)
                        wandb.log({'training_lr': self.optimizer.param_groups[0]['lr']}, step)

            if self.config.local_rank == 0 or self.config.local_rank == -1:
                logging.info(f'--------evaluation at epoch {epoch}----------')            
                self.model.eval()
                os.makedirs(os.path.join(
                    self.config.checkpoint_dir,
                    f'epoch-{epoch}'
                ), exist_ok=True)
                try:
                    self.model.module.save_pretrained(os.path.join(
                        self.config.checkpoint_dir,
                        f'epoch-{epoch}'
                    ))
                    self.tokenizer.save_pretrained(os.path.join(
                        self.config.checkpoint_dir,
                        f'epoch-{epoch}'
                    ))
                except:
                    pass
                try:
                    state_dict = {}
                    state_dict['step'] = step
                    state_dict['epoch'] = epoch
                    state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
                    torch.save(state_dict, os.path.join(
                        self.config.checkpoint_dir,
                        f'epoch-{epoch}',
                        self.config.params
                    ))
                except:
                    pass






