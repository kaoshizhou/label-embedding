import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert import BertConfig, BertModel, BertTokenizer

from src.dataset import IPC_LABEL

from .config import Config


class IPC_Classification_Model(nn.Module):
    def __init__(self, config: Config, encoder, tokenizer):
        super(IPC_Classification_Model, self).__init__()
        self.config = config
        self.encoder = encoder
        self.tokenizer = tokenizer

    def forward(self, input):
        embedding = self.encoder(**input).pooler_output
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding


        