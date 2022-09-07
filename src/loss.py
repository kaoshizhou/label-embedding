import torch
import torch.nn as nn

import torch.nn.functional as F
from src.config import Config


class IPC_LOSS(nn.Module):
    def __init__(self, config: Config):
        super(IPC_LOSS, self).__init__()
        self.config = config
        
    
    def forward(self, sentence_embedding, label_embedding, batch_size):
        sentence_embedding = sentence_embedding.reshape(batch_size, -1, sentence_embedding.shape[-1])
        label_embedding = label_embedding.reshape(batch_size, -1, label_embedding.shape[-1])
        sim_xy = torch.einsum('ijk, ijk -> ij', sentence_embedding, label_embedding)
        sim_xy = sim_xy / self.config.temperature
        softmax = F.softmax(sim_xy, dim=-1)[:, 0]
        lm = softmax.log().mean()

        transpose_label_embedding = torch.transpose(label_embedding, 1, 2)
        label_sim = torch.bmm(label_embedding, transpose_label_embedding)
        _mask = torch.eye(self.config.num_negative_labels + 1, device=label_sim.device)
        _mask = _mask.repeat(batch_size, 1, 1) * -100
        label_sim = label_sim + _mask
        max_sim = torch.max(label_sim, dim=-1)[0]
        max_sim = max_sim.reshape(-1, 1)
        delta_matrix = torch.full_like(max_sim, self.config.delta)
        lr = torch.max(delta_matrix, max_sim).mean()
    
        loss = -1 * lm + self.config.alpha * lr
        return loss
        

