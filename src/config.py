from dataclasses import dataclass, field

from typing import Optional



@dataclass
class Config:
    # -------- data config ---------
    label_name_file: str = field(
        default='data/label/ipc_label.txt',
        metadata={'help': 'ipc label, one line one label name'}
    )
    label_definition_file: str = field(
        default='data/label/ipc_definition.txt',
        metadata={'help': 'ipc label definition, `label_name \\t label_definition` in each line'}
    )
    train_file: str = field(
        default='data/data/train.json',
        metadata={'help': 'train file'}
    )
    valid_file: str = field(
        default='data/data/dev.json',
        metadata={'help': 'valid file'}
    )
    test_file: str = field(
        default='data/data/test.json',
        metadata={'help': 'test file'}
    )
    batch_size: int = field(
        default=2,
        metadata={'help': 'batch size, number of positive labels in a batch'}
    )
    num_negative_labels: int = field(
        default=7,
        metadata={'help': 'number of negative labels for each positive label'}
    )
    num_near_labels: int = field(
        default=4,
        metadata={'help': 'number of near labels in same department or same category'
                  'less than `num_negative_labels`'}
    )
    near_strategy: str = field(
        default='category',
        metadata={'help': 'choose near labels in same category or in same department'
                  'optionals are `category` and `department`'}
    )
    num_workers: int = field(
        default=8,
        metadata={'help': 'num_workers'}
    )
    shuffle: bool = field(
        default=True,
        metadata={'help': 'shuffle training data or not'}
    )

    # ------------model config-------------
    model_name_or_path: Optional[str] = field(
        default='bert-base-chinese',
        metadata={'help': 'path of model or model name in huggingface to load a pretrained model'}
    )
    config_name: Optional[str] = field(
        default='bert-base-chinese',
        metadata={'help': 'config name of model to train a model from scratch'
                          'only if `model_name_or_path` is `None`, `config_name` is activated'}
    )
    tokenizer_name: str = field(
        default='bert-base-chinese',
        metadata={'help': 'tokenizer path or name'}
    )
    checkpoint_dir: str = field(
        default='./checkpoint',
        metadata={'help': 'save directory of checkpoint'}
    )
    resume_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'resume directory'}
    )
    embedding: str = field(
        default='embedding.pkl',
        metadata={'help': 'label embedding file'}
    )
    params: str = field(
        default='params.pkl',
        metadata={'help': 'training params, including `step`, `epoch` and `optimizer`'}
    )
    

    # ------------training config-----------
    epochs: int = field(
        default=10,
        metadata={'help': 'training epoch'}
    )
    lr: float = field(
        default=5e-5,
        metadata={'help': 'learning rate in optimizer'}
    )
    weight_decay: float = field(
        default=0,
        metadata={'help': 'weight decay in optimizer'}
    )
    seed: int = field(
        default=42,
        metadata={'help': 'random seed'}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={'help': 'warmup ratio to get the max learning rate'}
    )
    max_grad_norm: Optional[float] = field(
        default=None,
        metadata={'help': 'max grad norm after loss backward, deactivated when set `None`'}
    )
    dropout: float = field(
        default=0.1,
        metadata={'help': 'dropout ratio'}
    )
    distribute: bool = field(
        default=True,
        metadata={'help': 'distributed training or not'}
    )
    local_rank: int = field(
        default=-1,
        metadata={'help': 'local rank'}
    )
    backend: str = field(
        default='nccl',
        metadata={'help': 'backend of distributed training'}
    )
    use_gpu: bool = field(
        default=True,
        metadata={'help': 'use gpu or not'}
    )
    early_stop: bool = field(
        default=True,
        metadata={'help': 'early stop according to validation set performence or not'}
    )
    unimproved_epochs: int = field(
        default=3,
        metadata={'help': 'unimproved duration before early stop'}
    )
    temperature: float = field(
        default=2,
        metadata={'help': 'temperature'}
    )
    delta: float = field(
        default=0.5,
        metadata={'help': 'delta'}
    )
    alpha: float = field(
        default=0.8,
        metadata={'help': 'alpha'}
    )
