import pandas as pd
from torch.utils.data import Dataset
from .config import Config
import random

class IPC_LABEL:
    def __init__(self, label_name_file, label_definition_file=None):
        assert label_name_file is not None, 'class `IPC_LABEL` missing required argument `label_name_file` '
        self.label_df = pd.read_csv(label_name_file, header=None)
        if label_definition_file is None:
            self.definition_df = pd.DataFrame([], columns=[0, 1])
        else:
            self.definition_df = pd.read_csv(label_definition_file, sep='\t', header=None)

    @property
    def label_name(self):
        return self.label_df.iloc[:, 0].tolist()
    
    @property
    def label_with_definition(self):
        return self.definition_df.iloc[:, 0].tolist()

    @property
    def label_without_definition(self):
        df1 = self.label_df.iloc[:, 0]
        df2 = self.definition_df.iloc[:, 0]
        diff_df = pd.concat([df1, df2, df2]).drop_duplicates(keep=False)
        return diff_df.tolist()
    
    @property
    def num_label(self):
        return len(self.label_name)

    @property
    def num_label_with_definition(self):
        return len(self.label_with_definition)

    @property
    def num_label_without_definition(self):
        return len(self.label_without_definition)

    @property
    def label_definition(self):
        return dict(zip(self.label_with_definition, self.definition_list))

    @property
    def definition_list(self):
        return self.definition_df.iloc[:, 1].tolist()
    
    @property
    def label2id(self):
        return dict(zip(self.label_name, range(self.num_label)))
    
    @property
    def id2label(self):
        return {v: k for k, v in self.label2id.items()}
    
    def add_definition(self, add_dict):
        for item in add_dict:
            self.definition_df = self.definition_df.append({0: item, 1: add_dict[item]}, ignore_index=True)

    def same_department(self, _key):
        if _key not in self.label_name:
            raise KeyError(f'{_key} is not a legal label')
        res = []
        department = _key[0]
        for l in self.label_name:
            if l.startswith(department) and l != _key:
                res.append(l)
        return res
    
    def same_category(self, _key):
        if _key not in self.label_name:
            raise KeyError(f'{_key} is not a legal label')
        res = []
        category = _key[:3]
        for l in self.label_name:
            if l.startswith(category) and l != _key:
                res.append(l)
        return res

    def __getitem__(self, _key):
        if _key not in self.label_definition:
            raise KeyError(f'{_key} don\'t have a definition or {_key} is not a legal label')
        return self.label_definition[_key]
        

class IPC_DATA(Dataset):
    def __init__(self, config: Config, split='train'):
        self.config = config
        self.split = split
        split_map = {'train': config.train_file, 'valid': config.valid_file, 'test': config.test_file}
        self.file = split_map[split]
        self.label = IPC_LABEL(config.label_name_file, config.label_definition_file)
        self._build_dataset()
    
    def _build_dataset(self):
        with open(self.file, 'r', encoding='utf-8') as f:
            self.ipc_examples = f.read().rstrip('\n').split('\n')
    
    def __getitem__(self, index):
        example = self.ipc_examples[index]
        example = eval(example)
        sentence = example['sentence']
        label = example['label_des']
        num_negative_labels = self.config.num_negative_labels if self.split == 'train' else 0
        num_near_labels = self.config.num_near_labels if self.split == 'train' else 0
        assert num_near_labels <= num_negative_labels, 'num_near_labels should be less than num_negative_labels'

        near_function_map = {'category': self.label.same_category, 'department': self.label.same_department}
        near_function = near_function_map[self.config.near_strategy]
        near_labels = near_function(label)
        non_near_labels = [l for l in self.label.label_name if l not in near_labels
                            and l != label]

        num_near_labels = min(num_near_labels, len(near_labels))
        near_labels = random.sample(near_labels, num_near_labels)
        non_near_labels = random.sample(non_near_labels, num_negative_labels - num_near_labels)

        return {
            'sentence': [sentence] * (1 + num_negative_labels),
            'label': [label] + near_labels + non_near_labels,
            'positive': [1] + [0] * num_negative_labels
        }

    def __len__(self):
        return len(self.ipc_examples)
