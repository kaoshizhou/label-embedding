from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 4
    
    def __getitem__(self, index):
        return {
            'key1': ['ab', 'ab'],
            'key2': ['A01K', 'A01M']
        }

def fn(batch):
    key_list = batch[0].keys()
    res = {key: [] for key in key_list}
    for b in batch:
        for k in key_list:
            res[k].extend(b[k])
    return res

if __name__ == '__main__':
    dataset = Data()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=fn)
    for item in dataloader:
        print(item)