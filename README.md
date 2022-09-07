# label-embedding

## 数据集

存在data/data目录下

![数据集](微信截图_20220907174440.png)

## 运行命令

### 单卡
```
python train.py --no_distribute
```

### 多卡
```
python -m torch.distributed.launch --nproc num_gpus train.py
```

`num_gpus`填显卡个数
