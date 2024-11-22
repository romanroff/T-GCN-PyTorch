# T-GCN-PyTorch

This is a PyTorch implementation of T-GCN in the following paper: [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320).

A stable version of this repository can be found at [the official repository](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch).

## Requirements

* numpy
* matplotlib
* pandas
* torch
* pytorch-lightning>=1.3.0
* torchmetrics>=0.3.0
* python-dotenv

## Model Training

```bash
# GCN
python main.py --model_name GCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1
# GRU
python main.py --model_name GRU --max_epochs 3000 --learning_rate 0.001 --weight_decay 1.5e-3 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1
# T-GCN
python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1
```

You can also adjust the `--data`, `--seq_len` and `--pre_len` parameters.

Run `tensorboard --logdir lightning_logs/version_0` to monitor the training progress and view the prediction results.


## Обзор `examples`

В папке `examples` находятся `.ipynb`, которые показывает следующие данные:

1. **ARIMA.ipynb**
    Хранит в себе 3 реализации предсказания трафика: `Ha`, `SVR`, `ARIMA`

2. **graph clustering.ipynb**
    Реализованы кластеризация Лувена и спектральная кластеризация для изучения параметов графа. Также проверяются такие данные, как `Betweenness`, `Closeness` и средняя скорость на узлах.

3. **graph visualization.ipynb**
    Реализованы график средней скорости на всем графе и направленный граф.

4. **main_review.ipynb**
    Реализованы кластерация Лувена, средняя скорость на кластерах и GIF-анимация изменения скорости на узлах графа с течением времени.

5. **T-GCN.ipynb**
    Собственная реализация `TGCN` из `torch_geometric_temporal`

6. **traffic_prediction.ipynb**
    Реализация использования модели `A3TGCN`
