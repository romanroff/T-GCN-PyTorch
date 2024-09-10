import numpy as np
import pandas as pd
import torch
import torchmetrics
import matplotlib.pyplot as plt
import utils.metrics
import utils.losses
from sklearn.metrics import mean_squared_error
from tasks import SupervisedForecastTask

import warnings
warnings.simplefilter("ignore")

# Подключаем функции для загрузки данных
def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat

def prepare_input_data(data, seq_len, normalize=True):
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    else:
        max_val = 1
    X = np.array(data[-seq_len:])
    X = np.expand_dims(X, axis=0)  # Добавляем размерность для batch
    return X, max_val

# Загрузка модели из файла .ckpt
def load_model(ckpt_path):
    model = SupervisedForecastTask.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

# Получение количества шагов предсказания от модели
def get_prediction_steps(model, X_tensor):
    with torch.no_grad():
        pred = model(X_tensor)
    return pred.shape[2]  # Размерность по временной оси

# Основной код для загрузки данных, выполнения предсказания и накопления ошибки
def predict_and_accumulate_loss(ckpt_path, feat_path, seq_len=12, desired_pre_len=12):
    # Загрузка модели
    model = load_model(ckpt_path)

    # Загрузка и подготовка данных
    feat = load_features(feat_path)
    Y_true = feat[-desired_pre_len:]
    feat = feat[:-desired_pre_len]

    # Подготовка данных для модели
    X, max_val = prepare_input_data(feat, seq_len, normalize=True)
    X_tensor = torch.FloatTensor(X)

    # Получаем количество шагов предсказания модели
    model_pre_len = get_prediction_steps(model, X_tensor)

    # Для накопления loss на каждом шаге
    losses = []

    # Выполнение предсказания и вычисление loss на каждом шаге
    total_predictions = []
    for step in range(0, desired_pre_len, model_pre_len):
        with torch.no_grad():
            pred = model(X_tensor)
        
        pred_numpy = pred.numpy()
        pred_numpy = np.transpose(pred_numpy, (0, 2, 1))
        total_predictions.append(pred_numpy[0])
        
        # Обновляем входные данные
        X = np.concatenate((X[:, model_pre_len:, :], pred_numpy), axis=1)
        X_tensor = torch.FloatTensor(X)

        # Преобразуем предсказания и реальные значения к исходному масштабу
        Y_pred_step = pred_numpy[0] * max_val
        Y_true_step = Y_true[step:step + model_pre_len]
        
        # Вычисляем loss (например, MSE) для данного шага
        step_loss = mean_squared_error(Y_true_step, Y_pred_step)
        losses.append(step_loss)

    return losses

# Функция для запуска моделей и построения графика
def run_multiple_models_and_plot_loss(pre_len_dict, general_path, feat_path, seq_len=12, desired_pre_len=48, plot_path="loss_plot.png"):
    # Создаем список для хранения всех накопленных ошибок
    all_losses = {}

    # Проходим по каждой модели
    for folder, ckpt_name in pre_len_dict.items():
        print(f"Запуск модели: {ckpt_name} из {folder}")
        
        # Определяем путь к чекпоинту
        ckpt_path = f'{general_path}/{folder}/checkpoints/{ckpt_name}.ckpt'
        
        # Выполняем предсказание и накапливаем ошибки
        losses = predict_and_accumulate_loss(ckpt_path, feat_path, seq_len, desired_pre_len)
        
        # Сохраняем ошибки для текущей модели
        all_losses[folder] = losses

    # Построение графика
    plt.figure(figsize=(10, 6))

    # Для каждой модели строим график накопленной ошибки
    for folder, losses in all_losses.items():
        plt.plot(range(1, len(losses) + 1), losses, marker='o', label=f'{folder}')
    
    plt.xlabel('Step')
    plt.ylabel('Loss (MSE)')
    plt.title('Accumulated Loss (MSE) per Prediction Step for Different Models')
    plt.legend()
    
    # Сохранение графика
    plt.savefig(plot_path)
    print(f"График ошибок сохранен в {plot_path}")

# Пример использования:
# PRE_LEN = {
#     'pre_len_1': 'epoch=499-step=25000',
#     'pre_len_3': 'epoch=360-step=18050',
#     'pre_len_6': 'epoch=420-step=21050',
#     'pre_len_9': 'epoch=499-step=25000',
#     'pre_len_12': 'epoch=476-step=23850'
# }

PRE_LEN = {
    'version_0': 'epoch=499-step=37000',
    'version_1': 'epoch=499-step=37000',
    'version_2': 'epoch=499-step=37000',
    'version_3': 'epoch=359-step=26640',
    'version_4': 'epoch=476-step=23850'
}

general_path = 'lightning_logs'
feat_path = 'data/sz_speed.csv'
plot_path = 'loss_plot.png'

# Запускаем предсказания для всех моделей и строим график
run_multiple_models_and_plot_loss(PRE_LEN, general_path, feat_path, seq_len=12, desired_pre_len=36, plot_path=plot_path)




# import numpy as np
# import pandas as pd
# import torch
# import torchmetrics
# import utils.metrics
# import utils.losses
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
# from tasks import SupervisedForecastTask

# import warnings
# warnings.simplefilter("ignore")

# # Подключаем функции для загрузки данных
# def load_features(feat_path, dtype=np.float32):
#     feat_df = pd.read_csv(feat_path)
#     feat = np.array(feat_df, dtype=dtype)
#     return feat

# def prepare_input_data(data, seq_len, normalize=True):
#     if normalize:
#         max_val = np.max(data)
#         data = data / max_val
#     else:
#         max_val = 1

#     # Берем последние seq_len наблюдений
#     X = np.array(data[-seq_len:])
#     X = np.expand_dims(X, axis=0)  # Добавляем размерность для batch
#     return X, max_val

# # Загрузка модели из файла .ckpt
# def load_model(ckpt_path):
#     model = SupervisedForecastTask.load_from_checkpoint(ckpt_path)
#     model.eval()
#     return model

# # Получение количества шагов предсказания от модели
# def get_prediction_steps(model, X_tensor):
#     with torch.no_grad():
#         pred = model(X_tensor)
#     return pred.shape[2]  # Размерность по временной оси

# # Основной код для загрузки данных, выполнения предсказания и расчета метрик
# def predict_and_evaluate(ckpt_path, feat_path, seq_len=12, desired_pre_len=12, output_path="predictions.csv"):
#     # Загрузка модели
#     model = load_model(ckpt_path)

#     # Загрузка и подготовка данных
#     feat = load_features(feat_path)
#     print(f'{feat.shape=}')

#     # Реальные значения для сравнения
#     Y_true = feat[-desired_pre_len:]
#     feat = feat[:-desired_pre_len]

#     # [12, 207]
#     X, max_val = prepare_input_data(feat, seq_len, normalize=True)
#     X_tensor = torch.FloatTensor(X)
#     print(f'{Y_true.shape, feat.shape, X.shape=}')

#     # Получаем количество шагов предсказания модели
#     model_pre_len = get_prediction_steps(model, X_tensor)
#     print(f"Модель предсказывает {model_pre_len} шага(ов) за один раз.")
    
#     # Выполнение предсказания
#     total_predictions = []
#     for _ in range(0, desired_pre_len, model_pre_len):
#         with torch.no_grad():
#             pred = model(X_tensor)
        
#         # Преобразуем предсказания к исходному масштабу и сохраняем
#         pred_numpy = pred.numpy()# * max_val
#         # [3, 207]
#         pred_numpy = np.transpose(pred_numpy, (0, 2, 1))  # Приводим к (1, 3, 207)
#         total_predictions.append(pred_numpy[0])
        
#         # Обновляем входные данные, добавляя предсказанные значения к последовательности
#         # X[:, model_pre_len:, :] -> [9, 207], X = [12, 207]
#         X = np.concatenate((X[:, model_pre_len:, :], pred_numpy), axis=1)
#         X_tensor = torch.FloatTensor(X)
    
#     # Объединяем все предсказания
#     total_predictions = np.concatenate(total_predictions, axis=0)  # (12, 207)
#     total_predictions *= max_val
#     feat_df = pd.read_csv(feat_path)
    
#     # Транспонируем предсказания для создания DataFrame
#     total_predictions_df = pd.DataFrame(total_predictions, columns=feat_df.columns)
    
#     # Сохраняем предсказания в CSV файл
#     total_predictions_df.to_csv(output_path, index=False)
#     print(f"Предсказания сохранены в {output_path}")
    
#     # Вычисляем метрики
#     Y_pred = total_predictions
    
#     # Преобразование numpy массивов в тензоры
#     Y_true_tensor = torch.tensor(Y_true[:3])
#     Y_pred_tensor = torch.tensor(Y_pred[:3])

#     # Вычисление метрик
#     rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(Y_pred_tensor.reshape(-1), Y_true_tensor.reshape(-1)))
#     mae = torchmetrics.functional.mean_absolute_error(Y_pred_tensor.reshape(-1), Y_true_tensor.reshape(-1))
#     accuracy = utils.metrics.accuracy(Y_pred_tensor, Y_true_tensor)
#     r2 = utils.metrics.r2(Y_pred_tensor, Y_true_tensor)
#     explained_variance = utils.metrics.explained_variance(Y_pred_tensor, Y_true_tensor)

#     loss = mean_squared_error(Y_true, Y_pred)  # Значение loss

#     # Вывод метрик
#     metrics = {
#         "val_loss": loss,
#         "RMSE": rmse.item(),
#         "MAE": mae.item(),
#         "accuracy": accuracy.item(),
#         "R2": r2.item(),
#         "ExplainedVar": explained_variance.item(),
#     }

#     # Создание DataFrame
#     df_metrics = pd.DataFrame([metrics])

#     # Вывод с форматированием
#     print("Metrics:")
#     print(df_metrics.to_string(index=False))
#     return metrics

# # Задаем пути к файлам

# def run_multiple_models(pre_len_dict, general_path, feat_path, seq_len=12, desired_pre_len=48):
#     # Создаем список для сохранения всех метрик
#     all_metrics = []

#     # Проходим по каждому элементу в словаре PRE_LEN
#     for folder, ckpt_name in pre_len_dict.items():
#         print(f"Запуск модели: {ckpt_name} из {folder}")
        
#         # Определяем путь к чекпоинту и файлу для сохранения предсказаний
#         ckpt_path = f'{general_path}/{folder}/checkpoints/{ckpt_name}.ckpt'
#         output_path = f'{general_path}/{folder}/predictions.csv'
        
#         # Выполняем предсказание и оцениваем метрики
#         metrics = predict_and_evaluate(ckpt_path, feat_path, seq_len, desired_pre_len, output_path)
        
#         # Добавляем имя модели в метрики
#         metrics["model"] = ckpt_name
#         metrics["folder"] = folder
        
#         # Сохраняем метрики в общий список
#         all_metrics.append(metrics)

#     # Создаем итоговый DataFrame для всех моделей
#     df_all_metrics = pd.DataFrame(all_metrics)
    
#     # Выводим и сохраняем все метрики в файл
#     print("\nВсе метрики:")
#     print(df_all_metrics.to_string(index=False))
#     df_all_metrics.to_csv(f'{general_path}/all_metrics.csv', index=False)

# # Задаем пути к файлам
# PRE_LEN = {
#     'pre_len_1': 'epoch=499-step=25000',
#     'pre_len_3': 'epoch=360-step=18050',
#     'pre_len_6': 'epoch=420-step=21050',
#     'pre_len_9': 'epoch=499-step=25000',
#     'pre_len_12': 'epoch=476-step=23850'
# }

# general_path = 'lightning_logs'
# feat_path = 'data/los_speed.csv'

# # Запускаем предсказания для всех моделей
# run_multiple_models(PRE_LEN, general_path, feat_path, seq_len=12, desired_pre_len=36)
