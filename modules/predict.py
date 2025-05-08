import dill
import os
import platform
import glob
import json
import pandas as pd

from datetime import datetime

if platform.system() == "Windows":
    path = os.environ.get("PROJECT_PATH", ".")
else:  # Для Linux или других систем
    path = os.environ.get("PROJECT_PATH", "/opt/airflow/dags/plugins")


def load_model(): 
    path_pkl = glob.glob(f'{path}/data/models/*.pkl')[0] # Определяем путь до pkl модели
    
    with open(path_pkl, 'rb') as file:
        model = dill.load(file) # Читаем файл

    return model # Отдаем модель


def get_predict(model):
    test_files_path = glob.glob(f'{path}/data/test/*.json') # Считываем путь к каждому json файлу
    preds_df = pd.DataFrame() # Создаем объект pd.DataFrame для сохранения результатов предсказаний
    
    for file_path in test_files_path:
        with open(file_path, 'r') as file:
            feature = json.load(file) # Читаем json файл
        df = pd.DataFrame.from_dict([feature])
        y = model.predict(df) #Делаем предсказание с помощью считанной модели
        new_row = {'id': df.id[0], 'pred': y[0]}
        preds_df = pd.concat([preds_df, pd.DataFrame([new_row])], ignore_index=True)
        
    return preds_df # Возращаем предсказания ввиде df с содержанием id и результата
    
        
def saved_predictions(predictions_df):
    predictions_df.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
    # Сохраняем результат предсказаний в csv
    

def predict():
    model = load_model()
    preds_df = get_predict(model)
    saved_predictions(preds_df)


if __name__ == '__main__':
    predict()
