import json
import os
import time

import math
import numpy as np
import torch
import argparse
import pandas as pd
from functools import partial
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import r2_score
try:
    from sklearn.metrics import root_mean_squared_error as mean_squared_error
except:
    from sklearn.metrics import mean_squared_error
    mean_squared_error = partial(mean_squared_error, squared=False)
from inference.predictor import LimiXPredictor
from utils.inference_utils import generate_infenerce_config
import torch.distributed as dist
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from utils.utils import  download_datset, download_model

if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')


def inference_dataset(X_train, X_test, y_train, y_test, model):
    """
    Process the dataset, perform inference, calculate RMSE and R²
    """
    sample_size, feature_count = X_train.shape
    rmse_results = {"Sample_Size": sample_size, "Feature_Count": feature_count}
    r2_results = {}

    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_normalized = (y_train - y_mean) / y_std
    y_test_normalized = (y_test - y_mean) / y_std
    
    y_pred = model.predict(X_train, y_train_normalized, X_test)    

    # calculate RMSE and R²
    y_pred = y_pred.to('cpu')
    rmse = mean_squared_error(y_test_normalized, y_pred)
    r2 = r2_score(y_test_normalized, y_pred)

    r2_results[f"R2"] = r2
    rmse_results["rmse"] = rmse
    
    pred_result = {'label':y_test}
    pred_result['pred'] = y_pred * y_std +y_mean

    return rmse_results, r2_results, pred_result


def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LimiX inference')
    parser.add_argument('--data_dir', type=str, default=None, help='Specify the local storage directory of the dataset')
    parser.add_argument('--save_name', default=None, type=str, help="path to save result")
    parser.add_argument('--inference_config_path', type=str,required=True, help="path to example config")
    parser.add_argument('--model_path',type=str, default=None, help="path to you model")
    parser.add_argument('--inference_with_DDP', default=False, action='store_true', help="Inference with DDP")
    parser.add_argument('--debug', default=False, action='store_true', help="debug mode")
    args = parser.parse_args()
    model_file = args.model_path
    data_root = args.data_dir
    
    if data_root is None:
        download_datset(repo_id="stableai-org/bcco_reg", revision="main", save_dir="./cache")
        data_root = "./cache/bcco_reg"
    if model_file is None:
        model_file = download_model(repo_id="stableai-org/LimiX-16M", filename="LimiX-16M.ckpt", save_path="./cache")
 
    if args.save_name is None:
        # Dynamically generate the save path
        args.save_name = time.strftime("%Y%m%d-%H%M%S")

    save_root = f"./result/{args.save_name}"
    os.makedirs(save_root, exist_ok=True)

    if not os.path.exists(args.inference_config_path):
        generate_infenerce_config(args)

    with open(args.inference_config_path, 'r') as f:
        inference_config = json.load(f)

    save_result_path = os.path.join(save_root, f"all_rst.csv")
    save_config_path = os.path.join(save_root, "config.json")
    with open(save_config_path, "w") as f:
        json.dump(inference_config, f)

    model = LimiXPredictor(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                model_path=model_file, inference_config=inference_config,
                                inference_with_DDP=args.inference_with_DDP,task_type="Regression")

    rsts = []
    for idx, dataset_name in tqdm(enumerate(os.listdir(data_root))):
        try:
            train_data_path = Path(data_root, dataset_name, f'{dataset_name}_train.csv')
            test_data_path = Path(data_root, dataset_name, f'{dataset_name}_test.csv')
            
            if os.path.isfile(os.path.join(data_root, dataset_name)):
                continue
            
            X_train, y_train = train_data = load_data(train_data_path)
            X_test, y_test = test_data = load_data(test_data_path)
            rst = {
                'dataset name': dataset_name,
                'num_data_train': len(X_train),
                'num_data_test': len(X_test),
                'num_feat': X_train.shape[1],
                'num_class': len(np.unique(y_train)),
            }
            rmse_results, r2_results, pred_result = inference_dataset(X_train, X_test, y_train, y_test, model)
            

            if args.debug:
                print(f"[{idx}] {dataset_name} -> {rmse_results}, {r2_results}")
            if not (int(os.environ.get('WORLD_SIZE', -1)) > 0 and dist.get_rank() != 0):
                rst.update(**rmse_results)
                rst.update(**r2_results)
                rsts.append(rst)
                pd.DataFrame(pred_result).to_csv(os.path.join(save_root, rst['dataset name']+'_pred_LimiX.csv'), index=False)
                
        except Exception as e:
            # raise e
            print(f"Error processing {dataset_name}: {e}")
    if not (int(os.environ.get('WORLD_SIZE', -1)) > 0 and dist.get_rank() != 0):
        rstsdf = pd.DataFrame(rsts)
        rstsdf.to_csv(os.path.join(save_root, 'all_rst.csv'), index=False)


    


