import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd

import psutil

cudnn.benchmark = True
cudnn.deterministic = False


def get_memory_usage_psutil():
    """以字节为单位返回当前脚本的内存使用量 (常驻内存)。"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # 常驻内存 (Resident Set Size)


def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

if __name__ == "__main__":

    print(f"suhao ============================================================== 当前脚本占用的内存 (psutil 模块): {get_memory_usage_psutil():.2f} MB")

    opt = get_config("config_files/en_mixed_filtered_config.yaml")
    train(opt, show_number = 5 ,amp=False)


