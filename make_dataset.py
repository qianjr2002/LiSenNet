'''
$ ls /home/qianjingrui0827/data/VCTK-DEMAND/train/clean/ | wc -l
11572
$ ls /home/qianjingrui0827/data/VCTK-DEMAND/train/noisy/ | wc -l
11572
$ ls /home/qianjingrui0827/data/VCTK-DEMAND/test/clean/ | wc -l
824
$ ls /home/qianjingrui0827/data/VCTK-DEMAND/test/noisy/ | wc -l
824
'''

import os
import random
import shutil
from tqdm import tqdm

random.seed(42)

train_clean_dir = '/home/qianjingrui0827/data/VCTK-DEMAND/train/clean/'
train_noisy_dir = '/home/qianjingrui0827/data/VCTK-DEMAND/train/noisy/'
val_clean_dir = '/home/qianjingrui0827/data/VCTK-DEMAND/val/clean/'
val_noisy_dir = '/home/qianjingrui0827/data/VCTK-DEMAND/val/noisy/'

os.makedirs(val_clean_dir, exist_ok=True)
os.makedirs(val_noisy_dir, exist_ok=True)

# Get list of files
train_clean_files = os.listdir(train_clean_dir)
train_noisy_files = os.listdir(train_noisy_dir)

num_files_to_move = int(len(train_clean_files) * 0.1)

selected_files = random.sample(train_clean_files, num_files_to_move)

for file in tqdm(selected_files):
    shutil.move(os.path.join(train_clean_dir, file), os.path.join(val_clean_dir, file))
    noisy_file = file  # assuming same filename
    shutil.move(os.path.join(train_noisy_dir, noisy_file), os.path.join(val_noisy_dir, noisy_file))
