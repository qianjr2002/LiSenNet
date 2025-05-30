import argparse
import sys
import yaml
import pytorch_lightning as pl
from pathlib import Path

from model import Model
from data_module import DataModule

import torch

def main(args):

    torch.set_float32_matmul_precision('medium')  # Fixes Tensor Core warning
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.save_enhanced is not None:
        config['save_enhanced'] = args.save_enhanced
        Path(args.save_enhanced).mkdir(parents=True, exist_ok=True)
    
    model = Model(config=config)
    data_module = DataModule(**config['dataset_config'])
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        logger=False,
    )

    trainer.test(model, data_module, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test model')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--save_enhanced', type=str, default=None, help='The dir path to save enhanced wavs.')

    args = parser.parse_args()
    sys.exit(main(args))

'''
 CUDA_VISIBLE_DEVICES=3 python test.py --config ./config.yaml --ckpt_path  log/ckpts/version_0/epoch\=99-last.ckpt
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_pesq         │     3.055063247680664     │
│        test_sisnr         │    12.762089729309082     │
│         test_stoi         │    0.9369952082633972     │
└───────────────────────────┴───────────────────────────┘
'''