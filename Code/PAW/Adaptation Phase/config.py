import argparse
import os
import torch
parser = argparse.ArgumentParser(description='CAiDA')
parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
parser.add_argument('--max_epoch', type=int, default=100, help="max iterations")
parser.add_argument('--interval', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
parser.add_argument('--lr', type=float, default=1 * 1e-3, help="learning rate")
parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--not_use_lrsch',action='store_true')
parser.add_argument('--lr_decay1', type=float, default=0.1)

# source weight
parser.add_argument('--reduction_ratio', type=float, default=2, help="attention weighted module MLP bottoleneck")

# loss determine and weight
parser.add_argument('--gent', type=bool, default=True)
parser.add_argument('--gent_source',type=bool, default=True)
parser.add_argument('--ent', type=bool, default=True)
parser.add_argument('--cls_par', type=float, default=0.7)
parser.add_argument('--ent_par', type=float, default=1.0)
parser.add_argument('--gent_par', type=float, default=1)
parser.add_argument('--gent_source_par', type=float, default=0.1)

# other experimental parameters
parser.add_argument('--low_confi_weight', type=float, default=0.0)
parser.add_argument('--confident_ratio', type=float, default=0.5)

# experient paradigm
parser.add_argument('--dataset', type=str, default='2a',choices=['2a','2b'])
parser.add_argument('--base_model',type=str, default='eegnet',choices=['eegnet', 'eegtcnet'])
parser.add_argument('--paradigm', type=str, default='offline',choices=['offline','online'])

# wandb
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--not_use_wandb',action='store_true')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
