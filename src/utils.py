import argparse
import os
import torch

parser = argparse.ArgumentParser(usage='python main.py')

parser.add_argument('-i', '--input', action='store', dest='input',
                    default='image', choices=['image', 'mordred_descriptor', 'rdkit_descriptor',
                                              'mol2vec', 'ecfp', 'pubchem_fp', 'maccs',
                                              'spectrophore'])
parser.add_argument('-d', '--dataset', action='store', dest='dataset',
                    default='None', choices=['None', 'solubility', 'cocrystal'])
parser.add_argument('--no_augs', action='store_true', dest='no_augs', default=False)
parser.add_argument('--gpu_idx', action='store', dest='gpu_idx', default='0',
                  choices=['0', '1', '2', '3', '4', '5'])
parser.add_argument('--cpu', action='store_true', dest='cpu', default=False)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')