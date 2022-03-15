import argparse

parser = argparse.ArgumentParser(usage='python main.py')

parser.add_argument('-i', '--input', action='store', dest='input',
                    default='image', choices=['image', 'descriptor'])
parser.add_argument('-d', '--dataset', action='store', dest='dataset',
                    default='None', choices=['None', 'solubility', 'cocrystal'])
parser.add_argument('--no_augs', action='store_true', dest='no_augs', default=False)



# parser.add_argument('-m', '--model', action='store', dest='model',
#                     default='RF', choices=['RF', 'ResNet'])
# parser.add_argument('--kfold', action='store_true', dest='kfold')
# parser.add_argument('-u', '--user', action='store', dest='user',
#                     default='matt', choices=['laura', 'matt'])

args = parser.parse_args()
