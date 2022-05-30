from src.utils import args
from src.models import *
from src.image_generator import ImageGenerator
from src.descriptors import RepresentationGenerator
from src.graph import GraphGenerator
from src.factory import kfold_fastai
import os
import glob
import pandas as pd

def empty_file(path):
    files = glob.glob(path + '/*')
    for file in files:
        os.remove(file)
    return None

if __name__ == '__main__':
    if args.input == 'image':
        img_gen = ImageGenerator(args.dataset)  # Generate the training images - uses all cpu cores
        kfold_fastai()   # Apply fastai model
        empty_file(f'./data/{args.dataset}/aug_images')     # Storage cleaning!

    elif 'graph' in args.input:
        graph_gen = GraphGenerator(args.dataset)
        graphs = graph_gen.ml_set
        print('testing breakpoint')

    else:
        desc_gen = RepresentationGenerator(args.dataset)
        desc_df = desc_gen.ml_set

        labels = np.array(desc_df.loc[:, 'label'])
        features = np.array(desc_df.drop('label', axis=1))
        print(features.shape)

        if args.dataset == 'cocrystal':
            model = random_forest_classifier(features, labels, do_kfold=True)
        elif args.dataset == 'solubility':
            model = random_forest_regressor(features, labels, do_kfold=True)


        # still need to implement all descriptor baselines
        ## implement ANN
        ### implement other Desc types!
        ##### Add oversampling
        ###### Make non kfold for desc models - break the code after 1 itteration of kfold?
        ####### RUN RUN RUN!!!

