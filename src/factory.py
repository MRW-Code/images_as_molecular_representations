from src.utils import args
from src.image_processing import ImageStacker, ImageAugmentations
from src.preprocessing import split_cc_dataset, get_sol_df
from src.models import *
from src.descriptors import RepresentationGenerator
from src.graph import GraphGenerator
from tqdm import tqdm
import os
import glob
import re
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

def get_aug_df(dataset):
    print('GETTING AUG DF')
    image_dir = f'./data/{dataset}/aug_images'
    paths = [f'{image_dir}/{x}' for x in tqdm(os.listdir(image_dir))]
    if args.dataset == 'solubility':
        labels = [float(re.findall(r'.*__(.*).png', y)[0]) for y in tqdm(paths)]
    elif args.dataset == 'cocrystal':
        labels = [int(re.findall(r'.*__(.*).png', y)[0]) for y in tqdm(paths)]
    model_df = pd.DataFrame({'path': paths,
                             'label': labels})
    model_df['is_valid'] = 0
    return model_df

def kfold_fastai(n_splits=10):
    if args.dataset == 'cocrystal':
        # get stacked images
        img_stacker = ImageStacker()
        raw_df = img_stacker.fastai_data_table()

        # split into unique pairs
        unique_raw = raw_df.iloc[::2]
        unique_raw = unique_raw.reset_index(drop=True)
        paths = unique_raw['path']
        labels = unique_raw['label']

        # start kfold
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
        best_metrics = []
        count = 0
        for train_index, val_index in tqdm(kfold.split(paths, labels)):
            X_train_single, X_val = paths.reindex(index=train_index), paths.reindex(index=val_index)
            y_train_single, y_val = labels.reindex(index=train_index), labels.reindex(index=val_index)

            # make sure flips and augs are only applied to training set
            x_indexes = raw_df.loc[(raw_df['path'].isin(X_train_single))].index
            X_train = []
            y_train = []
            for idx in x_indexes:
                if idx % 2 == 0:
                    X_train.append(raw_df.loc[idx, 'path'])
                    X_train.append(raw_df.loc[idx+1, 'path'])
                    y_train.append(raw_df.loc[idx, 'label'])
                    y_train.append(raw_df.loc[idx+1, 'label'])
                else:
                    X_train.append(raw_df.loc[idx, 'path'])
                    X_train.append(raw_df.loc[idx-1, 'path'])
                    y_train.append(raw_df.loc[idx, 'label'])
                    y_train.append(raw_df.loc[idx-1, 'label'])

            # Make fastai dataframe for model
            train_df = pd.DataFrame({'path' : X_train, 'label' : y_train})
            train_df['is_valid'] = 0
            val_df = pd.DataFrame({'path' : X_val, 'label' : y_val})
            val_df['is_valid'] = 1
            raw_df_new = pd.concat([train_df, val_df], axis=0)

            # Apply Augmentations to train set
            if not args.no_augs:
                augmentor = ImageAugmentations(args.dataset)
                augmentor.do_image_augmentations(raw_df_new)
                aug_df = get_aug_df(args.dataset)
                model_df = pd.concat([aug_df, val_df], axis=0)
            else:
                model_df = raw_df_new

            # Fastai model applied
            trainer = train_fastai_model_classification(model_df, count)
            model = load_learner(f'./checkpoints/{args.dataset}/trained_model_{args.no_augs}_{count}.pkl', cpu=True)
            best_metrics.append(model.final_record)
            count +=1

        # Return best metrics
        print(best_metrics)
        print(f'mean acc = {np.mean([best_metrics[x][2] for x in range(n_splits)])}')
        print(f'mean roc = {np.mean([best_metrics[x][3] for x in range(n_splits)])}')

    if args.dataset == 'solubility':
        # Get dataset
        raw_df = get_sol_df()
        paths = raw_df['path']
        labels = raw_df['label']

        # Kfold split
        kfold = KFold(n_splits=n_splits, shuffle=True)
        best_metrics = []
        count = 0
        for train_index, val_index in tqdm(kfold.split(paths)):
            X_train_single, X_val = paths.reindex(index=train_index), paths.reindex(index=val_index)
            y_train_single, y_val = labels.reindex(index=train_index), labels.reindex(index=val_index)

            train_df = pd.DataFrame({'path': X_train_single, 'label': y_train_single})
            train_df['is_valid'] = 0
            val_df = pd.DataFrame({'path': X_val, 'label': y_val})
            val_df['is_valid'] = 1
            raw_df_new = pd.concat([train_df, val_df], axis=0)

            # Apply Augmentations to train set
            if not args.no_augs:
                augmentor = ImageAugmentations(args.dataset)
                augmentor.do_image_augmentations(raw_df_new)
                aug_df = get_aug_df(args.dataset)
                model_df = pd.concat([aug_df, val_df], axis=0)
            else:
                model_df = raw_df_new

            # Fastai model applied
            trainer = train_fastai_model_regression(model_df, count)

            model = load_learner(f'./checkpoints/{args.dataset}/trained_model_{args.no_augs}_{count}.pkl', cpu=True)
            best_metrics.append(model.final_record)
            count +=1

            # Return best metrics
        print(best_metrics)
        print(f'mean R2 = {np.mean([best_metrics[x][1] for x in range(n_splits)])}')
        print(f'mean mse = {np.mean([best_metrics[x][2] for x in range(n_splits)])}')
        print(f'mean rmse = {np.mean([best_metrics[x][3] for x in range(n_splits)])}')
        print(f'mean mae = {np.mean([best_metrics[x][4] for x in range(n_splits)])}')

    return None
