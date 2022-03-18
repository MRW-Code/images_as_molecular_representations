from src.utils import *
from fastai.vision.all import *
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, r2_score,\
    mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch import nn
import torch
import numpy as np
from tqdm import tqdm

def train_fastai_model_classification(model_df, count):
    # print(model_df.head(20), model_df.shape)
    dls = ImageDataLoaders.from_df(model_df,
                                   fn_col=0,
                                   label_col=1,
                                   valid_col=2,
                                   item_tfms=None,
                                   batch_tfms=None,
                                   y_block=CategoryBlock(),
                                   bs=256,
                                   shuffle=True)
    metrics = [error_rate, accuracy, RocAucBinary()]
    learn = cnn_learner(dls, resnet18, metrics=metrics)
    learn.fine_tune(100, cbs=[SaveModelCallback(monitor='accuracy', fname=f'./{args.dataset}_best_cbs.pth'),
                            ReduceLROnPlateau(monitor='valid_loss',
                                              min_delta=0.1,
                                              patience=2)])
    print(learn.validate())
    learn.export(f'./checkpoints/{args.dataset}/trained_model_{args.no_augs}_{count}.pkl')

    interp = ClassificationInterpretation.from_learner(learn)
    interp.print_classification_report()

def train_fastai_model_regression(model_df, count):
    # print(model_df.head(20), model_df.shape)
    dls = ImageDataLoaders.from_df(model_df,
                                   fn_col=0,
                                   label_col=1,
                                   valid_col=2,
                                   item_tfms=None,
                                   batch_tfms=None,
                                   y_block=RegressionBlock(),
                                   bs=256,
                                   shuffle=True)
    metrics = [R2Score(), mse, rmse, mae]
    learn = cnn_learner(dls, resnet18, metrics=metrics)
    learn.fine_tune(100, cbs=[SaveModelCallback(monitor='mse', comp=np.greater, fname=f'./{args.dataset}_best_cbs.pth'),
                            ReduceLROnPlateau(monitor='valid_loss',
                                              min_delta=0.1,
                                              patience=2)])
    print(learn.validate())
    learn.export(f'./checkpoints/{args.dataset}/trained_model_{args.no_augs}_{count}.pkl')

def random_forest_classifier(features, labels, do_kfold=True):
    print(f'RUNNING RF CLASSIFIER WITH KFOLD = {do_kfold}')
    if do_kfold:
        splits = 10
        count = 0
        kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

        # Placeholders for metrics
        acc = np.empty(splits)
        roc = np.empty(splits)
        f1 = np.empty(splits)

        # stratified kfold training
        for train_index, val_index in tqdm(kfold.split(features, labels)):
            X_train, X_test = features[train_index], features[val_index]
            y_train, y_test = labels[train_index], labels[val_index]
            model = RandomForestClassifier(n_estimators=1000,
                                           n_jobs=-1,
                                           verbose=0)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # generate metrics
            f1[count] = f1_score(y_test, preds, labels=[0,1])
            acc[count] = accuracy_score(y_test, preds)
            roc[count] = roc_auc_score(y_test, preds, labels=[0,1])
            count += 1
            torch.cuda.empty_cache()
        print(f'Mean Accuracy = {np.mean(acc)}')
        print(f'Mean F1 = {np.mean(f1)}')
        print(f'Mean ROC = {np.mean(roc)}')
    return None

def random_forest_regressor(features, labels, do_kfold=True):
    print(f'RUNNING RF REGRESSOR WITH KFOLD = {do_kfold}')
    if do_kfold:
        splits = 10
        count = 0
        kfold = KFold(n_splits=splits, shuffle=True, random_state=42)

        # Placeholders for metrics
        r2 = np.empty(splits)
        mse = np.empty(splits)
        rmse = np.empty(splits)
        mae = np.empty(splits)

        # stratified kfold training
        for train_index, val_index in tqdm(kfold.split(features)):
            X_train, X_test = features[train_index], features[val_index]
            y_train, y_test = labels[train_index], labels[val_index]
            model = RandomForestRegressor(n_estimators=1000,
                                           n_jobs=-1,
                                           verbose=0)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # generate metrics
            r2[count] = r2_score(y_test, preds)
            # print(r2)
            mse[count] = mean_squared_error(y_test, preds, squared=True)
            rmse[count] = mean_squared_error(y_test, preds, squared=False)
            mae[count] = mean_absolute_error(y_test, preds)
            count += 1
        print(f'Mean R2 = {np.mean(r2)}')
        print(f'Mean mse = {np.mean(mse)}')
        print(f'Mean rmse = {np.mean(rmse)}')
        print(f'Mean mae = {np.mean(mae)}')
    return None