from src.utils import args
from fastai.vision.all import *
import pandas as pd
from sklearn.model_selection import KFold
import random
from sklearn.metrics import accuracy_score
import cv2
import torch
import numpy as np

def train_fastai_model_classification(model_df):
    # print(model_df.head(20), model_df.shape)
    dls = ImageDataLoaders.from_df(model_df,
                                   fn_col=0,
                                   label_col=1,
                                   valid_col=2,
                                   item_tfms=None,
                                   batch_tfms=None,
                                   y_block=CategoryBlock(),
                                   bs=32)
    metrics = [error_rate, accuracy, Precision(), Recall(), F1Score(), RocAuc()]
    learn = cnn_learner(dls, resnet18, metrics=metrics)
    learn.fine_tune(50, cbs=[SaveModelCallback(fname=f'./checkpoints/{args.dataset}/best_cbs.pth'),
                            ReduceLROnPlateau(monitor='valid_loss',
                                              min_delta=0.1,
                                              patience=2)])
    learn.export(f'./checkpoints/{args.dataset}/trained_model.pkl')

def train_fastai_model_regression(model_df):
    # print(model_df.head(20), model_df.shape)
    dls = ImageDataLoaders.from_df(model_df,
                                   fn_col=0,
                                   label_col=1,
                                   valid_col=2,
                                   item_tfms=None,
                                   batch_tfms=None,
                                   y_block=RegressionBlock(),
                                   bs=64)
    metrics = [R2Score(), mse, rmse, mae]
    learn = cnn_learner(dls, resnet18, metrics=metrics)
    learn.fine_tune(50, cbs=[SaveModelCallback(fname=f'./checkpoints/{args.dataset}/best_cbs.pth'),
                            ReduceLROnPlateau(monitor='valid_loss',
                                              min_delta=0.1,
                                              patience=2)])
    learn.export(f'./checkpoints/{args.dataset}/trained_model.pkl')