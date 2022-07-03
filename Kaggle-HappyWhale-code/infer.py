import os
import gc
import cv2
import math
import copy
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import joblib
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import StratifiedKFold
import timm
import faiss
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore")

from configs import CONFIG
from utils import set_seed, criterion,fetch_scheduler,run_training, get_embeddings,get_predictions,map_per_image
from model import HappyWhaleModel
from dataset import prepare_loaders, prepare_df, HappyWhaleDataset,data_transforms

if __name__ =='__main__':
    CKPT_PATH='best_weight.pth'
    model = HappyWhaleModel(CONFIG['model_name'], CONFIG['embedding_size'])
    model.load_state_dict(torch.load(CKPT_PATH))
    model.to(CONFIG['device'])

    train_loader, valid_loader = prepare_loaders(df, fold=0)
    train_embeds, train_labels, train_ids = get_embeddings(model, train_loader, CONFIG['device'])
    valid_embeds, valid_labels, valid_ids = get_embeddings(model, valid_loader, CONFIG['device'])
    # 归一化embedding
    train_embeds = normalize(train_embeds, axis=1, norm='l2')
    valid_embeds = normalize(valid_embeds, axis=1, norm='l2')

    encoder = LabelEncoder()

    with open("le.pkl", "rb") as fp:
        encoder = joblib.load(fp)

    train_labels = encoder.inverse_transform(train_labels)
    valid_labels = encoder.inverse_transform(valid_labels)

    index = faiss.IndexFlatIP(CONFIG['embedding_size'])
    index.add(train_embeds)
    D, I = index.search(valid_embeds, k=50)
    allowed_targets = np.unique(train_labels)
    val_targets_df = pd.DataFrame(np.stack([valid_ids, valid_labels], axis=1), columns=['image','target'])
    val_targets_df.loc[~val_targets_df.target.isin(allowed_targets), 'target'] = 'new_individual'

    valid_df = []
    for i, val_id in tqdm(enumerate(valid_ids)):
        targets = train_labels[I[i]]
        distances = D[i]
        subset_preds = pd.DataFrame(np.stack([targets,distances],axis=1),columns=['target','distances'])
        subset_preds['image'] = val_id
        valid_df.append(subset_preds)

    valid_df = pd.concat(valid_df).reset_index(drop=True)
    valid_df = valid_df.groupby(['image','target']).distances.max().reset_index()

    valid_df = valid_df.sort_values('distances', ascending=False).reset_index(drop=True)
    # valid_df.to_csv('val_neighbors.csv')

    sample_list = ['938b7e931166', '5bf17305f073', '7593d2aee842', '7362d7a01d00','956562ff2888']

    best_th = 0
    best_cv = 0
    for th in [0.1*x for x in range(11)]:
        all_preds = get_predictions(sample_list,valid_df, threshold=th)
        cv = 0
        for i,row in val_targets_df.iterrows():
            target = row.target
            preds = all_preds[row.image]
            val_targets_df.loc[i,th] = map_per_image(target, preds)
        cv = val_targets_df[th].mean()
        print(f"CV at threshold {th}: {cv}")
        if cv > best_cv:
            best_th = th
            best_cv = cv

    print("Best threshold", best_th)
    print("Best cv", best_cv)

    ## Adjustment: Since Public lb has nearly 10% 'new_individual' (Be Careful for private LB)
    val_targets_df['is_new_individual'] = val_targets_df.target=='new_individual'
    print(val_targets_df.is_new_individual.value_counts().to_dict())
    val_scores = val_targets_df.groupby('is_new_individual').mean().T
    val_scores['adjusted_cv'] = val_scores[True]*0.1+val_scores[False]*0.9
    best_threshold_adjusted = val_scores['adjusted_cv'].idxmax()
    print("best_threshold",best_threshold_adjusted)


    # inference
    train_embeds = np.concatenate([train_embeds, valid_embeds])
    train_labels = np.concatenate([train_labels, valid_labels])
    # print(train_embeds.shape,train_labels.shape)

    index = faiss.IndexFlatIP(CONFIG['embedding_size'])
    index.add(train_embeds)

    test = pd.DataFrame()
    test["image"] = os.listdir("../input/happy-whale-and-dolphin/test_images")
    test["file_path"] = test["image"].apply(lambda x: f"{CONFIG.TEST_DIR}/{x}")
    test["individual_id"] = -1  #dummy value

    test_dataset = HappyWhaleDataset(test, transforms=data_transforms["valid"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['valid_batch_size'], num_workers=2, shuffle=False, pin_memory=True)

    test_embeds, _, test_ids = get_embeddings(model, test_loader, CONFIG['device'])
    test_embeds = normalize(test_embeds, axis=1, norm='l2')

    D, I = index.search(test_embeds, k=50)

    test_df = []
    for i, test_id in tqdm(enumerate(test_ids)):
        targets = train_labels[I[i]]
        distances = D[i]
        subset_preds = pd.DataFrame(np.stack([targets, distances], axis=1), columns=['target','distances'])
        subset_preds['image'] = test_id
        test_df.append(subset_preds)
        
    test_df = pd.concat(test_df).reset_index(drop=True)
    test_df = test_df.groupby(['image','target']).distances.max().reset_index()
    test_df = test_df.sort_values('distances', ascending=False).reset_index(drop=True)
    # test_df.to_csv('test_neighbors.csv')

    # 生成提交结果
    predictions = get_predictions(test_df, best_threshold_adjusted)
    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ['image','predictions']
    predictions['predictions'] = predictions['predictions'].apply(lambda x: ' '.join(x))
    predictions.to_csv('submission.csv',index=False)