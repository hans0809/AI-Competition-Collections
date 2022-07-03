import os
import gc
import math
import copy
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.cuda import amp
from tqdm import tqdm
from collections import defaultdict
import timm
import warnings
warnings.filterwarnings("ignore")


from configs import CONFIG
from utils import set_seed, criterion,fetch_scheduler,run_training
from model import HappyWhaleModel
from dataset import prepare_loaders,prepare_df

if __name__=='__main__':
    set_seed(CONFIG['seed'])# 固定随机种子

    ##################### step1. 读取原始数据集，并制作成PyTorch的数据格式
    df=prepare_df()
    train_loader, valid_loader = prepare_loaders(df, fold=0)

    ##################### step2.构建模型、优化器和调度器
    model = HappyWhaleModel(CONFIG['model_name'], CONFIG['embedding_size'])
    model.to(CONFIG['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer)
    
    ##################### step3.训练模型
    model = run_training(train_loader, valid_loader, model, optimizer, scheduler,device=CONFIG['device'],num_epochs=CONFIG['epochs'])

