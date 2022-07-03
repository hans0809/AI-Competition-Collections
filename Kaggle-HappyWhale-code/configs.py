import torch

CONFIG = {"seed": 2022,
          "epochs": 4,
          "img_size": 448,
          "model_name": "efficientnet_b7",
          "num_classes": 15587,
          "embedding_size": 2048,
          "train_batch_size": 32,
          "valid_batch_size": 64,
          "learning_rate": 1e-4,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_fold": 5,
          "n_accumulate": 1,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          # ArcFace Hyperparameters
          "s": 15.0, 
          "m": 0.30,
          "ls_eps": 0.0,
          "easy_margin": False,
          "ROOT_DIR" :'happy-whale-and-dolphin',
          "TRAIN_DIR" : 'happy-whale-and-dolphin/train_images',
          "TEST_DIR" :'happy-whale-and-dolphin/test_images'
          }