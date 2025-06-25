# Update templates.py
import argparse
import torch
import os
import random

def train_bert_genre():
    """Train the BERT genre model with genre-specific metrics"""
    parser = argparse.ArgumentParser(description='RecPlay')
    args = parser.parse_args([])

    # Experiment settings
    args.experiment_dir = os.path.join(os.getcwd(), 'experiments')
    args.experiment_description = 'genre_recommendation'

    # Model and dataset settings
    args.model_code = 'bert_genre'
    args.dataset_code = 'ml-1m'
    args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
    args.min_uc = 5
    args.min_sc = 0
    args.split = 'leave_one_out'

    # Model initialization
    args.model_init_seed = 0

    # Dataloader settings
    args.dataloader_code = 'bert_genre'
    args.train_negative_sampler_code = 'random'
    args.train_negative_sample_size = 0
    args.train_batch_size = 64
    args.val_batch_size = 64
    args.test_batch_size = 64
    args.train_negative_sampling_seed = 0
    args.test_negative_sampling_seed = 98765
    args.dataloader_random_seed = 98765
    args.workers = 0

    # Trainer settings
    args.trainer_code = 'bert_genre'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.num_gpu = 1 if torch.cuda.is_available() else 0
    args.device_idx = 0
    args.optimizer = 'Adam'
    args.lr = 0.001
    args.num_epochs = 100   ###############################################################################  NUMBER OF EPOCHS ##################################
    args.metric_ks = [1, 5, 10]
    args.best_metric = 'NDCG@10'
    args.l2_reg = 0.0
    args.weight_decay = 0.0
    args.momentum = 0.9

    # Learning rate schedule
    args.enable_lr_schedule = True
    args.decay_step = 20
    args.gamma = 0.1

    # Logging settings
    args.log_period_as_iter = 100  # Log every 100 iterations
    args.log_period_as_epoch = 1   # Log every epoch

    # BERT genre-specific parameters
    args.bert_max_len = 100
    args.bert_num_blocks = 2
    args.bert_num_heads = 4
    args.bert_hidden_units = 256
    args.bert_dropout = 0.1
    args.bert_mask_prob = 0.15
    args.genre_embedding_size = 64
    args.num_items = 3706
    args.num_genres = 18

    return args

def set_template(args):
    if args.template == 'train_bert_genre':
        args = train_bert_genre()
    return args
