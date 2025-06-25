# Update main.py
import torch
from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
import os

def train_bert_genre():
    """Train the BERT genre model with genre-specific metrics"""
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

def setup_train(args):
    """Set up training environment"""
    export_root = os.path.join(args.experiment_dir, args.experiment_description)
    os.makedirs(export_root, exist_ok=True)
    return export_root

if __name__ == '__main__':
    train_bert_genre()
