import sys
import argparse
import torch
from templates import set_template

# Check if running in Colab
is_colab = 'ipykernel' in sys.argv[0]

# Create argument parser
parser = argparse.ArgumentParser(description='RecPlay')

# Add all arguments
parser.add_argument('--dataset_code', type=str, default='ml-1m', choices=['ml-1m', 'ml-20m'])
parser.add_argument('--min_rating', type=int, default=0, help='Minimum rating to include')
parser.add_argument('--min_uc', type=int, default=5, help='Filter threshold for users')
parser.add_argument('--min_sc', type=int, default=0, help='Filter threshold for items')
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')

parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--eval_set_size', type=int, default=500,
                    help='Size of val and test set when running evaluation')

# Dataloader arguments
parser.add_argument('--dataloader_random_seed', type=int, default=98765)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--workers', type=int, default=0)  # Added workers argument

# Negative sampler arguments
parser.add_argument('--train_negative_sampler_code', type=str, default='random',
                    choices=['popular', 'random'], help='Negative sampling technique for training')
parser.add_argument('--train_negative_sample_size', type=int, default=0)
parser.add_argument('--train_negative_sampling_seed', type=int, default=0)
parser.add_argument('--test_negative_sampler_code', type=str, default='random',
                    choices=['popular', 'random'], help='Negative sampling technique for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=98765)

# Model arguments
parser.add_argument('--model_code', type=str, default='bert_genre', choices=['bert', 'dae', 'vae', 'bert_genre'])
parser.add_argument('--model_init_seed', type=int, default=0)

# BERT arguments
parser.add_argument('--bert_max_len', type=int, default=100)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=4)
parser.add_argument('--bert_hidden_units', type=int, default=256)
parser.add_argument('--bert_dropout', type=float, default=0.1)
parser.add_argument('--bert_mask_prob', type=float, default=0.15)

# Device configuration
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--device_idx', type=str, default='0')
parser.add_argument('--num_gpu', type=int, default=1 if torch.cuda.is_available() else 0)

# Experiment arguments
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='genre_recommendation')

# Add template argument
parser.add_argument('--template', type=str, default='train_bert_genre')

def is_colab_or_ipython():
    return any(word in sys.argv[0] for word in ['ipykernel', 'colab', 'kernel'])

if is_colab_or_ipython():
    # In Colab: Initialize args with default values
    args = parser.parse_args([])
else:
    args = parser.parse_args()
    set_template(args)

# Set device configuration
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device_idx = '0'
args.num_gpu = 1 if torch.cuda.is_available() else 0

# Apply template settings
args = set_template(args)
