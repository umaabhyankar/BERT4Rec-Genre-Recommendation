from .bert import BERTTrainer
# from .dae import DAERecommenderTrainer
# from .vae import VAETrainer
from .bert_genre import BERTGenreTrainer  # Import our genre trainer

TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    # DAERecommenderTrainer.code(): DAERecommenderTrainer,
    # VAETrainer.code(): VAETrainer,
    BERTGenreTrainer.code(): BERTGenreTrainer  # Add our genre trainer
}

def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
