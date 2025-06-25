from .bert import BERTModel
from .bert_genre import BERTGenreModel

MODELS = {
    BERTModel.code(): BERTModel,
    BERTGenreModel.code(): BERTGenreModel
}

def model_factory(args):
    """Factory function to create model based on args"""
    model = MODELS[args.model_code]
    return model(args)
