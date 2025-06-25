from .base import AbstractDataloader
# from .bert import BERTDataloader
from .bert_genre import BERTGenreDataloader
from datasets import dataset_factory

DATALOADERS = {
    # 'bert': BERTDataloader,
    'bert_genre': BERTGenreDataloader
}

def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
