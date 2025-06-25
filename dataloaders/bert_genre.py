from .base import AbstractDataloader
import numpy as np
import torch

class BERTGenreDataloader(AbstractDataloader):
    @classmethod
    def code(cls):
        return 'bert_genre'

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.dataset = dataset
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.dataset.get_num_items() + 1

    def _get_dataloader(self, data, shuffle):
        # Pre-allocate numpy arrays
        n_samples = len(data)
        sequences = np.zeros((n_samples, self.max_len), dtype=np.int32)
        labels = np.zeros(n_samples, dtype=np.int32)
        genre_matrices = []

        idx = 0
        for user_id, items in data.items():
            # Add items to sequence
            seq = np.zeros([self.max_len], dtype=np.int32)
            seq_idx = self.max_len - 1

            for i in reversed(items[:-1]):
                seq[seq_idx] = i
                seq_idx -= 1
                if seq_idx == -1: break

            sequences[idx] = seq

            # Get positive item and its genre matrix
            pos_id = items[-1]
            genre_matrix = self.dataset.get_genre_matrix([pos_id])
            genre_matrices.append(genre_matrix.numpy())

            labels[idx] = pos_id
            idx += 1

        # Convert to tensors
        sequences = torch.LongTensor(sequences)
        labels = torch.LongTensor(labels)
        genre_matrices = torch.FloatTensor(np.stack(genre_matrices))

        dataset = torch.utils.data.TensorDataset(sequences, labels, genre_matrices)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            shuffle=shuffle,
            num_workers=self.args.workers
        )

    def get_pytorch_dataloaders(self):
        train = self._get_dataloader(self.dataset.train, shuffle=True)
        val = self._get_dataloader(self.dataset.val, shuffle=False)
        test = self._get_dataloader(self.dataset.test, shuffle=False)
        return train, val, test
