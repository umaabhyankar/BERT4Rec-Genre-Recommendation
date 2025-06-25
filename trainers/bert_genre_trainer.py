from .base import AbstractTrainer as BaseTrainer
import torch
import torch.nn as nn

class BERTGenreTrainer(BaseTrainer):
    @classmethod
    def code(cls):
        return 'bert_genre'

    def __init__(self, args, model, train_dataloader, val_dataloader, test_dataloader, export_root):
        super().__init__(args, model, train_dataloader, val_dataloader, test_dataloader, export_root)
        self.criterion = nn.CrossEntropyLoss()

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        """
        batch = (seqs, labels, genre_matrix)
        logits: [batch_size, num_genres, num_items]
        labels: [batch_size, num_genres]
        """
        seqs, labels, genre_matrix = batch
        logits = self.model(seqs, genre_matrix)  # shape: [B, G, I]

        batch_size, num_genres, num_items = logits.shape
        logits = logits.view(-1, num_items)           # [B * G, I]
        labels = labels.view(-1)                      # [B * G]

        loss = self.criterion(logits, labels)         # CrossEntropyLoss across all genre-item targets
        return loss

    def calculate_metrics(self, batch):
        """
        Calculate Recall@5 per genre, averaged over all users
        """
        seqs, labels, genre_matrix = batch
        logits = self.model(seqs, genre_matrix)  # [B, G, I]

        # Top-5 item indices per genre
        top5 = torch.topk(logits, 5, dim=2).indices  # Expecting: [B, G, 5]

        # Add these debug prints
        print("top5 shape:", top5.shape)        # Should be [B, G, 5]
        print("labels shape:", labels.shape)    # Should be [B]

        # Ensure proper shape alignment: [B, G, 1]
        labels_expanded = labels.unsqueeze(1).expand(-1, top5.size(1)).unsqueeze(2)
        print("labels_expanded shape:", labels_expanded.shape)  # Should be [B, G, 1]

        # Check if label is in top-5 per genre
        hits = (top5 == labels_expanded).any(dim=2).float()  # [B, G]

        recall_5 = hits.mean().item()  # scalar average over all users and genres

        return {
            "Recall@5": recall_5
        }
