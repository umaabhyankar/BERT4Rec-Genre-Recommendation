from .base import AbstractTrainer
import torch
import torch.nn as nn
from tqdm import tqdm

class BERTGenreTrainer(AbstractTrainer):
    @classmethod
    def code(cls):
        return 'bert_genre'

    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.best_metric = args.best_metric
        self.num_items = args.num_items

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, labels, genre_matrices = batch
        seqs = seqs.to(self.device)
        labels = labels.to(self.device)
        genre_matrices = genre_matrices.to(self.device)

        logits = self.model(seqs, genre_matrices)  # [B, G, I]
        B, G, I = logits.size()

        # Fix shape: from [B, 1, G] → [B, G]
        if genre_matrices.shape[1] == 1:
            genre_matrices = genre_matrices.squeeze(1)

        genre_indices = torch.argmax(genre_matrices, dim=1)  # [B]
        gather_indices = genre_indices.view(B, 1, 1).expand(-1, 1, I)  # [B, 1, I]
        genre_logits = logits.gather(dim=1, index=gather_indices).squeeze(1)  # [B, I]

        return self.criterion(genre_logits, labels)

    def calculate_metrics(self, batch):
        seqs, labels, genre_matrices = batch
        seqs = seqs.to(self.device)
        labels = labels.to(self.device)
        genre_matrices = genre_matrices.to(self.device)

        logits = self.model(seqs, genre_matrices)  # [B, G, I]
        batch_size, num_genres, num_items = logits.size()

        top5 = torch.topk(logits, 5, dim=2).indices  # [B, G, 5]
        labels_expanded = labels.unsqueeze(1).expand(-1, num_genres).unsqueeze(2)  # [B, G, 1]
        hits = (top5 == labels_expanded).any(dim=2).float()  # [B, G]
        recall_per_genre = hits.mean(dim=0)  # [G]

        metrics = {f"Recall@5_Genre{g}": recall_per_genre[g].item() for g in range(num_genres)}
        metrics["Recall@5_Avg"] = recall_per_genre.mean().item()
        return metrics

    def train_epoch(self, epoch_idx):
        self.model.train()
        avg_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch_idx}')):
            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
        avg_loss /= (batch_idx + 1)
        return {'avg_loss': avg_loss}

    def validate_epoch(self, epoch_idx):
        self.model.eval()
        metrics = {}
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch_metrics = self.calculate_metrics(batch)
                for k, v in batch_metrics.items():
                    metrics[k] = metrics.get(k, 0) + v
        for k in metrics:
            metrics[k] /= len(self.val_loader)
        return metrics

    def test(self):
        self.model.eval()
        metrics = {}
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                batch_metrics = self.calculate_metrics(batch)
                for k, v in batch_metrics.items():
                    metrics[k] = metrics.get(k, 0) + v
        for k in metrics:
            metrics[k] /= len(self.test_loader)

        # Print and save metrics
        print("\n📊 Final Evaluation (Test Set):")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        import json
        with open("final_recall_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def get_genre_recommendations(self):
        self.model.eval()
        recommendations = {}  # {user_id: {genre: [item_ids]}}

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Generating Genre-Based Recommendations'):
                seqs, labels, genre_matrices = [x.to(self.device) for x in batch]
                logits = self.model(seqs, genre_matrices)  # [B, G, I]
                top5_items = torch.topk(logits, 5, dim=2).indices  # [B, G, 5]

                for i in range(seqs.size(0)):
                    user_id = i  # Replace with actual user ID if available
                    recommendations[user_id] = {}
                    for g in range(logits.size(1)):
                        recommendations[user_id][f'Genre_{g}'] = top5_items[i, g].cpu().tolist()

        import json
        with open('genre_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        print("✅ Genre-based recommendations saved to genre_recommendations.json")
