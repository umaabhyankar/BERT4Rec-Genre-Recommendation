import torch
import torch.nn as nn
from .bert import BERT

class BERTGenreModel(nn.Module):
    @classmethod
    def code(cls):
        return 'bert_genre'

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = BERT(args)
        self.num_items = args.num_items
        self.genre_embedding_size = args.genre_embedding_size
        self.num_genres = args.num_genres
        
        # Genre embeddings
        self.genre_embeddings = nn.Embedding(self.num_genres, self.genre_embedding_size)
        
        # Output layer for genre-item predictions
        self.output_layer = nn.Linear(args.bert_hidden_units + self.genre_embedding_size, self.num_items)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights for BERT and output layer"""
        self.bert.init_weights()
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, genre_matrix):
        batch_size = x.size(0)

        # BERT output: [batch_size, seq_len, hidden_dim]
        bert_output = self.bert(x)

        # Get final token embedding (CLS token)
        bert_output = bert_output[:, -1, :]  # [batch_size, hidden_dim]

        # Repeat genre embeddings: [batch_size, num_genres, genre_emb_dim]
        genre_embeddings = self.genre_embeddings.weight  # [num_genres, genre_emb_dim]
        genre_embeddings_expanded = genre_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Expand BERT output to match genre dimension: [batch_size, num_genres, hidden_dim]
        bert_output_expanded = bert_output.unsqueeze(1).expand(-1, self.num_genres, -1)

        # Concatenate: [batch_size, num_genres, hidden + genre_emb_dim]
        combined = torch.cat([bert_output_expanded, genre_embeddings_expanded], dim=-1)

        # Final output layer: [batch_size, num_genres, num_items]
        logits = self.output_layer(combined)

        return logits


    def get_topk_per_genre(self, x, genre_matrix, k=5):
        """Get top-k items for each genre"""
        logits = self.forward(x, genre_matrix)
        topk_per_genre = torch.topk(logits, k, dim=2).indices  # Shape: [batch_size, num_genres, k]
        return topk_per_genre
