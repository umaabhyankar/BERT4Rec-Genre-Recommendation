from .base import AbstractDataset
import pandas as pd
import os
from datetime import datetime
import numpy as np
import torch

class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['ratings.dat', 'movies.dat']

    def __init__(self, args):
        super().__init__(args)
        self._load_dataset()
        self._set_num_items()
        self._split_dataset()

    def _load_dataset(self):
        df = self.load_ratings_df()
        movies_df = self.load_movies_df()

        # Create genre mapping
        all_genres = set()
        for genres in movies_df['genres']:
            all_genres.update(genres.split('|'))

        self.genre2id = {genre: idx for idx, genre in enumerate(all_genres)}
        self.id2genre = {idx: genre for genre, idx in self.genre2id.items()}

        # Create movie-genre mapping
        self.movie_genre_map = {}
        for _, row in movies_df.iterrows():
            movie_id = int(row['sid'])
            genres = row['genres'].split('|')
            self.movie_genre_map[movie_id] = [self.genre2id[genre] for genre in genres]

        # Add genre information to dataset
        df['genres'] = df['sid'].apply(lambda x: self.movie_genre_map.get(x, []))

        self.df = df

    def _set_num_items(self):
        """Set the number of items in the dataset"""
        self._num_items = len(self.movie_genre_map)
        print(f"Number of items: {self._num_items}")  # Debug print

    def _split_dataset(self):
        """Split the dataset into train, val, and test sets"""
        df = self.df.copy()

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Group by user
        user_groups = df.groupby('uid')

        # Split each user's history into train, val, and test
        train_data = {}
        val_data = {}
        test_data = {}

        for user_id, group in user_groups:
            items = group['sid'].tolist()
            if len(items) >= 3:  # Need at least 3 interactions for train/val/test
                train_data[user_id] = items[:-2]  # All but last 2 items
                val_data[user_id] = items[-2:-1]  # Second to last item
                test_data[user_id] = items[-1:]   # Last item
            else:
                # If user has less than 3 interactions, skip them
                continue

        self.train = train_data
        self.val = val_data
        self.test = test_data

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        return pd.read_csv(os.path.join(folder_path, 'ratings.dat'),
                          sep='::',
                          header=None,
                          names=['uid', 'sid', 'rating', 'timestamp'],
                          engine='python',
                          encoding='ISO-8859-1')

    def load_movies_df(self):
        folder_path = self._get_rawdata_folder_path()
        return pd.read_csv(os.path.join(folder_path, 'movies.dat'),
                          sep='::',
                          header=None,
                          names=['sid', 'title', 'genres'],
                          engine='python',
                          encoding='ISO-8859-1')

    def get_genre_matrix(self, item_ids):
        """Get genre matrix for given item IDs"""
        genre_matrix = torch.zeros(len(item_ids), len(self.genre2id))
        for i, item_id in enumerate(item_ids):
            genres = self.movie_genre_map.get(item_id, [])
            for genre_id in genres:
                genre_matrix[i, genre_id] = 1
        return genre_matrix

    def get_num_items(self):
        return len(self.movie_genre_map)

    @property
    def num_items(self):
        return self.get_num_items()
