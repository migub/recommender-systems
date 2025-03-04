# -*- coding: utf-8 -*-
"""
Content-based recommender for music
"""
from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pandas as pd

class ContentBasedRecommender(AlgoBase):

    def __init__(self, musicData, k=40):
        AlgoBase.__init__(self)
        self.k = k
        self.musicData = musicData
        self.sim = None
        self.trainset = None
        # Feature weights
        self.weights = {
            'genre': 2.0,    # Genre is most important
            'artist': 1.5,   # Artist is second most important
            'year': 0.5,     # Year has less importance
            'duration': 0.3  # Duration has least importance
        }
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.trainset = trainset
        
        # Compute item features
        print("Computing the cosine similarity matrix...")
        
        try:
            # Get all features
            genres = self.musicData.getGenres()
            years = self.musicData.getYears()
            artists = self.musicData.getArtists()
            durations = self.musicData.getDurations()
            
            # Get all unique media IDs from the training set
            all_media_ids = [trainset.to_raw_iid(iid) for iid in range(trainset.n_items)]
            
            # Create feature matrices
            genre_features = []
            year_features = []
            artist_features = []
            duration_features = []
            
            # Get unique artists for one-hot encoding
            unique_artists = sorted(set(artists.values()))
            artist_to_idx = {artist: idx for idx, artist in enumerate(unique_artists)}
            
            # Process each media item
            for media_id in all_media_ids:
                # Add genre features with weight
                genre_vec = genres.get(media_id, [0] * len(next(iter(genres.values()))) if genres else [])
                genre_vec = [x * self.weights['genre'] for x in genre_vec]
                genre_features.append(genre_vec)
                
                # Add year features with weight
                year = years.get(media_id, None)
                if year is not None:
                    year_features.append([float(year) * self.weights['year']])
                else:
                    year_features.append([0.0])
                
                # Add artist features with weight
                artist = artists.get(media_id, '')
                artist_vec = [0] * len(unique_artists)
                if artist in artist_to_idx:
                    artist_vec[artist_to_idx[artist]] = 1 * self.weights['artist']
                artist_features.append(artist_vec)
                
                # Add duration features with weight
                duration = durations.get(media_id, None)
                if duration is not None:
                    duration_features.append([float(duration) * self.weights['duration']])
                else:
                    duration_features.append([0.0])
            
            # Convert to numpy arrays
            genre_matrix = np.array(genre_features, dtype=np.float32)
            year_matrix = np.array(year_features, dtype=np.float32)
            artist_matrix = np.array(artist_features, dtype=np.float32)
            duration_matrix = np.array(duration_features, dtype=np.float32)
            
            # Normalize numerical features
            if len(year_features) > 0:
                year_scaler = MinMaxScaler()
                year_matrix = year_scaler.fit_transform(year_matrix)
            
            if len(duration_features) > 0:
                duration_scaler = MinMaxScaler()
                duration_matrix = duration_scaler.fit_transform(duration_matrix)
            
            # Combine all features
            feature_matrices = []
            
            if genre_matrix.size > 0:
                feature_matrices.append(genre_matrix)
            if year_matrix.size > 0:
                feature_matrices.append(year_matrix)
            if artist_matrix.size > 0:
                feature_matrices.append(artist_matrix)
            if duration_matrix.size > 0:
                feature_matrices.append(duration_matrix)
            
            if not feature_matrices:
                raise ValueError("No valid features available")
            
            # Combine features
            self.item_features = np.hstack(feature_matrices)
            
            # Normalize the combined feature matrix
            norms = np.linalg.norm(self.item_features, axis=1)
            norms[norms == 0] = 1  # Avoid division by zero
            self.item_features = self.item_features / norms[:, np.newaxis]
            
            # Compute similarities in batches
            batch_size = 1000
            n_items = len(all_media_ids)
            n_batches = (n_items + batch_size - 1) // batch_size
            self.sim = np.zeros((n_items, n_items))
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_items)
                batch_features = self.item_features[start_idx:end_idx]
                
                # Compute similarities for this batch
                batch_sim = cosine_similarity(batch_features, self.item_features)
                self.sim[start_idx:end_idx] = batch_sim
            
            # Convert similarities to sparse matrix
            self.sim = csr_matrix(self.sim)
            
            print(f"Feature matrix shape: {self.item_features.shape}")
            print(f"Number of items: {n_items}")
            print(f"Features per item: Genre={len(genre_vec)}, Artist={len(unique_artists)}, Year=1, Duration=1")
            
        except Exception as e:
            logging.error(f"Error during feature computation: {str(e)}")
            raise
        
    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        
        try:
            # Get similar items
            neighbors = []
            for rating in self.trainset.ur[u]:
                if rating[0] == i:
                    continue
                sim = float(self.sim[i, rating[0]])  # Convert to float to avoid numpy type issues
                if not np.isnan(sim) and sim > 0:  # Only consider positive similarities
                    neighbors.append((sim, rating[1]))
            
            # Sort by similarity
            neighbors.sort(key=lambda x: x[0], reverse=True)
            
            # Take top k neighbors
            k_neighbors = neighbors[:self.k]
            
            if not k_neighbors:
                # No similar items found, return middle value
                return 0.5
            
            # Compute weighted average
            sim_sum = sum(sim for sim, _ in k_neighbors)
            if sim_sum == 0:
                return 0.5
                
            weighted_sum = sum(sim * rating for sim, rating in k_neighbors)
            prediction = weighted_sum / sim_sum
            
            # Ensure prediction is between 0 and 1 for implicit feedback
            return max(0.0, min(1.0, prediction))
            
        except Exception as e:
            logging.error(f"Error during rating estimation: {str(e)}")
            raise PredictionImpossible('Error computing prediction.')
            
    def compute_similarities(self):
        """Return the similarity matrix for diversity calculation."""
        if self.sim is None:
            return np.zeros((self.trainset.n_items, self.trainset.n_items))
        return self.sim.toarray()

def main():
    try:
        # Load data
        from MusicDataNew import MusicData
        md = MusicData()
        print("Loading music listening data...")
        data = md.loadMusicData()
        
        if data is None:
            print("Failed to load data. Exiting.")
            return
        
        print("Computing popularity rankings...")
        rankings = md.getPopularityRanks()
        
        if not rankings:
            print("Failed to compute popularity rankings. Exiting.")
            return
        
        # Set up evaluator
        from Evaluator import Evaluator
        evaluator = Evaluator(data, rankings)
        
        # Add content-based recommender
        content_rec = ContentBasedRecommender(md)
        evaluator.AddAlgorithm(content_rec, "Content-Based")
        
        # Add SVD for comparison
        from surprise import SVD
        svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        evaluator.AddAlgorithm(svd, "SVD")
        
        # Evaluate
        evaluator.Evaluate(True)
        
        # Sample recommendations
        evaluator.SampleTopNRecs(md)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()