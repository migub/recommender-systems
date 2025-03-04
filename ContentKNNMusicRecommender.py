# -*- coding: utf-8 -*-
"""
Content-based KNN for Music Recommendation
"""

from MusicDataNew import MusicData
from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class ContentKNNMusicRecommender(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k
        # Feature weights
        self.weights = {
            'genre': 2.0,    # Genre is most important
            'artist': 1.5,   # Artist is second most important
            'year': 0.5,     # Year has less importance
            'duration': 0.3  # Duration has least importance
        }

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        # Load music features
        md = MusicData()
        
        # Get all available features
        genres = md.getGenres()
        artists = md.getArtists()
        years = md.getYears()
        durations = md.getDurations()
        
        # Get all unique artists for one-hot encoding
        unique_artists = sorted(set(artists.values()))
        artist_to_idx = {artist: idx for idx, artist in enumerate(unique_artists)}
        
        # Combine all features
        self.track_features = {}
        all_track_ids = set(list(genres.keys()) + list(artists.keys()) + list(years.keys()) + list(durations.keys()))
        
        for track_id in all_track_ids:
            features = []
            
            # Add genre features with weight
            genre_vec = genres.get(track_id, [0] * len(next(iter(genres.values()))) if genres else [])
            genre_vec = [x * self.weights['genre'] for x in genre_vec]
            features.extend(genre_vec)
            
            # Add artist features with weight
            artist = artists.get(track_id, '')
            artist_vec = [0] * len(unique_artists)
            if artist in artist_to_idx:
                artist_vec[artist_to_idx[artist]] = 1 * self.weights['artist']
            features.extend(artist_vec)
            
            # Add year feature with weight
            year = years.get(track_id, 0)
            features.append(float(year) * self.weights['year'])
            
            # Add duration feature with weight
            duration = durations.get(track_id, 0.0)
            features.append(float(duration) * self.weights['duration'])
            
            if len(features) > 0:
                self.track_features[str(track_id)] = np.array(features)
        
        # Normalize numerical features (year and duration)
        if self.track_features:
            # Get the indices for year and duration
            year_idx = len(genre_vec) + len(unique_artists)
            duration_idx = year_idx + 1
            
            # Extract year and duration values
            years = np.array([[features[year_idx]] for features in self.track_features.values()])
            durations = np.array([[features[duration_idx]] for features in self.track_features.values()])
            
            # Normalize
            year_scaler = MinMaxScaler()
            duration_scaler = MinMaxScaler()
            
            if len(years) > 0:
                normalized_years = year_scaler.fit_transform(years)
                normalized_durations = duration_scaler.fit_transform(durations)
                
                # Update the features with normalized values
                for i, track_id in enumerate(self.track_features.keys()):
                    self.track_features[track_id][year_idx] = normalized_years[i][0]
                    self.track_features[track_id][duration_idx] = normalized_durations[i][0]
        
        # Compute item similarity matrix
        self.compute_similarities()
                
        return self
    
    def compute_similarities(self):
        """Return the similarity matrix for diversity calculation."""
        # Create a list of inner track IDs and corresponding feature vectors
        inner_ids = []
        feature_vectors = []
        
        # Map from raw track ID to inner ID
        id_map = {}
        
        for raw_track_id in self.track_features:
            try:
                inner_id = self.trainset.to_inner_iid(str(raw_track_id))
                id_map[raw_track_id] = inner_id
                inner_ids.append(inner_id)
                feature_vectors.append(self.track_features[raw_track_id])
            except:
                # Track not in the training set
                pass
        
        # Convert to numpy array for calculation
        feature_matrix = np.array(feature_vectors)
        
        # Normalize feature vectors
        norms = np.linalg.norm(feature_matrix, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_matrix = feature_matrix / norms[:, np.newaxis]
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(normalized_matrix)
        
        # Create full-size similarity matrix
        n_items = self.trainset.n_items
        full_matrix = np.zeros((n_items, n_items))
        
        # Fill in the similarities we computed
        for i, inner_id1 in enumerate(inner_ids):
            for j, inner_id2 in enumerate(inner_ids):
                full_matrix[inner_id1, inner_id2] = similarity_matrix[i, j]
        
        # Store the similarity matrix
        self.sim = {}
        for i in range(n_items):
            self.sim[i] = {}
            for j in range(n_items):
                if full_matrix[i, j] > 0:  # Only store non-zero similarities
                    self.sim[i][j] = full_matrix[i, j]
        
        return full_matrix
    
    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User or item is unknown.')
        
        # Get top k similar items the user has rated
        neighbors = []
        for j in self.trainset.ur[u]:
            item_id = j[0]  # Item ID
            rating = j[1]   # Rating
            if i == item_id:
                continue    # Don't include the item itself
            
            if item_id in self.sim.get(i, {}):
                similarity = self.sim[i][item_id]
                if not np.isnan(similarity):  # Skip NaN similarities
                    neighbors.append((similarity, rating))
        
        # Sort by similarity
        neighbors.sort(key=lambda x: x[0], reverse=True)
        
        # Limit to k neighbors
        neighbors = neighbors[:self.k]
        
        if len(neighbors) == 0:
            raise PredictionImpossible('No neighbors found.')
        
        # Weighted average of ratings
        weighted_sum = sum([sim * rating for sim, rating in neighbors])
        sim_sum = sum([sim for sim, _ in neighbors])
        
        if sim_sum == 0:
            raise PredictionImpossible('Sum of similarities is zero.')
        
        prediction = weighted_sum / sim_sum
        
        # Ensure prediction is between 0 and 1 for implicit feedback
        return float(np.clip(prediction, 0, 1))

def main():
    try:
        # Load data
        print("Loading music listening data...")
        md = MusicData()
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
        
        # Add content KNN recommender
        content_knn = ContentKNNMusicRecommender()
        evaluator.AddAlgorithm(content_knn, "Content-KNN")
        
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