# -*- coding: utf-8 -*-
"""
Implicit feedback recommender for music listening data
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
from MusicDataNew import MusicData
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

class ImplicitMF(AlgoBase):
    """
    Implicit Matrix Factorization recommender for music listening data.
    
    This algorithm is designed for implicit feedback (listening events)
    rather than explicit ratings.
    """
    
    def __init__(self, n_factors=20, n_epochs=20, reg=0.1, lr=0.01, alpha=40):
        AlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.lr = lr
        self.alpha = alpha  # Controls the confidence level
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        # Initialize user and item factors
        n_users = trainset.n_users
        n_items = trainset.n_items
        
        # Initialize user and item factors with small random values
        self.user_factors = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.01, (n_items, self.n_factors))
        
        # Create sparse matrix for efficient computation
        rows, cols, data = [], [], []
        for u, i, r in trainset.all_ratings():
            rows.append(u)
            cols.append(i)
            data.append(r)
        
        self.R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        # Optimize using SGD with numerical stability improvements
        for epoch in range(self.n_epochs):
            # Process each rating
            for u, i, r in trainset.all_ratings():
                # For implicit feedback, all ratings are "positive"
                # but we apply a confidence level to them
                confidence = 1 + self.alpha * r
                
                # Compute current prediction
                pred = np.dot(self.user_factors[u], self.item_factors[i])
                pred = np.clip(pred, -10, 10)  # Prevent numerical overflow
                
                # Compute error with gradient clipping
                error = confidence * (r - pred)
                error = np.clip(error, -1, 1)  # Prevent extreme gradients
                
                # Update user and item factors with regularization
                user_factor = self.user_factors[u]
                item_factor = self.item_factors[i]
                
                # SGD update with smaller learning rate
                user_update = self.lr * (error * item_factor - self.reg * user_factor)
                item_update = self.lr * (error * user_factor - self.reg * item_factor)
                
                # Clip updates to prevent overflow
                user_update = np.clip(user_update, -1, 1)
                item_update = np.clip(item_update, -1, 1)
                
                self.user_factors[u] += user_update
                self.item_factors[i] += item_update
            
            # Normalize factors after each epoch
            self.user_factors = normalize(self.user_factors)
            self.item_factors = normalize(self.item_factors)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Completed epoch {epoch + 1}/{self.n_epochs}")
                
        return self
    
    def estimate(self, u, i):
        # Check that the user and item exist
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User or item is unknown.')
            
        # Compute dot product of user and item factors
        pred = np.dot(self.user_factors[u], self.item_factors[i])
        
        # Apply sigmoid to map prediction to [0,1]
        pred = 1 / (1 + np.exp(-pred))
        
        # Ensure prediction is between 0 and 1
        return float(np.clip(pred, 0, 1))
        
    def compute_similarities(self):
        """Return the similarity matrix for diversity calculation."""
        if not hasattr(self, 'item_factors'):
            return np.zeros((self.trainset.n_items, self.trainset.n_items))
        # Compute normalized item-item similarities
        sim = np.dot(self.item_factors, self.item_factors.T)
        # Normalize similarities to [-1, 1] range
        sim = sim / np.maximum(np.abs(sim).max(), 1e-8)
        return sim

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
        
        # Add implicit recommender with tuned parameters
        implicit_rec = ImplicitMF(n_factors=50, n_epochs=30, reg=0.01, lr=0.005, alpha=20)
        evaluator.AddAlgorithm(implicit_rec, "ImplicitMF")
        
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