# -*- coding: utf-8 -*-
"""
Hybrid algorithm for Music Recommendation
"""

from MusicDataNew import MusicData
from surprise import AlgoBase, SVD
from surprise import KNNBasic
from ContentKNNMusicRecommender import ContentKNNMusicRecommender
from Evaluator import Evaluator
from ImplicitRecommender import ImplicitMF

import random
import numpy as np

class HybridRecommender(AlgoBase):
    def __init__(self, svd_weight=0.4, content_weight=0.4, implicit_weight=0.2):
        AlgoBase.__init__(self)
        self.svd_weight = svd_weight
        self.content_weight = content_weight
        self.implicit_weight = implicit_weight
        
        # Create the component recommenders
        self.svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        self.content_knn = ContentKNNMusicRecommender(k=40)
        self.implicit_mf = ImplicitMF(n_factors=50, n_epochs=30, reg=0.01, lr=0.005, alpha=20)
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        # Train each component
        self.svd.fit(trainset)
        self.content_knn.fit(trainset)
        self.implicit_mf.fit(trainset)
        
        return self
    
    def estimate(self, u, i):
        # Get predictions from each component
        try:
            svd_pred = self.svd.estimate(u, i)
        except:
            svd_pred = 0.5  # Default to middle value
            
        try:
            content_pred = self.content_knn.estimate(u, i)
        except:
            content_pred = 0.5  # Default to middle value
            
        try:
            implicit_pred = self.implicit_mf.estimate(u, i)
        except:
            implicit_pred = 0.5  # Default to middle value
        
        # Return weighted average
        prediction = (self.svd_weight * svd_pred + 
                     self.content_weight * content_pred + 
                     self.implicit_weight * implicit_pred)
        
        # Ensure prediction is between 0 and 1
        return float(np.clip(prediction, 0, 1))
        
    def compute_similarities(self):
        """Return the similarity matrix for diversity calculation."""
        if hasattr(self.content_knn, 'sim'):
            return self.content_knn.compute_similarities()
        return np.zeros((self.trainset.n_items, self.trainset.n_items))

def main():
    try:
        print("Loading music listening data...")
        md = MusicData()
        data = md.loadMusicData()
        
        if data is None:
            print("Failed to load data. Exiting.")
            return
        
        print("\nComputing popularity rankings...")
        rankings = md.getPopularityRanks()
        
        if not rankings:
            print("Failed to compute popularity rankings. Exiting.")
            return
        
        # Set up evaluator
        evaluator = Evaluator(data, rankings)
        
        # Add algorithms
        evaluator.AddAlgorithm(SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02), "SVD")
        evaluator.AddAlgorithm(ContentKNNMusicRecommender(k=40), "Content-KNN")
        evaluator.AddAlgorithm(ImplicitMF(n_factors=50, n_epochs=30, reg=0.01, lr=0.005, alpha=20), "ImplicitMF")
        evaluator.AddAlgorithm(HybridRecommender(), "Hybrid")
        
        # Evaluate
        evaluator.Evaluate(True)
        
        # Sample recommendations
        evaluator.SampleTopNRecs(md)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    main() 