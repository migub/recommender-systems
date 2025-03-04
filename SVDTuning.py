# -*- coding: utf-8 -*-
"""
SVD Parameter Tuning for Music Recommendation with Implicit Feedback
"""

from MusicDataNew import MusicData
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV
from ImplicitRecommender import ImplicitMF

import random
import numpy as np

def LoadMusicData():
    md = MusicData()
    print("Loading music listening data...")
    data = md.loadMusicData()
    if data is None:
        print("Failed to load data. Exiting.")
        exit(1)
    print("\nComputing media popularity ranks so we can measure novelty later...")
    rankings = md.getPopularityRanks()
    return (md, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(md, evaluationData, rankings) = LoadMusicData()

print("Searching for best parameters...")
param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
              'n_factors': [50, 100]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(evaluationData)

# best RMSE score
print("Best RMSE score attained: ", gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# Construct an Evaluator
evaluator = Evaluator(evaluationData, rankings)

# Add the tuned SVD
params = gs.best_params['rmse']
SVDtuned = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])
evaluator.AddAlgorithm(SVDtuned, "SVD - Tuned")

# Add regular SVD
SVDUntuned = SVD()
evaluator.AddAlgorithm(SVDUntuned, "SVD - Untuned")

# Add Implicit MF algorithm
ImplicitAlgo = ImplicitMF()
evaluator.AddAlgorithm(ImplicitAlgo, "Implicit MF")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Evaluate algorithms
evaluator.Evaluate(True)  # Set to True to evaluate top-N recommendations

# Sample recommendations for a user
evaluator.SampleTopNRecs(md) 