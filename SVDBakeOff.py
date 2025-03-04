# -*- coding: utf-8 -*-
"""
Comparison of SVD algorithms for Music Recommendation
"""

from MusicDataNew import MusicData
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadMusicData():
    md = MusicData()
    print("Loading music ratings...")
    data = md.loadMusicData()
    print("\nComputing track popularity ranks so we can measure novelty later...")
    rankings = md.getPopularityRanks()
    return (md, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(md, evaluationData, rankings) = LoadMusicData()

# Construct an Evaluator
evaluator = Evaluator(evaluationData, rankings)

# SVD
SVD_algo = SVD()
evaluator.AddAlgorithm(SVD_algo, "SVD")

# SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(md) 