# -*- coding: utf-8 -*-
"""
Evaluator for Music Recommendation
"""
from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm
import numpy as np

class Evaluator:
    
    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed
        self.algorithms = []
        
    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)
        
    def Evaluate(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.GetName(), "...")
            try:
                metrics = algorithm.Evaluate(self.dataset, doTopN)
                results[algorithm.GetName()] = metrics
            except Exception as e:
                print(f"Error evaluating {algorithm.GetName()}: {str(e)}")
                results[algorithm.GetName()] = {
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "HR": np.nan,
                    "cHR": np.nan,
                    "ARHR": np.nan,
                    "Coverage": np.nan,
                    "Diversity": np.nan,
                    "Novelty": np.nan
                }

        # Print results
        print("\n")
        
        if (doTopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name, 
                        metrics.get("RMSE", np.nan),
                        metrics.get("MAE", np.nan),
                        metrics.get("HR", np.nan),
                        metrics.get("cHR", np.nan),
                        metrics.get("ARHR", np.nan),
                        metrics.get("Coverage", np.nan),
                        metrics.get("Diversity", np.nan),
                        metrics.get("Novelty", np.nan)
                ))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(
                    name,
                    metrics.get("RMSE", np.nan),
                    metrics.get("MAE", np.nan)
                ))
                
        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
        print("cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
        print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better.")
        print("Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
        print("Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
        print("           for a given user. Higher means more diverse.")
        print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")
    
    def SampleTopNRecs(self, ml, testSubject=85, k=10):
        try:
            for algo in self.algorithms:
                print("\nUsing recommender", algo.GetName())
                
                print("\nBuilding recommendation model...")
                trainSet = self.dataset.GetFullTrainSet()
                algo.GetAlgorithm().fit(trainSet)
                
                print("Computing recommendations...")
                testSet = self.dataset.GetAntiTestSetForUser(testSubject)
                
                predictions = algo.GetAlgorithm().test(testSet)
                
                recommendations = []
                for userID, mediaID, actualRating, estimatedRating, _ in predictions:
                    recommendations.append((mediaID, estimatedRating))
                
                recommendations.sort(key=lambda x: x[1], reverse=True)
                
                print("\nTop", k, "recommendations for user", testSubject)
                for ratings in recommendations[:k]:
                    print(ml.getTrackName(ratings[0]), ratings[1])
                    
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}") 