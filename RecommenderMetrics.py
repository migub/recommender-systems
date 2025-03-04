# -*- coding: utf-8 -*-
"""
Metrics for evaluating Music Recommendation
"""
import itertools
from surprise import accuracy
from collections import defaultdict
import numpy as np

class RecommenderMetrics:

    @staticmethod
    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    @staticmethod
    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    @staticmethod
    def GetTopN(predictions, n=10, minimumRating=0.5):
        """Return the top-N recommendations for each user from a set of predictions.
        
        Args:
            predictions: The list of predictions, as returned by the test method of an algorithm.
            n: The number of recommendation to output for each user. Default is 10.
            minimumRating: The minimum rating to consider as a valid recommendation. Default is 0.5.
            
        Returns:
            A dict where keys are user (raw) ids and values are lists of tuples:
                [(raw item id, rating estimation), ...] of size n.
        """
        # First map the predictions to each user.
        topN = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            if est >= minimumRating:
                topN[str(uid)].append((str(iid), est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in topN.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            topN[uid] = user_ratings[:n]

        return topN

    @staticmethod
    def HitRate(topNPredicted, leftOutPredictions):
        """Compute the hit rate between predictions and left-out ratings.
        
        Args:
            topNPredicted: The predictions as returned by GetTopN
            leftOutPredictions: The left out ratings to test against
            
        Returns:
            The hit rate as a float between 0 and 1
        """
        hits = 0
        total = 0

        # For each left-out rating
        for uid, iid, true_r, est, _ in leftOutPredictions:
            total += 1
            # Is it in the predicted top N for this user?
            hit = False
            if str(uid) in topNPredicted:
                for (pred_iid, _) in topNPredicted[str(uid)]:
                    if str(iid) == pred_iid:
                        hit = True
                        break
            if hit:
                hits += 1

        # Protect against divide-by-zero
        if total == 0:
            return 0.0
            
        return hits / total

    @staticmethod
    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0.5):
        """Compute the cumulative hit rate for the given predictions.
        
        Args:
            topNPredicted: The predictions as returned by GetTopN
            leftOutPredictions: The left out ratings to test against
            ratingCutoff: The rating cutoff to consider. Default is 0.5.
            
        Returns:
            The cumulative hit rate as a float between 0 and 1
        """
        hits = 0
        total = 0

        # For each left-out rating
        for uid, iid, true_r, est, _ in leftOutPredictions:
            if true_r >= ratingCutoff:
                total += 1
                hit = False
                if str(uid) in topNPredicted:
                    for (pred_iid, _) in topNPredicted[str(uid)]:
                        if str(iid) == pred_iid:
                            hit = True
                            break
                if hit:
                    hits += 1

        # Protect against divide-by-zero
        if total == 0:
            return 0.0
            
        return hits / total

    @staticmethod
    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        """Compute the average reciprocal hit rank.
        
        Args:
            topNPredicted: The predictions as returned by GetTopN
            leftOutPredictions: The left out ratings to test against
            
        Returns:
            The average reciprocal hit rank as a float
        """
        summation = 0
        total = 0
        # For each left-out rating
        for uid, iid, true_r, est, _ in leftOutPredictions:
            total += 1
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            if str(uid) in topNPredicted:
                for (pred_iid, _) in topNPredicted[str(uid)]:
                    rank += 1
                    if str(iid) == pred_iid:
                        hitRank = rank
                        break
            if hitRank > 0:
                summation += 1.0 / hitRank

        # Protect against divide-by-zero
        if total == 0:
            return 0.0
            
        return summation / total

    @staticmethod
    def UserCoverage(topNPredicted, numUsers):
        """Compute the user coverage for the given predictions.
        
        Args:
            topNPredicted: The predictions as returned by GetTopN
            numUsers: The total number of users to compute coverage against
            
        Returns:
            The user coverage as a float between 0 and 1
        """
        if numUsers == 0:
            return 0.0
        return len(topNPredicted) / numUsers

    @staticmethod
    def Diversity(topNPredicted, simsAlgo):
        """Compute the diversity of recommendations.
        
        Args:
            topNPredicted: The predictions as returned by GetTopN
            simsAlgo: The similarity algorithm to use
            
        Returns:
            The diversity score as a float between 0 and 1
        """
        if simsAlgo is None:
            return 0.0  # Return 0 if no similarities are available
            
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for uid in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[uid], 2)
            for pair in pairs:
                item1 = pair[0][0]
                item2 = pair[1][0]
                inner_id1 = simsAlgo.trainset.to_inner_iid(item1)
                inner_id2 = simsAlgo.trainset.to_inner_iid(item2)
                similarity = simsMatrix[inner_id1][inner_id2]
                total += similarity
                n += 1

        if n == 0:
            return 0.0
            
        s = total / n
        return 1-s

    @staticmethod
    def Novelty(topNPredicted, rankings):
        """Compute the novelty of recommendations.
        
        Args:
            topNPredicted: The predictions as returned by GetTopN
            rankings: The popularity rankings to use
            
        Returns:
            The novelty score as a float
        """
        n = 0
        total = 0
        for uid in topNPredicted.keys():
            for iid, _ in topNPredicted[uid]:
                rank = rankings.get(iid, 0)
                total += rank
                n += 1
        
        if n == 0:
            return 0.0
            
        return total / n 