import pandas as pd 

from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:
    """
    A simple custom class for greedy feature selecton.
    *Will need to be modified quite a bit for each dataset*
    """
    def evaluate_score(self, X, y):
        """
        This function evaluates model on data and returns 
        Area under ROC curve (AOC) 
        *We fit and evaluate AUC on same data - WE ARE OVERFITTING*
        :param X: training data
        :param y: targets
        :return: overfitted AUC score
        """
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc
    
    def _feature_selection(self, X, y):
        """
        This function does the actual greedy selection
        :param X: data, numpy array
        :param y: targets, numpy array
        :return: (best scores, best features)
        """
        good_features = []
        best_scores = []

        num_features = X.shape[1]

        # infinite loop
        while True:
            #initialise best feature and score of this loop
            this_feature = None 
            best_score = 0 

            # loop over all features
            for feature in range(num_features):
                # if feature is already in good features, skip
                if feature in good_features:
                    continue 
                # selected feature are all good features + current feature
                selected_features = good_features + [feature]
                # remove other features from data
                xtrain = X[:, selected_features]
                # calculate the score (e.g. AUC)
                score = self.evaluate_score(xtrain, y)
                # if score > best score of loop, change best score and features
                if score > best_score:
                    this_feature = feature
                    best_score = score

            # if we have selected a feature, add it to good feature list and update best scores list
            if this_feature != None:
                good_features.append(this_feature)
                best_scores.append(best_score)

            # if we didn't improve in previous round then exit the while loop 
            if len(best_scores) > 2:
                if best_scores[-1] < best_scores[-2]:
                    break 
            
        # return best scores and good features
        return best_scores[:-1], good_features[:-1]
        
    def __call__(self, X, y):
        """
        Call function will call the class on a set of arguments
        """
        # select features, return scores and selected features
        scores, features = self._feature_selection(X, y)
        # transform data with selected features
        return X[:, features], scores

if __name__ == "__main__":
    # generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=100)

    # transform data by greedy feature selection 
    X_transformed, scores = GreedyFeatureSelection()(X, y)
    print("Number of features left: ", X_transformed.shape[1])
    print("Scores: ", scores)