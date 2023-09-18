# Third Party Imports
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    return rf

# # Optional: implement hyperparameter tuning.
# def train_model(X_train, y_train):
#     """
#     Trains a machine learning model using RandomizedSearchCV for hyperparameter tuning
#     and returns the best model.

#     Parameters
#     ----------
#     X_train : np.array
#         Training data.
#     y_train : np.array
#         Labels.

#     Returns
#     -------
#     best_model : RandomForestClassifier
#         Trained machine learning model with best hyperparameters.
#     """

#     # Define the parameter grid
#     param_dist = {
#         'n_estimators': [50, 100, 150],
#         'max_depth': [None, 10, 20, 30],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'bootstrap': [True, False],
#     }

#     # Create the base model
#     rf = RandomForestClassifier(random_state=0)

#     # Instantiate the RandomizedSearchCV object
#     random_search = RandomizedSearchCV(
#         rf,
#         param_distributions=param_dist,
#         n_iter=10,
#         cv=3, 
#         verbose=1, 
#         random_state=0,
#         n_jobs=-1
#     )

#     # Perform hyperparameter search
#     random_search.fit(X_train, y_train)

#     # Get the best model
#     best_model = random_search.best_estimator_

#     return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pred = model.predict(X)
    return pred
