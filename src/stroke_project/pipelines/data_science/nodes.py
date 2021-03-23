import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostClassifier
from sklearn.metrics import f1_score, recall_score



def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """
    Splits data into train and test splits.
    Args:
        data: DataFrame containing the features and target to split
        parameters: Dictionary of parameters to split the data using sklearn.train_test_split
    Returns:
        Tuple containing the splited data.
    """

    X = data[parameters['features']]
    y = data.stroke

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = parameters['test_size'],
        random_state = parameters['random_state'],
        stratify=y
    )

    return X_train, X_test, y_train, y_test



def over_sample_data(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict) -> Tuple:
    """
    Over sample data using SMOTE.
    Args:
        X_train, y_train: data for over sampling
    Return:
        Tuple containing the oversampled train data.
    """

    sm = SMOTE(random_state=parameters['random_state'])
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    return X_train_res, y_train_res


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostClassifier:
    """
    Trains a LGBMClassifier with default configurations.
    Args:
        X_train, y_train: training data
    Return:
        LGBMClassifier object with the trained weights.
    """

    gradient_boost = GradientBoostClassifier()
    gradient_boost.fit(X_train, y_train)
    return gradient_boost


def evaluate_model(classifier: GradientBoostClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Calculates and logs the f1 score and recall.
    Args:
        classifier: Trained LGBMClassifier
        X_test, y_test: Test_data for evaluation
    """

    y_pred = classifier.predict(X_test)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a f1_score of %.3f and a recall of %3f.", f1, recall)