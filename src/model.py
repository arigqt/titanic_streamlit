"""
All functins refering to the model we use for this app.

"""
from catboost import CatBoostClassifier
import pandas as pd


def train_and_predict_same_dataset(x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Train and predict on the same dataset for a foo app.

    The aim of this function is to train with train data, and return predictions from
    the trained model on train data too. The model is CatBoostClassifier using boosting
    trees algorithms.
    The output is a concatenated dataframe with x_train, y_train and lasty the predictions of 
    the model.

    """
    model = CatBoostClassifier(verbose=False, cat_features=["Sex", "Embarked"])
    model.fit(x_train, y_train)
    y_train_predict = model.predict(x_train)

    y_train_predict = pd.DataFrame(y_train_predict, columns=[
        "Predictions"]).astype(int)
    y_train = pd.DataFrame(y_train, columns=["Survived"]).astype(int)

    return pd.concat([x_train, y_train, y_train_predict], axis=1)
