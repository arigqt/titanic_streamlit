import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from model import train_and_predict_same_dataset


train = pd.read_csv('../data/titanic_train.csv')

train.drop(columns=["PassengerId", "Ticket", "Name", "Cabin"], inplace=True)

y_train = train["Survived"]
x_train = train.drop(columns=["Survived"])

x_train.Embarked.fillna(
    x_train["Embarked"].value_counts().index[0], inplace=True)

x_train.fillna(x_train.mean(), inplace=True)

df_with_results = train_and_predict_same_dataset(x_train, y_train)
df_with_results.to_csv("../data/predictions/train_with_predictions.csv")
