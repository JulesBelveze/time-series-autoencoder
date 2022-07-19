from enum import Enum
from typing import List

import pandas as pd
import pkg_resources
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader


class Tasks(Enum):
    prediction = "prediction"
    reconstruction = "reconstruction"


class TimeSeriesDataset(object):
    def __init__(self, task: Tasks, data_path: str, categorical_cols: List[str], index_col: str, target_col: str,
                 seq_length: int, batch_size: int, prediction_window: int = 1):
        """
        :param task: name of the task
        :param data_path: path to datafile
        :param categorical_cols: name of the categorical columns, if None pass empty list
        :param index_col: column to use as index
        :param target_col: name of the targeted column
        :param seq_length: window length to use
        :param batch_size:
        :param prediction_window: window length to predict
        """
        self.task = task.value

        data_path = pkg_resources.resource_filename("tsa", data_path)
        self.data = pd.read_csv(data_path, index_col=index_col)
        self.categorical_cols = categorical_cols
        self.numerical_cols = list(set(self.data.columns) - set(categorical_cols) - set(target_col))
        self.target_col = target_col

        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.batch_size = batch_size

        transformations = [("scaler", StandardScaler(), self.numerical_cols)]
        if len(self.categorical_cols) > 0:
            transformations.append(("encoder", OneHotEncoder(), self.categorical_cols))
        self.preprocessor = ColumnTransformer(transformations, remainder="passthrough")

        if self.task == "prediction":
            self.y_scaler = StandardScaler()

    def preprocess_data(self):
        """Preprocessing function"""
        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        if self.task == "prediction":
            y_train = self.y_scaler.fit_transform(y_train)
            y_test = self.y_scaler.transform(y_test)
            return X_train, X_test, y_train, y_test
        return X_train, X_test, None, None

    def frame_series(self, X, y=None):
        """
        Function used to prepare the data for time series prediction
        :param X: set of features
        :param y: targeted value to predict
        :return: TensorDataset
        """
        nb_obs, nb_features = X.shape
        features, target, y_hist = [], [], []

        for i in range(1, nb_obs - self.seq_length - self.prediction_window):
            features.append(torch.FloatTensor(X[i:i + self.seq_length, :]).unsqueeze(0))

            if self.task == "prediction":
                # lagged output used for prediction
                y_hist.append(torch.FloatTensor(y[i - 1:i + self.seq_length - 1]).unsqueeze(0))
                # shifted target
                target.append(torch.FloatTensor(y[i + self.seq_length:i + self.seq_length + self.prediction_window]))
            else:
                y_hist.append(torch.FloatTensor(X[i - 1: i + self.seq_length - 1, :]).unsqueeze(0))
                target.append(
                    torch.FloatTensor(X[i + self.seq_length:i + self.seq_length + self.prediction_window, :]))

        features_var = torch.cat(features)
        y_hist_var = torch.cat(y_hist)
        target_var = torch.cat(target)

        return TensorDataset(features_var, y_hist_var, target_var)

    def get_loaders(self):
        """
        Preprocess and frame the dataset

        :return: DataLoaders associated to training and testing data
        """
        X_train, X_test, y_train, y_test = self.preprocess_data()
        nb_features = X_train.shape[1]

        train_dataset = self.frame_series(X_train, y_train)
        test_dataset = self.frame_series(X_test, y_test)

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_iter = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return train_iter, test_iter, nb_features

    def invert_scale(self, predictions):
        """
        Inverts the scale of the predictions
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        if self.task == "prediction":
            unscaled = self.y_scaler.inverse_transform(predictions)
        else:
            unscaled = self.preprocessor.named_transformers_["scaler"].inverse_transform(predictions)
        return torch.Tensor(unscaled)
