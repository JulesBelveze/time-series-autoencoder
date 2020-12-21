import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader


class TimeSeriesDataset(object):
    def __init__(self, data, categorical_cols, target_col, seq_length, prediction_window=1):
        '''
        :param data: dataset of type pandas.DataFrame
        :param categorical_cols: name of the categorical columns, if None pass empty list
        :param target_col: name of the targeted column
        :param seq_length: window length to use
        :param prediction_window: window length to predict
        '''
        self.data = data
        self.categorical_cols = categorical_cols
        self.numerical_cols = list(set(data.columns) - set(categorical_cols) - set(target_col))
        self.target_col = target_col
        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.preprocessor = None

    def preprocess_data(self):
        '''Preprocessing function'''
        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]

        self.preprocess = ColumnTransformer(
            [("scaler", StandardScaler(), self.numerical_cols),
             ("encoder", OneHotEncoder(), self.categorical_cols)],
            remainder="passthrough"
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        if self.target_col:
            return X_train, X_test, y_train.values, y_test.values
        return X_train, X_test

    def frame_series(self, X, y=None):
        '''
        Function used to prepare the data for time series prediction
        :param X: set of features
        :param y: targeted value to predict
        :return: TensorDataset
        '''
        nb_obs, nb_features = X.shape
        features, target, y_hist = [], [], []

        for i in range(1, nb_obs - self.seq_length - self.prediction_window):
            features.append(torch.FloatTensor(X[i:i + self.seq_length, :]).unsqueeze(0))

        features_var = torch.cat(features)

        if y is not None:
            for i in range(1, nb_obs - self.seq_length - self.prediction_window):
                target.append(
                    torch.tensor(y[i + self.seq_length:i + self.seq_length + self.prediction_window]))
                # lagged output used for prediction
                y_hist.append(
                    torch.tensor(y[i + self.seq_length - 1:i + self.seq_length + self.prediction_window - 1]))
            target_var, y_hist_var = torch.cat(target), torch.cat(y_hist)
            return TensorDataset(features_var, y_hist_var, target_var)
        return TensorDataset(features_var)

    def get_loaders(self, batch_size: int):
        '''
        Preprocess and frame the dataset
        :param batch_size: batch size
        :return: DataLoaders associated to training and testing data
        '''
        X_train, X_test, y_train, y_test = self.preprocess_data()
        nb_features = X_train.shape[1]

        train_dataset = self.frame_series(X_train, y_train)
        test_dataset = self.frame_series(X_test, y_test)

        train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        return train_iter, test_iter, nb_features
