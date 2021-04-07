from scipy.io import arff
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

def drop_row_column_if_percent(df, percent):
    percent = percent / 100
    count_by_rows = df.isna().mean(axis=0)
    result = df[count_by_rows[count_by_rows < percent].index.values]
    count_by_columns = result.isna().mean(axis=1)
    result = result[count_by_columns < percent]
    return result

class Preprocessing:
    def __init__(self, dataset_name, path):
        self.dataset_name = dataset_name
        self.path=path

    def get_dataset(self, flag='csv'):
        return self.execute(flag=flag)

    def load_arff(self):
        filename = "{}/{}.arff".format(
            self.path, self.dataset_name
        )
        data, meta = arff.loadarff(filename)
        return data, meta
    def load_csv(self):
        filename = "{}/{}.csv".format(
            self.path, self.dataset_name
        )
        return pd.read_csv(filename)
    def execute(self, flag='csv'):
        min_max_scaler = MinMaxScaler()
        if flag=='csv':
            df = self.load_csv()

            df.columns = df.columns.str.strip().str.lower()

            X = df.iloc[:, :-1]
            Y = df.iloc[:, -1]

            numerical_columns = X.select_dtypes(include="number").columns
            categorical_columns = X.select_dtypes(exclude="number").columns
            if not numerical_columns.empty:
                for column in numerical_columns:
                    X[column] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(X[column])), columns=[column])
                X[numerical_columns] = X[numerical_columns].fillna(-1)

            if not categorical_columns.empty:
                X[categorical_columns] = X[categorical_columns].fillna('?')
                '''for column in categorical_columns:
                    X[column] = pd.DataFrame(X[column].str.decode("utf-8"), columns=[column])

            if pd.api.types.is_string_dtype(Y):
                Y = Y.str.decode("utf-8")'''


        elif flag=='arff':
            data, meta = self.load_arff()

            df = pd.DataFrame(data)

            # convert all columns name to lowercase
            df.columns = df.columns.str.strip().str.lower()

            # split dataset and class
            X = df.iloc[:, :-1]
            Y = df.iloc[:, -1]

            # get columns by types
            numerical_columns = X.select_dtypes(include="number").columns
            categorical_columns = X.select_dtypes(exclude="number").columns
            if not numerical_columns.empty:
                for column in numerical_columns:
                    X[column] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(X[column])),columns=[column])
                X[numerical_columns] = X[numerical_columns].fillna(-1)


            if not categorical_columns.empty:
                X[categorical_columns] = X[categorical_columns].fillna('?')
                for column in categorical_columns:
                    X[column] = pd.DataFrame(X[column].str.decode("utf-8"), columns=[column])
            if pd.api.types.is_string_dtype(Y):
                Y = Y.str.decode("utf-8")

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

        return X_train, y_train, X_val, y_val, X_test, y_test


