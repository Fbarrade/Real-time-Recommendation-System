import pandas as pd

from abc import ABC, abstractmethod

from sklearn.preprocessing import StandardScaler, MinMaxScaler


__all__ = ["YelpUserGNNDataset", "YelpDestinationsGNNDataset", "YelpReviewsGNNDataset"]


class BaseDataset(ABC):

    def __init__(self):
        super().__init__()

        self.df: pd.DataFrame = None 

    @abstractmethod
    def process(self):
        ...

    def read_csv(self, path: str):
        self.df = pd.read_csv(path)

    def delete_unused_columns(self, df: pd.DataFrame, delete_cols: list[str]):
        return df.drop(columns=delete_cols)

    def encode_categorical(
        self, df: pd.DataFrame, exclude_columns: list[str], normalize: bool = True
    ):
        categorical_columns = df.select_dtypes(include=['object']).columns

        categorical_columns = [col for col in categorical_columns if col not in exclude_columns]

        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        bool_columns = df_encoded.select_dtypes(include=[bool]).columns
        df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)

        if normalize:
            numeric_columns = df_encoded.select_dtypes(include=['number']).columns
            scaler = StandardScaler()  # You can also use MinMaxScaler()
            df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])

        return df_encoded



class YelpUserGNNDataset(BaseDataset):

    def __init__(self):
        super(YelpUserGNNDataset, self).__init__()

    def process(self):
        assert self.df is not None, "self.df must be not None, call self.read_csv first."
        
        self.df = self.delete_unused_columns(
            self.df, [ 'name', "friends"  , 'elite' ,"yelping_since" ]
        )

        self.df = self.encode_categorical(
            df=self.df, exclude_columns=['user_id', 'friends']
        )


class YelpDestinationsGNNDataset(BaseDataset):

    def __init__(self):
        super(YelpDestinationsGNNDataset, self).__init__()

    def one_hot_encode_categories(self, df: pd.DataFrame, column_name: str):
        all_categories = df[column_name].str.split(',').explode().str.strip()
        unique_categories = all_categories.unique()

        encoded_columns = {}

        for category in unique_categories:
            encoded_columns[category] = df[column_name].apply(lambda x: 1 if category in x else 0)

        df = pd.concat([df, pd.DataFrame(encoded_columns)], axis=1)
        return df


    def process(self):
        assert self.df is not None, "self.df must be not None, call self.read_csv first."

        self.df = self.one_hot_encode_categories(self.df, "categories")

        self.df = self.delete_unused_columns(self.df, [
            'name', 'address','city','state',"postal_code", "hours","categories","attributes" ,"latitude","longitude"
        ])

        self.df = self.encode_categorical(
            df= self.df, exclude_columns= ['business_id']
        )


class YelpReviewsGNNDataset(BaseDataset):

    def __init__(self):
        super(YelpReviewsGNNDataset, self).__init__()

    def normalize(self, df: pd.DataFrame):
        scaler = MinMaxScaler()
        df["stars"] = scaler.fit_transform(df['stars'].values.reshape(-1, 1))

        return df

    def process(self):
        assert self.df is not None, "self.df must be not None, call self.read_csv first."

        self.df = self.normalize(df= self.df)

        

        
