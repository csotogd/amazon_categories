import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle
import re
   
class Numerical_Preprocessor:
    """
    Class for performing numerical preprocessing on a DataFrame.

    Attributes:
    - scaler_handler (MinMaxScalerHandler): Handler for Min-Max scaler
    """
 

    def __init__(self, training):
        """
        Initialize Numerical_Preprocessor object.
        """
        self.training =training
        self.scaler = None
        self.path_scaler = 'minmax_scaler.pkl'

    def preprocess(self, df):
        """
        Perform numerical preprocessing on the input DataFrame.

        Parameters:
        - df (pd.DataFrame): Input DataFrame
        - training (bool): Flag indicating whether the function is called during training

        Returns:
        - torch.Tensor: Preprocessed data as a PyTorch tensor
        """
        df = self.filter_irrelevant_columns(df)
        df = self.replace_nan_with_empty_string(df)
        df = self.convert_price_to_float(df)
        df = self.mean_imputation_with_indicator(df)

        df = self.convert_str_lists_to_lists(df)
        df = self.get_nr_also_interacted_lists(df)
        df = self.drop_also_interacted_columns(df)

        np_array = self.scale_data(df)

        tensor = self.np_to_tensor(np_array)
        return tensor


    def filter_irrelevant_columns(self, df):
        """
        Filter the DataFrame to include only numerical columns of interest.

        Parameters:
        - df (pd.DataFrame): Input DataFrame

        Returns:
        - pd.DataFrame: DataFrame containing only relevant numerical columns
        """
        numerical_columns = ['price', 'also_view', 'also_buy']
        df = df[numerical_columns]
        return df

    def replace_nan_with_empty_string(self, df):
        """
        Replace NaN values in the 'price' column of a DataFrame with an empty string.

        Parameters:
        - df (pd.DataFrame): Input DataFrame

        Returns:
        - pd.DataFrame: DataFrame with NaN values replaced by empty strings in the 'price' column
        """
        df.loc[:, 'price'] = df['price'].fillna('')
        return df

    def convert_price_to_float(self, df, column_name='price'):
        """
        Convert the specified column in the DataFrame to float.

        Parameters:
        - df (pd.DataFrame): Input DataFrame
        - column_name (str): Name of the column to convert

        Returns:
        - pd.DataFrame: DataFrame with the specified column values converted to float or NaN
        """
        if column_name not in df.columns:
            raise ValueError(f"'{column_name}' column not found in the DataFrame.")
        df.loc[:, column_name] = df[column_name].apply(convert_ranges_string)
        df.loc[:, column_name] = df[column_name].apply(convert_string_to_float)
        df.loc[:, column_name] = df[column_name].astype(float)
        return df

    


    def mean_imputation_with_indicator(self, df, column_name='price'):
        """
        Perform mean imputation for NaN values in a specified column of a DataFrame.
        Create a new column indicating whether mean imputation was performed for each row.

        Parameters:
        - df (pd.DataFrame): Input DataFrame
        - column_name (str): Name of the column to perform mean imputation

        Returns:
        - pd.DataFrame: DataFrame with mean imputation applied and indicator column added
        """
        mean_value = df[column_name].mean()
        df.loc[:, column_name + '_missing'] = df[column_name].isna().astype(int)
        df.loc[:, column_name] = df[column_name].fillna(mean_value)

        return df

    def convert_str_list_one_col(self, column):
        """
        Convert a column containing string representations of lists into lists of strings.

        Parameters:
        - column (pandas.Series): The pandas Series containing string representations of lists.

        Returns:
        - pandas.Series: The pandas Series with each string representation converted into a list of strings.
        """
        return column.apply(lambda x: x.strip("[]").split(", ") if isinstance(x, str) and x.strip("[]") else [])

    def convert_str_lists_to_lists(self, df):
        """
        Convert columns containing string representations of lists into lists of strings.

        Parameters:
        - df (pd.DataFrame): Input DataFrame

        Returns:
        - pd.DataFrame: DataFrame with string lists converted into lists of strings
        """
        df.loc[:, 'also_view'] = self.convert_str_list_one_col(df['also_view'])
        df.loc[:, 'also_buy'] = self.convert_str_list_one_col(df['also_buy'])
        return df

    def get_nr_also_interacted_lists(self, df):
        """
        Calculate the number of elements in each list column and add as new columns.

        Parameters:
        - df (pd.DataFrame): Input DataFrame

        Returns:
        - pd.DataFrame: DataFrame with new columns containing the number of elements in each list
        """
        df.loc[:, 'nr_also_view'] = df['also_view'].apply(lambda x: len(x))
        df.loc[:, 'nr_also_buy'] = df['also_buy'].apply(lambda x: len(x))
        return df

    def drop_also_interacted_columns(self, df):
        """
        Drop columns 'also_view' and 'also_buy' from the DataFrame.

        Parameters:
        - df (pd.DataFrame): Input DataFrame

        Returns:
        - pd.DataFrame: DataFrame with specified columns dropped
        """
        df = df.drop(columns=['also_view', 'also_buy'])
        return df

    
    def np_to_tensor(self, arr):
        """
        Convert a pandas DataFrame to a PyTorch tensor.

        Parameters:
        - arr : numpy array

        Returns:
        - torch.Tensor: PyTorch tensor created from the DataFrame
        """
        tensor = torch.tensor(arr, dtype=torch.float32)
        return tensor

    def scale_data(self, data_batch):
        """
        Preprocess a batch of data using MinMaxScaler.

        Parameters:
        - data_batch (pd.DataFrame or np.ndarray): Batch of input data
        - training (bool): Flag indicating whether the function is called during training or inference

        Returns:
        - np.ndarray: Preprocessed batch of data
        """
        if self.training:
            if self.scaler is None:
                self.scaler = MinMaxScaler()
            self.scaler.partial_fit(data_batch)
        else:
            if self.scaler is None:
                with open(self.path_scaler, 'rb') as f:
                    self.scaler = pickle.load(f)
        

        return self.scaler.transform(data_batch)

    def save(self):
        
        with open(self.path_scaler, "wb") as f:
            pickle.dump(self.scaler, f)

def convert_string_to_float(input_string):
    """
    Convert a string to float, handling special cases.

    Parameters:
    - input_string (str): Input string to convert

    Returns:
    - float or NaN: Converted float value or NaN if conversion fails
    """
 

    pattern = r'[^0-9.,]'
    if isinstance(input_string, float):
        return input_string
    if input_string == '':
        return np.nan
    if len(re.findall(pattern, str(input_string))) < 1:
        return np.nan
    result = str(input_string).replace('$', '').replace(',', '')
    
    if len(result)>14:
        return np.nan
    
    try:
        return float(result)
    except ValueError as e:
        print(e)
        return np.nan    

def convert_ranges_string(input_string):
    """
    Convert a string representing a price range to a single price.

    Parameters:
    - input_string (str): Input string representing a price range

    Returns:
    - str: Converted price string
    """
    if '-' in input_string:
        parts = input_string.split('-')
        first_part = parts[0].strip()
        first_digit_index = next((i for i, c in enumerate(first_part) if c.isdigit()), None)
        if first_digit_index is not None:
            result = first_part[first_digit_index:]
            result = result.replace(' ', '')
            result = '$' + result
            return result
    return input_string
