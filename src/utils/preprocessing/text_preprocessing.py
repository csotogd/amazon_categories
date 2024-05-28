import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from transformers import MobileBertTokenizer, MobileBertModel,  MobileBertTokenizerFast, BertModel, BertTokenizerFast
from datasets import Dataset


class Text_Preprocessor:

    def preprocess(self,df,  tokenizer, model):
        """
        Perform NLP preprocessing on a DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame containing text columns.
        model (torch.nn.Module): The NLP model used for generating embeddings.
        tokenizer: The tokenizer used to tokenize text data.

        Returns:
        torch.Tensor: A global tensor containing concatenated embeddings.
        """
        max_length = 60
        
        # Keep only NLP-related columns
        df, columns_nlp = self.keep_nlp_columns(df)
        
        # Convert all values in the DataFrame to string type
        df = self.convert_columns_to_string(df)
        
        # Truncate strings in NLP-related columns to a specified maximum length
        df = self.truncate_dataframe_columns(df, columns_nlp, max_length=max_length)
        
        # Convert DataFrame to Hugging Face Dataset
        dataset_hf = self.from_pandas_to_HF(df)

        # Tokenize the dataset
        tokenized_dataset = self.tokenize_dataset(dataset_hf, tokenizer, max_length)
        
        # Generate embeddings using the provided model
        embeddings = self.generate_embeddings(tokenized_dataset, model)
        
        return embeddings 

        # Optionally apply PCA here
        # return input_tensor


    def keep_nlp_columns(self, df):
        """
        Keep only the NLP-related columns in the DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.

        Returns:
        pandas.DataFrame: DataFrame with only NLP-related columns.
        list: List of NLP-related column names.
        """
        nlp_columns = ['brand', 'category', 'description', 'feature', 'title']
        df = df[nlp_columns]
        return df, nlp_columns

    def convert_columns_to_string(self, df):
        """
        Converts all values in all columns of a DataFrame to string type.

        Args:
            df (pandas.DataFrame): The DataFrame to convert.

        Returns:
            pandas.DataFrame: The DataFrame with all values converted to string type.
        """
        for column in df.columns:
            df.loc[:, column] = df[column].astype(str)
        return df

    def truncate_string(self, text, max_length):
        """
        Truncates a string to a specified maximum length.

        Args:
            text (str): The string to truncate.
            max_length (int, optional): The maximum length of the truncated string. Defaults to 180.

        Returns:
            str: The truncated string.
        """
        if len(text) <= max_length:
            return text
        else:
            return text[:max_length]

    def truncate_dataframe_columns(self, df, column_names, max_length=140):
        """
        Truncates strings in multiple columns of a DataFrame to a specified maximum length.

        Args:
            df (pandas.DataFrame): The DataFrame containing the columns to truncate.
            column_names (list of str): The names of the columns to truncate.
            max_length (int, optional): The maximum length of the truncated strings. Defaults to 180.

        Returns:
            pandas.DataFrame: The DataFrame with the specified columns truncated.
        """
        for column_name in column_names:
            df.loc[:, column_name] = df[column_name].apply(lambda x: self.truncate_string(x, max_length) if isinstance(x, str) else x)
        return df

    def from_pandas_to_HF(self, df):
        """
        Convert a pandas DataFrame to a Hugging Face Dataset.

        Args:
            df (pandas.DataFrame): The DataFrame to convert.

        Returns:
            datasets.Dataset: The converted dataset.
        """
        dataset = Dataset.from_pandas(df)
        return dataset

    def tokenize_dataset(self, dataset, tokenizer, max_length):
        """
        Tokenize the dataset using the provided tokenizer.

        Args:
            dataset (datasets.Dataset): The dataset to tokenize.
            tokenizer: The tokenizer to use for tokenization.
            max_length (int): The maximum length of tokenized sequences.

        Returns:
            dict: A dictionary containing tokenized sequences for each feature.
        """
        tokenized_dataset = {}
        
        # Tokenize each feature separately
        for feat in dataset.features.keys():
            tokenized_feat = tokenizer(dataset[feat], truncation=True, padding='max_length', max_length=max_length)
            tokenized_dataset[feat] = tokenized_feat
        
        return tokenized_dataset

    def generate_embeddings(self, tokenized_dataset, model):
        """
        Generate embeddings for tokenized sequences using the provided model.

        Args:
            tokenized_dataset (dict): A dictionary containing tokenized sequences for each feature.
            model (torch.nn.Module): The model used to generate embeddings.

        Returns:
            torch.Tensor: A tensor containing concatenated embeddings.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        embeddings = {}
        
        # Pass the tokenized features through the model
        for feat, tokenized_feat in tokenized_dataset.items():
            inputs = torch.tensor(tokenized_feat['input_ids']).to(device)  
            attention_mask = torch.tensor(tokenized_feat['attention_mask']).to(device)  
            
            with torch.no_grad():
                outputs = model(input_ids=inputs, attention_mask=attention_mask)
            
            # Aggregate the embeddings for each feature row
            embeddings[feat] = outputs.last_hidden_state.mean(dim=1)

        concatenated_embeddings = torch.cat([embeddings[key] for key in embeddings], dim=1)
        return concatenated_embeddings
    


