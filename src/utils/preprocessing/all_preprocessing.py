from utils.preprocessing.numerical_preprocessing import Numerical_Preprocessor
from utils.preprocessing.text_preprocessing import Text_Preprocessor
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import torch

class FeaturePreprocessor:
    def __init__(self, training) -> None:
        self.training=training #whether we are in training mode
        self.numerical_preprocessor = Numerical_Preprocessor(training=training)
        self.text_preprocessor = Text_Preprocessor()
    


    def preprocess(self, df, hf_wrapper, images=False):
        """
        Preprocess all features in the given DataFrame.

        This function preprocesses text and numerical features in the DataFrame.
        Optionally, it can also preprocess image features if `images` is set to True.

        Parameters:
        df (pandas.DataFrame): The input DataFrame containing features to be preprocessed.
        hf_wrapper: object of type HuggingFaceWrapper
        training (bool): A flag indicating whether the function is called during training or not.
        images (bool, optional): A flag indicating whether to preprocess image features. Defaults to False.

        Returns:
        tuple: A tuple containing preprocessed tensors for text and numerical features.
        """
        tensor_text = self.text_preprocessor.preprocess(df, hf_wrapper.get_tokenizer(), hf_wrapper.get_model())
        tensor_numeric = self.numerical_preprocessor.preprocess(df)
        if images:
            print('you should add image function')
        
        return tensor_text, tensor_numeric
    
    def save(self):
        self.numerical_preprocessor.save()#writes the scaler into disk
        
        #no need to do anything for the other one


class PreprocessorLabels:
    # Define labels_ref and nr_labels as class attributes
    labels_ref = ["All Electronics", "Amazon Fashion", "Amazon Home", "Arts, Crafts & Sewing", "Automotive",
                  "Books", "Buy a Kindle", "Camera & Photo", "Cell Phones & Accessories", "Computers",
                  "Digital Music", "Grocery", "Health & Personal Care", "Home Audio & Theater",
                  "Industrial & Scientific", "Movies & TV", "Musical Instruments", "Office Products",
                  "Pet Supplies", "Sports & Outdoors", "Tools & Home Improvement", "Toys & Games",
                  "Video Games"]
    nr_labels = len(labels_ref)

    def __init__(self):
        """
        Initialize PreprocessLabels class.
        """
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(PreprocessorLabels.labels_ref)  # Use class attribute here
        

    def transform(self, series_labels):
        """
        Encode labels into one-hot encoded tensors.

        Parameters:
        series_labels (pandas.Series): The input Pandas Series containing label strings.

        Returns:
        torch.Tensor: One-hot encoded label tensors.
        """

        # Transform labels to one-hot encoded vectors
        one_hot_encoded_labels = self.label_binarizer.transform(series_labels)

        # Convert to PyTorch tensor
        one_hot_encoded_labels_tensor = torch.tensor(one_hot_encoded_labels, dtype=torch.float32)

        return one_hot_encoded_labels_tensor
    
    def inverse_transform(self, torch_prediction, as_json=True):

        # Ensure input_tensor is a PyTorch tensor
        if not isinstance(torch_prediction, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")
        

        # Set maximum value in each row to 1 and the rest to 0
        max_values, _ = torch_prediction.max(dim=1, keepdim=True)
        output_tensor = torch.where(torch_prediction == max_values, torch.tensor(1), torch.tensor(0))
        
        # Convert tensor to pandas DataFrame
        main_cats =self.label_binarizer.inverse_transform(output_tensor.cpu().numpy())
        if as_json:
            main_cats = main_cats.to_json()
        return main_cats

    @classmethod
    def get_nr_labels(cls):
        """
        Getter method for nr_labels attribute.
        """
        return cls.nr_labels