import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.nlp_nn import EmbeddingReducer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing.all_preprocessing import PreprocessorLabels

class GlobalModel(nn.Module):
    def __init__(self, hf_embedding_size, numeric_input_size,dropout_prob, device):
        """
        Initialize the global model.

        Parameters:
        - hf_embedding_size (int): Size of the output of one single embedding from hf model
        - numeric_input_size (int): Size of the input tensor for numeric features.
        - output_size (int): Number of output classes.
        - device (str): Device to use for the model.
        """
        super(GlobalModel, self).__init__()
        

        # Define layers for NLP feature processing
        self.embedding_reducer = EmbeddingReducer(hf_embedding_size= hf_embedding_size, dropout_prob=dropout_prob, device=device,output_size=32)

        # Define layers for numeric feature processing
        self.numeric_model = nn.Linear(numeric_input_size, 32)  # Example linear layer for numeric features

        # Define prediction head
        self.fc1 = nn.Linear(self.embedding_reducer.get_output_size() + numeric_input_size, 64)  # Example hidden layer size
        self.fc2 = nn.Linear(64, PreprocessorLabels.get_nr_labels())  # Output layer size corresponds to number of labels
        self.dropout = nn.Dropout(p=dropout_prob) 
        self.dropout_prob = dropout_prob
        self.device= device
        self.to(device)


    def forward(self, nlp_tensor, numeric_tensor):
        """
        Forward pass of the global model.

        Parameters:
        - nlp_tensor (torch.Tensor): Input tensor for NLP features.
        - numeric_tensor (torch.Tensor): Input tensor for numeric features.

        Returns:
        - torch.Tensor: Output tensor after passing through the global model.
        """
        # Apply embedding reducer to NLP tensor
        nlp_embedding = self.embedding_reducer(nlp_tensor)


        # Concatenate NLP reduced embeddings with processed numeric features
        global_tensor = torch.cat((nlp_embedding, numeric_tensor.to(self.device)), dim=1)

        # Apply prediction head
        x = F.relu(self.fc1(global_tensor))  # Apply ReLU activation function
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer, no activation function
        x = F.softmax(x, dim=0)  # Apply softmax for classification

        return x
