import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingReducer(nn.Module):
    def __init__(self, hf_embedding_size, output_size,dropout_prob, device):
        """
        Embedding reducer module for reducing dimensionality of NLP embeddings.

        Parameters:
        - hf_embedding_size (int): Size of the embedding ouput from hugging face.
        - output_size (int): Size of the output embeddings.
        - device (str): Device to use for the model.
        """
        super(EmbeddingReducer, self).__init__()
        self._output_size = output_size
        self.device = device
        self.fc1 = nn.Linear(hf_embedding_size*5, 64)  # Example hidden layer size
        self.fc2 = nn.Linear(64, output_size)  # Output layer
        self.dropout = nn.Dropout(p=dropout_prob) 

        self.to(device)

    def forward(self, x):
        """
        Forward pass of the embedding reducer.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after passing through the embedding reducer.
        """
        x = F.relu(self.fc1(x))  # Apply ReLU activation function
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer, no activation function
        return x

    
    def get_output_size(self):
        """Getter method for output_size."""
        return self._output_size