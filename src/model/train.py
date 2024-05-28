from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import torch
import torch.nn as nn
import torch.optim as optim
from global_model import GlobalModel
from utils.dataLoaders.dataLoader import Dataloader
from hugging_wrapper import HuggingFaceWrapper
import os
import sys
import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing.all_preprocessing import *

def training_loop(model, path_data, hf_wrapper,num_epochs=2):
    """
    Train the model using the provided dataloader.

    Args:
    model (torch.nn.Module): The neural network model.
    path_data(str):  path to directory containing the files to read by dataloader
    hf_wrapper (HuggingFaceWrapper object)
    num_epochs (int): Number of training epochs. Default is 10.
    """

    dataloader = Dataloader(file_path=path_data+'train.csv',batch_size=500)
    preprocessor_labels = PreprocessorLabels()
    feat_preprocessor= FeaturePreprocessor(training=True)
    # good for classification where labels are ohe and passed through a softmax layer
    criterion = nn.CrossEntropyLoss()
    # Define your optimizer
    learning_rate = 0.0005
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    
    experiment = Experiment(api_key="4VkJgVdNgrS4RP4dnW1Fq4vP6", project_name="fever", workspace="csotogd")
    experiment.log_parameters({
    "learning_rate": learning_rate,
    "batch_size": dataloader.batch_size,
    "loss_criterion": str(criterion),
    "optimizer": str(optimizer),
    "dropout_prob": model.dropout_prob,
    "num_epochs": num_epochs,
    # Add more parameters as needed
    })
    


    # Set model to training mode
    model.train()

    # Training loop
    for epoch in range(num_epochs):
        
        print('epoch: ', epoch)
        # Reset dataloader for each epoch
        i=0
        dataloader.restart()

        # Initialize epoch loss
        epoch_loss = 0.0

        # Loop over batches
        while dataloader.hasNext():
            batch_df = dataloader.next()
            batch_text, batch_numeric = feat_preprocessor.preprocess(batch_df, hf_wrapper)
            batch_labels = preprocessor_labels.transform(batch_df['main_cat'])
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_text, batch_numeric)

            # Compute loss
            loss = criterion(outputs,batch_labels.to(hf_wrapper.get_device()))

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            experiment.log_metric("batch loss", loss.item(), step=epoch * i )
            i+=1

        # Calculate average loss for the epoch
        feat_preprocessor.save()
        avg_train_epoch_loss = epoch_loss / dataloader.total_rows
        experiment.log_metric("training avge epoch loss", avg_train_epoch_loss, step=epoch )
        evaluate_validation_set(model, path_data, hf_wrapper, preprocessor_labels, criterion, experiment, epoch)
    
    experiment.end()





import torch

def evaluate_validation_set(model, path_data_loaders, hf_wrapper, preprocessor_labels, criterion, experiment, epoch):
    """
    Evaluate the model on the validation set without tracking gradients.

    Parameters:
    - model (torch.nn.Module): PyTorch model to evaluate.
    - path_data_loaders (str): Path to the validation data loaders.
    - hf_wrapper: HFWrapper object for preprocessing text data.
    - preprocessor_labels: PreprocessorLabels object for preprocessing labels.
    - criterion: Loss function criterion.
    - experiment: Experiment object for logging metrics.
    - epoch (int): Current epoch number.

    Returns:
    - model (torch.nn.Module): Updated model.
    """
    # Create DataLoader for validation set
    dataloader = Dataloader(file_path=path_data_loaders + 'val.csv', batch_size=500)
    
    # Initialize feature preprocessor and label preprocessor
    feat_preprocessor = FeaturePreprocessor(training=False)
    preprocessor_labels = PreprocessorLabels()

    # Initialize epoch loss
    epoch_loss = 0
    
    # Evaluate the model without tracking gradients
    with torch.no_grad():
        # Iterate over validation set batches
        while dataloader.hasNext():
            batch_df = dataloader.next()
            batch_text, batch_numeric = feat_preprocessor.preprocess(batch_df, hf_wrapper)
            batch_labels = preprocessor_labels.transform(batch_df['main_cat'])
            model.eval()

            # Forward pass
            outputs = model(batch_text, batch_numeric)
            
            # Compute loss
            loss = criterion(outputs, batch_labels.to(hf_wrapper.get_device()))

            # Accumulate loss
            epoch_loss += loss.item()

    # Calculate average loss for the epoch
    avg_val_epoch_loss = epoch_loss / dataloader.total_rows
    experiment.log_metric("val avge epoch loss", avg_val_epoch_loss, step=epoch)

    return model







# Example usage:
def train_model():

    # Instantiate the Dataloader with your CSV file path
    path_data= 'data/subset/'

    hf_wrapper = HuggingFaceWrapper()
    model = GlobalModel(hf_wrapper.get_nr_hidden_layers(), numeric_input_size= 4, dropout_prob=0.5,  device=hf_wrapper.get_device())

    # Train the model
    trained_model = training_loop(model, path_data, hf_wrapper,num_epochs=5)
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    train_model()