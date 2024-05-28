from transformers import BertTokenizerFast, BertModel, BertConfig
import torch

class HuggingFaceWrapper:
    """
    A class representing an object using Hugging Face's transformers library.
    
    Attributes:
        model_name (str): The name of the pre-trained model.
        _tokenizer (BertTokenizerFast): A tokenizer object for the model.
        _model (BertModel): A pre-trained model object.
        _nr_hidden_layers (int): Number of hidden layers in the model.
    """
    
    model_name = "gaunernst/bert-tiny-uncased"
    
    def __init__(self) -> None:
        """
        Initialize HuggingFaceObject with a pre-trained model and tokenizer.
        """
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._tokenizer = BertTokenizerFast.from_pretrained(HuggingFaceWrapper.model_name)
        self._model = BertModel.from_pretrained(HuggingFaceWrapper.model_name).to(self._device)
        self._nr_hidden_layers = BertConfig.from_pretrained(HuggingFaceWrapper.model_name).hidden_size

        # Freeze the parameters of the BERT model, we are not training this part of the model
        for param in self._model.parameters():
            param.requires_grad = False
    
    def get_tokenizer(self) -> BertTokenizerFast:
        """
        Get the tokenizer object.

        Returns:
            BertTokenizerFast: The tokenizer object.
        """
        return self._tokenizer
    
    def get_model(self) -> BertModel:
        """
        Get the pre-trained model object.

        Returns:
            BertModel: The pre-trained model object.
        """
        return self._model
    
    def get_nr_hidden_layers(self) -> int:
        """
        Get the number of hidden layers in the model.

        Returns:
            int: Number of hidden layers.
        """
        return self._nr_hidden_layers

    def get_device(self) -> str:
        """
        Get the torch device

        Returns:
            str: 'cpu' or 'cuda'
        """
        return self._device
