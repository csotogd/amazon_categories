from fastapi import FastAPI, HTTPException
import torch
import pandas as pd
from src.utils.preprocessing.all_preprocessing import FeaturePreprocessor, PreprocessorLabels  # Import data preprocessing function
from src.model.global_model import GlobalModel
import json

app = FastAPI()

# Load the trained model artifact
model = GlobalModel()
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Load the objects needed for inference 
feat_proc = FeaturePreprocessor(training=False)
label_proc = PreprocessorLabels()

@app.post("/predict/")
async def predict(json_file: bytes = UploadFile(...)):
    """
    Predicts labels using a pre-trained machine learning model.

    Args:
        json_file (bytes): Uploaded JSON file containing data for prediction.

    Returns:
        JSON: Predicted labels as JSON format.

    Raises:
        HTTPException: If unable to process the input file.
    """
    try:
        # Read JSON data from the uploaded file
        data = json.loads(json_file.decode("utf-8"))
        
        # Convert JSON data to DataFrame
        input_df = pd.DataFrame(data)
        
        # Preprocess input data
        preprocessed_data = feat_proc.preprocess(input_df)
        
        # Perform inference using the loaded model
        input_tensor = torch.tensor(preprocessed_data.values, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
        
        # Transform model output to labels
        labels_json = label_proc.inverse_transform(output, as_json=True)
        
        return labels_json
    except Exception as e:
        raise HTTPException(status_code=400, detail="Unable to process input file")
