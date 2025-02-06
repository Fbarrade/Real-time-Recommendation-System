from fastapi import FastAPI, Request
from recsys.datasets import (
    YelpUserGNNDataset, YelpDestinationsGNNDataset, YelpReviewsGNNDataset, YelpGNNDataset
)
from recsys.models import GNNConfigs, GNNModel
from pathlib import Path
import pandas as pd
import wandb
import os
import torch
from contextlib import asynccontextmanager

# Global variable for the model
model = None
model_is_downloaded = False

def lifespan(app: FastAPI):
    """
    The lifespan context loads the model artifact at startup.
    """
    global model, model_is_downloaded

    # Initialize wandb and download the artifact.
    run = wandb.init(project="your_project_name")  # add project or other configs as needed
    artifact = run.use_artifact('moussa2bacsmbiof-ensias/bigdata-recsys/model:v0', type='model')
    artifact_dir = artifact.download()

    # Assuming the model is saved as 'model.pt' within the artifact directory.
    model_path = Path(artifact_dir) / "gnn-recsys-model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected model file not found at {model_path}")

    # Load the model from the artifact.
    model_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)

    
    model = GNNModel(
        configs= GNNConfigs(
            hidden_channels=23,
            num_gnn_layers=3, 
            num_fc_layers=2
        )
    )

    model.load_state_dict(model_dict)

    model.eval()  # Set the model to evaluation mode

    print("Model loaded successfully with config:")
    print(run.config)

    # (Optional) Clean up resources or finish the wandb run.
    run.finish()

    model_is_downloaded = True

# Make sure to log in to wandb before starting the app.
wandb.login(key=os.getenv("WAND_API_KEY"))

# Create the FastAPI app using the lifespan context manager.
app = FastAPI()

@app.post("/recsys-gnn")
async def process_data(request: Request):
    
    if not model_is_downloaded:
        lifespan(app=app)

    save_dir = Path("./data/predict")
    save_dir.mkdir(parents=True, exist_ok=True)

    data = await request.json()

    user_resp = data["users"]
    places_resp = data["places"]
    reviews_resp = data["reviews"]

    user_df = pd.DataFrame.from_dict(user_resp)
    places_df = pd.DataFrame.from_dict(places_resp)
    reviews_df = pd.DataFrame.from_dict(reviews_resp)

    user_dataset = YelpUserGNNDataset()
    user_dataset.df = user_df
    user_dataset.process()

    places_dataset = YelpDestinationsGNNDataset()
    places_dataset.df = places_df
    places_dataset.process()

    reviews_dataset = YelpReviewsGNNDataset()
    reviews_dataset.df = reviews_df
    reviews_dataset.process()

    print(user_df)

    gnn_dataset = YelpGNNDataset(
        user_df=user_df,
        destinations_df=places_df,  
        reviews_df=reviews_df
    )
    gnn_dataset.process()
    gnn_dataset.save(save_dir)

    dataset, _, _ = torch.load(save_dir / "gnn_datasets.pt")

    model.eval()
    with torch.no_grad():
        predictions = model(dataset)

    predictions_list = predictions.cpu().numpy().tolist()

    return {"predictions": predictions_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
