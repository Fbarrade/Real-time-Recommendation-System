from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from recsys.datasets import (
    YelpDestinationsGNNDataset, YelpUserGNNDataset, YelpReviewsGNNDataset,
    YelpGNNDataset
)
import torch
from recsys.models import GNNModel, train_gnn_model, GNNConfigs  # Replace with the correct imports

from pathlib import Path 
import wandb 
import os 

wandb.login(key=os.getenv("WAND_API_KEY"))


data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)


models_dir = Path("./models")
models_dir.mkdir(exist_ok=True)


def process_places():
    places = YelpDestinationsGNNDataset()
    places.read_csv("./data/places.csv")
    places.process()
    return places.df

def process_users():
    users = YelpUserGNNDataset()
    users.read_csv("./data/users.csv")
    users.process()
    return users.df

def process_reviews():
    reviews = YelpReviewsGNNDataset()
    reviews.read_csv("./data/reviews.csv")
    reviews.process()
    return reviews.df

def aggregate_gnn_data(**context):

    user_df = context['ti'].xcom_pull(task_ids='process_users')
    destinations_df = context['ti'].xcom_pull(task_ids='process_places')
    reviews_df = context['ti'].xcom_pull(task_ids='process_reviews')

    gnn_dataset = YelpGNNDataset(
        user_df=user_df, 
        destinations_df=destinations_df, 
        reviews_df=reviews_df
    )
    gnn_dataset.process()
    gnn_dataset.save(data_dir)

def train_gnn_task():

    run = wandb.init(
        project="bigdata-recsys",
        notes="gnn-recsys",
        tags=["gnn", "recsys"],
        config= {"epochs": 300, }
    )

    trainset, testset, valset = torch.load(data_dir / "gnn_datasets.pt")
    
    configs = GNNConfigs(in_channels=495, hidden_channels=16, out_channels=1)
    model = GNNModel(configs)

    wandb.config.update({
        "lr": 0.001,
        "batch_size": 32,
    })
    
    train_gnn_model(model, trainset, run=run)

    torch.save(
        model.state_dict(), models_dir / "gnn-recsys-model.pt"
    )

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(models_dir / "gnn-recsys-model.pt")
    run.log_artifact(artifact)

    run.finish()

with DAG(
    dag_id="yelp_gnn_pipeline",
    schedule_interval=None,  # Can be adjusted based on your requirements
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["gnn", "yelp", "etl"]
) as dag:

    process_places_task = PythonOperator(
        task_id="process_places",
        python_callable=process_places
    )

    process_users_task = PythonOperator(
        task_id="process_users",
        python_callable=process_users
    )

    process_reviews_task = PythonOperator(
        task_id="process_reviews",
        python_callable=process_reviews
    )

    aggregate_gnn_task = PythonOperator(
        task_id="create_gnn_data",
        python_callable=aggregate_gnn_data,
        provide_context=True
    )

    train_gnn_model_task = PythonOperator(
        task_id="train_gnn_model",
        python_callable=train_gnn_task,
    )

    # Define task dependencies
    [process_places_task, process_users_task, process_reviews_task] >> aggregate_gnn_task >> train_gnn_model_task