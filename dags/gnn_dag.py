from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from recsys.datasets import (
    YelpDestinationsGNNDataset, YelpUserGNNDataset, YelpReviewsGNNDataset,
    YelpGNNDataset
)
import torch
from recsys.models import (
    GNNModel, train_gnn_model, GNNConfigs, train_step_optuna
    
) # Replace with the correct imports

from optuna.integration.wandb import WeightsAndBiasesCallback
import optuna
from optuna.trial import TrialState

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


def hyperparams_optimization(**context):

    def objective(trial): 
        trainset, _, _ = torch.load(data_dir / "gnn_datasets.pt")

        loss = train_step_optuna(trial, trainset=trainset)

        return loss 

    # wandbc = WeightsAndBiasesCallback(wandb_kwargs={
    #     "project": "my-project"
    # })
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=600)

    context['ti'].xcom_push(key='best_hyperparameters', value=study.best_trial.params)
    # context['ti'].xcom_push(key='run_id', value=study.best_trial.params)

def train_gnn_task(**context):

    best_hyperparams = context['ti'].xcom_pull(task_ids='hyperparams_optim_optuna', key='best_hyperparameters')

    run = wandb.init(
        project="bigdata-recsys",
        notes="gnn-recsys",
        tags=["gnn", "recsys"],
        config={"epochs": 300, **best_hyperparams}
    )

    trainset, testset, valset = torch.load(data_dir / "gnn_datasets.pt")
    
    model = GNNModel(
        configs=GNNConfigs(
            hidden_channels= run.config.get("hidden_channels"), 
            num_gnn_layers= run.config.get("n_gnn_layers"),
            num_fc_layers= run.config.get("n_fc_layers")
        )
    )

    # wandb.config.update({
    #     "batch_size": 32,
    # })
    
    train_gnn_model(model=model, trainset=trainset, testset=testset, run=run)

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

    hyperparams_optimization_task = PythonOperator(
        task_id="hyperparams_optim_optuna",
        python_callable=hyperparams_optimization,
        provide_context=True
    )

    train_gnn_model_task = PythonOperator(
        task_id="train_gnn_model",
        python_callable=train_gnn_task,
        provide_context=True
    )

    # Define task dependencies
    [process_places_task, process_users_task, process_reviews_task] >> aggregate_gnn_task >> hyperparams_optimization_task >> train_gnn_model_task
