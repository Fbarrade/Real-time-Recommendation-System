from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG('attractions_pipeline', default_args=default_args, schedule_interval='@daily') as dag:
    start_kafka_producer = BashOperator(
        task_id='start_kafka_producer',
        bash_command='python /opt/airflow/dags/kafka_producer.py'
    )

    start_spark_consumer = BashOperator(
        task_id='start_spark_consumer',
        bash_command='python /opt/airflow/dags/spark_consumer.py'
    )

    start_kafka_producer >> start_spark_consumer
