FROM quay.io/astronomer/astro-runtime:12.6.0

COPY packages.txt .

RUN /usr/local/bin/install-system-packages

COPY requirements.txt .

COPY recsys /usr/local/airflow/recsys

RUN pip install -e /usr/local/airflow/

RUN /usr/local/bin/install-python-dependencies
