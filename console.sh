#!/usr/bin/env bash

COMMAND=$1

case "$COMMAND" in

    mlflow)
        echo "Starting MLflow UI..."

        uv run mlflow ui \
            --backend-store-uri "sqlite:///$(pwd)/mlflow/mlflow.db" \
            --artifacts-destination "$(pwd)/mlflow/artifacts" \
            --host 0.0.0.0 \
            --port 5000
        ;;

    streamlit)
        echo "Starting Streamlit..."

        uv run streamlit run app/main.py
        ;;

    *)
        echo "Usage: ./console.sh {mlflow|streamlit}"
        exit 1
        ;;
esac