#!/bin/bash
# Start the FastAPI backend in the background
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend in the foreground
streamlit run src/app/ui.py --server.port 8501 --server.address 0.0.0.0