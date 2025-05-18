#!/bin/bash

# Start the FastAPI backend in the background
echo "Starting FastAPI backend..."
uvicorn server:app --host 0.0.0.0 --port 8882 &

# Start the Panel frontend in the foreground
echo "Starting Panel frontend..."
panel serve app.py --address 0.0.0.0 --port 5006 --allow-websocket-origin=* --autoreload
