#!/bin/bash

# PYTHONPATH=$PWD python api.py

gunicorn api:app --preload --bind 0.0.0.0:5556 --workers 8 -t=120 --worker-class uvicorn.workers.UvicornWorker
