#!/bin/sh
. .venv/bin/activate
. ../config/.env
streamlit run app.py --server.port 8509  --server.fileWatcherType none
