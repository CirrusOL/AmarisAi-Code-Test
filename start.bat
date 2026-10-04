@echo off
pip install -r requirements.txt
start "" http://127.0.0.1:5000/
python app.py
