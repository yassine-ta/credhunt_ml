@echo off

REM Create data directories
mkdir data
mkdir data\raw
mkdir data\processed

REM Create src directory and files
mkdir src
cd src
echo. > __init__.py
echo import pandas as pd > data_loader.py
echo import re > preprocessor.py
echo import torch > lcbow.py
echo import fasttext > fasttext_model.py
echo import pandas as pd > train.py
echo import pandas as pd > evaluate.py
cd ..

REM Create notebooks directory
mkdir notebooks
cd notebooks
echo. > exploration.ipynb
cd ..

REM Create requirements.txt
echo pandas > requirements.txt
echo scikit-learn >> requirements.txt
echo torch >> requirements.txt
echo torchvision >> requirements.txt
echo joblib >> requirements.txt
echo keras >> requirements.txt

REM Create README.md
echo # LCBOW Model Project > README.md
echo This project implements the LCBOW model using PyTorch and benchmarks it against the FastText model. >> README.md

REM Create .gitignore
echo __pycache__/ > .gitignore
echo *.pyc >> .gitignore
echo data/processed/ >> .gitignore
echo models/ >> .gitignore

echo Project structure created successfully!
pause