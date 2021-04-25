#!/bin/bash

conda create --name imdb python=3.7.8
conda activate imdb
pip install --upgrade pip
pip install tensorflow==2.4.1 ipykernel
python -m ipykernel install --user --name=imdb
conda deactivate
