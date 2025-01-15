#!/usr/bin/env bash

# Generate and manipulate data into the correct format
python data_gen.py

# Trains the model and saves the weights as adapters.npz
python lora.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --train \
        --iters 150 \
        --save-every 10

# Run inference
python query.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --query "Which year did John Wallace play for Syracuse?"