# run_modeling.py
import sys
from src.train import train_and_save_models

if __name__ == "__main__":
    print("====================================================")
    # Start modeling engine
    train_and_save_models()
    print("====================================================")