"""
The code for model inference goes here.
"""
from train import load_and_clean_data, training, process_data, clean_data

def predict(inp):
    df = load_and_clean_data()
    model = training(df)
    inp = process_data(inp)
    
    return(model.predict(inp))