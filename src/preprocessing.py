'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    model_pred_df = pd.read_csv('/Users/diyasayal/Desktop/INST414/problem-set-3/data/prediction_model_03.csv')
    genres_df = pd.read_csv('/Users/diyasayal/Desktop/INST414/problem-set-3/data/genres.csv')

    print("Model Predictions Columns:", model_pred_df.columns)
    print("Genres Columns:", genres_df.columns)
    
    return model_pred_df, genres_df
 


def process_data(model_pred_df, genres_df):
   
    genre_list = genres_df['genre'].tolist()
    
    genre_true_counts = {}
    genre_tp_counts = {}
    genre_fp_counts = {}
    
    # Initialize counts for each genre
    for genre in genre_list:
        genre_true_counts[genre] = model_pred_df[model_pred_df['actual genres'] == genre].shape[0]
        genre_tp_counts[genre] = model_pred_df[(model_pred_df['actual genres'] == genre) & (model_pred_df['predicted'] == genre)].shape[0]
        genre_fp_counts[genre] = model_pred_df[(model_pred_df['actual genres'] != genre) & (model_pred_df['predicted'] == genre)].shape[0]
    
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts

