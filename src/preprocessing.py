import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

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
    print("First few rows of model_pred_df:\n", model_pred_df.head())
    print("First few rows of genres_df:\n", genres_df.head())
    
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
    
    # Print to debug
    print("Genre True Counts:", genre_true_counts)
    print("Genre True Positive Counts:", genre_tp_counts)
    print("Genre False Positive Counts:", genre_fp_counts)
    
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts

def calculate_metrics(model_pred_df, genres_df):
    genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts = process_data(model_pred_df, genres_df)

    y_true = model_pred_df['actual genres']
    y_pred = model_pred_df['predicted']
    
    # Using scikit-learn to calculate micro and macro scores
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print("Micro-Precision:", micro_precision)
    print("Micro-Recall:", micro_recall)
    print("Micro-F1:", micro_f1)
    print("--------------------")
    print("Macro-Precision:", macro_precision)
    print("Macro-Recall:", macro_recall)
    print("Macro-F1:", macro_f1)
    print("True Labels Unique Values: ", y_true.unique())
    print("Predicted Labels Unique Values: ", y_pred.unique())
    print("Genre List: ", genre_list)

if __name__ == "__main__":
    model_pred_df, genres_df = load_data()
    calculate_metrics(model_pred_df, genres_df)


