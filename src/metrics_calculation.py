import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

    micro_tp = sum(genre_tp_counts.values())
    micro_fp = sum(genre_fp_counts.values())
    micro_fn = sum(genre_true_counts.values()) - micro_tp

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) != 0 else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) != 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) != 0 else 0

    macro_prec_list = []
    macro_recall_list = []
    macro_f1_list = []

    for genre in genre_list:
        tp = genre_tp_counts.get(genre, 0)
        fp = genre_fp_counts.get(genre, 0)
        fn = genre_true_counts.get(genre, 0) - tp

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

        macro_prec_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1)

    return micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list

def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    true_labels = model_pred_df['actual genres']
    predicted_labels = model_pred_df['predicted']

    # Debugging prints to ensure labels match correctly
    print("True Labels Unique Values: ", true_labels.unique())
    print("Predicted Labels Unique Values: ", predicted_labels.unique())
    print("Genre List: ", genre_list)

    macro_prec = precision_score(true_labels, predicted_labels, average='macro', labels=genre_list, zero_division=0)
    macro_rec = recall_score(true_labels, predicted_labels, average='macro', labels=genre_list, zero_division=0)
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro', labels=genre_list, zero_division=0)

    micro_prec = precision_score(true_labels, predicted_labels, average='micro', labels=genre_list, zero_division=0)
    micro_rec = recall_score(true_labels, predicted_labels, average='micro', labels=genre_list, zero_division=0)
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro', labels=genre_list, zero_division=0)

    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1

