import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def calculate_eer(labels, scores):
    """Calculate Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find the threshold where FPR and FNR are closest
    eer_index = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    
    return eer


def plot_roc_curve(labels, scores, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Twin Verification')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def calculate_verification_metrics(labels, scores, threshold=0.5):
    """Calculate comprehensive verification metrics"""
    predictions = (scores > threshold).astype(int)
    
    # Basic metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Verification specific metrics
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Accept Rate
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Reject Rate
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'far': far,
        'frr': frr,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
