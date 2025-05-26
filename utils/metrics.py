import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Class for calculating various metrics for twin verification."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and labels."""
        self.predictions = []
        self.labels = []
        self.similarities = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, similarities: torch.Tensor = None):
        """
        Update metrics with new batch of predictions.
        
        Args:
            predictions: Binary predictions (B,)
            labels: Ground truth labels (B,)
            similarities: Similarity scores (B,)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if similarities is not None and isinstance(similarities, torch.Tensor):
            similarities = similarities.detach().cpu().numpy()
        
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        if similarities is not None:
            self.similarities.extend(similarities)
    
    def compute_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute various classification metrics.
        
        Args:
            threshold: Decision threshold for binary classification
        Returns:
            Dictionary of metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        # Convert similarities to binary predictions if available
        if len(self.similarities) > 0:
            binary_preds = (np.array(self.similarities) >= threshold).astype(int)
        else:
            binary_preds = (np.array(self.predictions) >= threshold).astype(int)
        
        labels = np.array(self.labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, binary_preds),
            'precision': precision_score(labels, binary_preds, zero_division=0),
            'recall': recall_score(labels, binary_preds, zero_division=0),
            'f1_score': f1_score(labels, binary_preds, zero_division=0),
        }
        
        # ROC-AUC (requires probability scores)
        if len(self.similarities) > 0:
            try:
                metrics['roc_auc'] = roc_auc_score(labels, self.similarities)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        # Equal Error Rate (EER)
        if len(self.similarities) > 0:
            metrics['eer'] = self.compute_eer(labels, self.similarities)
        
        return metrics
    
    def compute_eer(self, labels: np.ndarray, scores: np.ndarray) -> float:
        """
        Compute Equal Error Rate (EER).
        
        Args:
            labels: Ground truth binary labels
            scores: Similarity scores
        Returns:
            Equal Error Rate
        """
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        
        # Find the point where FPR and FNR are closest
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        return eer
    
    def find_best_threshold(self) -> Tuple[float, Dict[str, float]]:
        """
        Find the best threshold based on F1 score.
        
        Returns:
            Best threshold and corresponding metrics
        """
        if len(self.similarities) == 0:
            return 0.5, {}
        
        labels = np.array(self.labels)
        similarities = np.array(self.similarities)
        
        # Try different thresholds
        thresholds = np.linspace(0.0, 1.0, 101)
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            binary_preds = (similarities >= threshold).astype(int)
            f1 = f1_score(labels, binary_preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'accuracy': accuracy_score(labels, binary_preds),
                    'precision': precision_score(labels, binary_preds, zero_division=0),
                    'recall': recall_score(labels, binary_preds, zero_division=0),
                    'f1_score': f1,
                    'threshold': threshold
                }
        
        return best_threshold, best_metrics
    
    def get_confusion_matrix(self, threshold: float = 0.5) -> np.ndarray:
        """Get confusion matrix."""
        if len(self.similarities) > 0:
            binary_preds = (np.array(self.similarities) >= threshold).astype(int)
        else:
            binary_preds = (np.array(self.predictions) >= threshold).astype(int)
        
        return confusion_matrix(self.labels, binary_preds)
    
    def get_classification_report(self, threshold: float = 0.5) -> str:
        """Get detailed classification report."""
        if len(self.similarities) > 0:
            binary_preds = (np.array(self.similarities) >= threshold).astype(int)
        else:
            binary_preds = (np.array(self.predictions) >= threshold).astype(int)
        
        return classification_report(
            self.labels, 
            binary_preds,
            target_names=['Non-Twin', 'Twin']
        )


def compute_similarity_metrics(similarities1: torch.Tensor, similarities2: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics between two sets of similarity scores.
    
    Args:
        similarities1: First set of similarities
        similarities2: Second set of similarities
    Returns:
        Dictionary of comparison metrics
    """
    if isinstance(similarities1, torch.Tensor):
        similarities1 = similarities1.detach().cpu().numpy()
    if isinstance(similarities2, torch.Tensor):
        similarities2 = similarities2.detach().cpu().numpy()
    
    # Correlation
    correlation = np.corrcoef(similarities1, similarities2)[0, 1]
    
    # Mean absolute error
    mae = np.mean(np.abs(similarities1 - similarities2))
    
    # Mean squared error
    mse = np.mean((similarities1 - similarities2) ** 2)
    
    # Root mean squared error
    rmse = np.sqrt(mse)
    
    return {
        'correlation': correlation,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }


def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, save_path: Optional[str] = None):
    """
    Plot ROC curve.
    
    Args:
        labels: Ground truth binary labels
        scores: Similarity scores
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(labels: np.ndarray, scores: np.ndarray, save_path: Optional[str] = None):
    """
    Plot Precision-Recall curve.
    
    Args:
        labels: Ground truth binary labels
        scores: Similarity scores
        save_path: Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Twin', 'Twin'],
                yticklabels=['Non-Twin', 'Twin'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_similarity_distribution(similarities: np.ndarray, labels: np.ndarray, save_path: Optional[str] = None):
    """
    Plot distribution of similarity scores for twins vs non-twins.
    
    Args:
        similarities: Similarity scores
        labels: Binary labels
        save_path: Path to save the plot
    """
    twin_similarities = similarities[labels == 1]
    non_twin_similarities = similarities[labels == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(non_twin_similarities, bins=50, alpha=0.7, label='Non-Twins', color='red', density=True)
    plt.hist(twin_similarities, bins=50, alpha=0.7, label='Twins', color='blue', density=True)
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.title('Distribution of Similarity Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


class ValidationMetrics:
    """Class to track metrics during validation."""
    
    def __init__(self):
        self.metrics_history = []
        self.best_metrics = {}
        self.best_epoch = 0
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update metrics for current epoch."""
        metrics['epoch'] = epoch
        self.metrics_history.append(metrics.copy())
        
        # Update best metrics based on F1 score
        if not self.best_metrics or metrics.get('f1_score', 0) > self.best_metrics.get('f1_score', 0):
            self.best_metrics = metrics.copy()
            self.best_epoch = epoch
    
    def get_best_metrics(self) -> Tuple[int, Dict[str, float]]:
        """Get best metrics and corresponding epoch."""
        return self.best_epoch, self.best_metrics
    
    def plot_metrics(self, metrics_to_plot: List[str] = None, save_path: Optional[str] = None):
        """Plot metrics over epochs."""
        if not self.metrics_history:
            return
        
        if metrics_to_plot is None:
            metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
        
        epochs = [m['epoch'] for m in self.metrics_history]
        
        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(metrics_to_plot):
            if metric in self.metrics_history[0]:
                values = [m[metric] for m in self.metrics_history]
                plt.subplot(2, 2, i + 1)
                plt.plot(epochs, values, marker='o')
                plt.title(f'{metric.title()}')
                plt.xlabel('Epoch')
                plt.ylabel(metric.title())
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def evaluate_model_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    similarities: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        predictions: Binary predictions or similarity scores
        labels: Ground truth labels
        similarities: Similarity scores (if different from predictions)
        save_dir: Directory to save plots
    Returns:
        Dictionary of computed metrics
    """
    # Initialize metrics calculator
    calculator = MetricsCalculator()
    
    # Update with predictions
    if similarities is not None:
        calculator.update(predictions, labels, similarities)
    else:
        calculator.update(predictions, labels, predictions)
    
    # Find best threshold
    best_threshold, best_metrics = calculator.find_best_threshold()
    
    # Compute metrics with best threshold
    metrics = calculator.compute_metrics(best_threshold)
    
    # Add best threshold to metrics
    metrics['best_threshold'] = best_threshold
    
    # Print results
    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    if 'eer' in metrics:
        print(f"Equal Error Rate: {metrics['eer']:.3f}")
    
    print("\nClassification Report:")
    print(calculator.get_classification_report(best_threshold))
    
    # Generate plots if save directory is provided
    if save_dir and similarities is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # ROC curve
        plot_roc_curve(labels, similarities, os.path.join(save_dir, 'roc_curve.png'))
        
        # Precision-Recall curve
        plot_precision_recall_curve(labels, similarities, os.path.join(save_dir, 'pr_curve.png'))
        
        # Confusion matrix
        cm = calculator.get_confusion_matrix(best_threshold)
        plot_confusion_matrix(cm, os.path.join(save_dir, 'confusion_matrix.png'))
        
        # Similarity distribution
        plot_similarity_distribution(similarities, labels, os.path.join(save_dir, 'similarity_distribution.png'))
    
    return metrics