import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Any
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.atvn import ATVNModel
from data.dataset import TwinDataset
from data.transforms import create_data_transforms
from utils.metrics import TwinMetrics
from utils.inference import load_model_checkpoint
from utils.visualization import plot_confusion_matrix, plot_roc_curve


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_test_loader(config: Dict[str, Any], test_data_path: str = None):
    """Create test data loader."""
    transforms = create_data_transforms(config)
    
    # Use provided test data path or default from config
    dataset_root = test_data_path if test_data_path else config['data']['dataset_root']
    
    test_dataset = TwinDataset(
        dataset_root=dataset_root,
        pairs_file=config['data']['pairs_file'],
        split='test',
        image_size=tuple(config['data']['image_size']),
        transform=transforms['test']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return test_loader


def evaluate_model(model, test_loader, device, save_predictions: bool = False):
    """Evaluate model on test data."""
    model.eval()
    all_similarities = []
    all_labels = []
    all_predictions = []
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluation'):
            image1, image2, labels = batch
            image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(image1, image2)
            similarities = outputs['similarity']
            
            # Convert similarities to predictions (threshold = 0.5)
            predictions = (similarities > 0.5).float()
            
            # Collect results
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    # Convert to numpy arrays
    similarities = np.array(all_similarities)
    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    
    # Compute metrics
    metrics = TwinMetrics()
    metrics_dict = metrics.compute_metrics(similarities, labels)
    
    # Save predictions if requested
    if save_predictions:
        predictions_data = {
            'similarities': similarities.tolist(),
            'predictions': predictions.tolist(),
            'labels': labels.tolist(),
            'metrics': metrics_dict
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/test_predictions.json', 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        print("Predictions saved to results/test_predictions.json")
    
    return similarities, labels, predictions, metrics_dict


def generate_evaluation_report(metrics_dict: Dict[str, Any], output_path: str = 'results/evaluation_report.txt'):
    """Generate detailed evaluation report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("ATVN Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Classification Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {metrics_dict['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics_dict['precision']:.4f}\n")
        f.write(f"Recall: {metrics_dict['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics_dict['f1_score']:.4f}\n\n")
        
        f.write("ROC Analysis:\n")
        f.write("-" * 20 + "\n")
        f.write(f"AUC: {metrics_dict['auc']:.4f}\n")
        f.write(f"Equal Error Rate (EER): {metrics_dict['eer']:.4f}\n\n")
        
        if 'confusion_matrix' in metrics_dict:
            f.write("Confusion Matrix:\n")
            f.write("-" * 20 + "\n")
            cm = metrics_dict['confusion_matrix']
            f.write(f"True Negatives:  {cm[0, 0]}\n")
            f.write(f"False Positives: {cm[0, 1]}\n")
            f.write(f"False Negatives: {cm[1, 0]}\n")
            f.write(f"True Positives:  {cm[1, 1]}\n\n")
        
        f.write("Classification Report:\n")
        f.write("-" * 20 + "\n")
        if 'classification_report' in metrics_dict:
            f.write(metrics_dict['classification_report'])
    
    print(f"Evaluation report saved to {output_path}")


def plot_evaluation_results(similarities, labels, output_dir: str = 'results'):
    """Plot evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plot_roc_curve(similarities, labels)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    predictions = (similarities > 0.5).astype(int)
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(labels, predictions)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot similarity distribution
    plt.figure(figsize=(10, 6))
    
    # Separate similarities by true label
    twin_similarities = similarities[labels == 1]
    non_twin_similarities = similarities[labels == 0]
    
    plt.hist(non_twin_similarities, bins=50, alpha=0.7, label='Non-twins', color='red')
    plt.hist(twin_similarities, bins=50, alpha=0.7, label='Twins', color='blue')
    
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


def benchmark_inference_speed(model, test_loader, device, num_batches: int = 100):
    """Benchmark model inference speed."""
    model.eval()
    
    print(f"Benchmarking inference speed on {num_batches} batches...")
    
    import time
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_batches:
                break
                
            image1, image2, _ = batch
            image1, image2 = image1.to(device), image2.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(image1, image2)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            batch_time = end_time - start_time
            times.append(batch_time)
    
    times = np.array(times)
    batch_size = test_loader.batch_size
    
    print(f"\nInference Speed Benchmark:")
    print(f"Average batch time: {times.mean():.4f} ± {times.std():.4f} seconds")
    print(f"Average per-image time: {times.mean() / batch_size:.4f} seconds")
    print(f"Throughput: {batch_size / times.mean():.2f} images/second")
    
    return {
        'batch_time_mean': times.mean(),
        'batch_time_std': times.std(),
        'per_image_time': times.mean() / batch_size,
        'throughput': batch_size / times.mean()
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate ATVN model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/atvn_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data directory (optional)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference speed benchmark')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.model_path}...')
    model = load_model_checkpoint(args.model_path, device)
    
    # Create test data loader
    print('Creating test data loader...')
    test_loader = create_test_loader(config, args.test_data)
    print(f'Test batches: {len(test_loader)}')
    
    # Evaluate model
    similarities, labels, predictions, metrics_dict = evaluate_model(
        model, test_loader, device, args.save_predictions
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {metrics_dict['accuracy']:.4f}")
    print(f"Precision: {metrics_dict['precision']:.4f}")
    print(f"Recall: {metrics_dict['recall']:.4f}")
    print(f"F1-Score: {metrics_dict['f1_score']:.4f}")
    print(f"AUC: {metrics_dict['auc']:.4f}")
    print(f"Equal Error Rate (EER): {metrics_dict['eer']:.4f}")
    
    # Generate detailed report
    generate_evaluation_report(metrics_dict, os.path.join(args.output_dir, 'evaluation_report.txt'))
    
    # Plot results
    plot_evaluation_results(similarities, labels, args.output_dir)
    
    # Benchmark inference speed if requested
    if args.benchmark:
        speed_metrics = benchmark_inference_speed(model, test_loader, device)
        
        # Save speed metrics
        speed_path = os.path.join(args.output_dir, 'speed_benchmark.json')
        with open(speed_path, 'w') as f:
            json.dump(speed_metrics, f, indent=2)
        print(f"Speed benchmark results saved to {speed_path}")
    
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
