import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import Dict, Any, Tuple
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.atvn import ATVNModel
from data.transforms import get_inference_transforms
from utils.inference import load_model_checkpoint
from utils.visualization import (
    visualize_attention_maps,
    plot_prediction_gallery,
    visualize_embedding_space,
    plot_feature_importance
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_preprocess_image(image_path: str, config: Dict[str, Any]) -> Tuple[np.ndarray, torch.Tensor]:
    """Load and preprocess image for visualization."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get transforms
    transform = get_inference_transforms(tuple(config['data']['image_size']))
    
    # Apply transforms
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0)
    
    return image_rgb, image_tensor


def visualize_twin_pair_attention(
    model,
    image1_path: str,
    image2_path: str,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: str = 'visualizations'
):
    """Visualize attention maps for a twin pair."""
    print(f"Visualizing attention for pair: {image1_path} vs {image2_path}")
    
    # Load and preprocess images
    img1_rgb, img1_tensor = load_and_preprocess_image(image1_path, config)
    img2_rgb, img2_tensor = load_and_preprocess_image(image2_path, config)
    
    # Move to device
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    
    # Get model predictions and attention maps
    model.eval()
    with torch.no_grad():
        outputs = model(img1_tensor, img2_tensor, return_attention=True)
        similarity = outputs['similarity'].item()
        attention_maps = outputs.get('attention_maps', {})
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize attention maps
    if attention_maps:
        fig = visualize_attention_maps(
            img1_rgb, img2_rgb, attention_maps, similarity
        )
        
        # Save attention visualization
        attention_path = os.path.join(output_dir, 'attention_maps.png')
        fig.savefig(attention_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Attention maps saved to {attention_path}")
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display images
    axes[0].imshow(img1_rgb)
    axes[0].set_title(f'Image 1\n{os.path.basename(image1_path)}')
    axes[0].axis('off')
    
    axes[1].imshow(img2_rgb)
    axes[1].set_title(f'Image 2\n{os.path.basename(image2_path)}')
    axes[1].axis('off')
    
    # Add similarity score
    prediction = "Twins" if similarity > 0.5 else "Not Twins"
    confidence = abs(similarity - 0.5) * 2
    
    fig.suptitle(
        f'Twin Verification Result\n'
        f'Similarity: {similarity:.4f} | Prediction: {prediction} | Confidence: {confidence:.4f}',
        fontsize=14
    )
    
    plt.tight_layout()
    
    # Save comparison
    comparison_path = os.path.join(output_dir, 'twin_comparison.png')
    fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison saved to {comparison_path}")
    
    return similarity, attention_maps


def visualize_model_features(
    model,
    image_paths: list,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: str = 'visualizations'
):
    """Visualize model features and embeddings."""
    print(f"Extracting features from {len(image_paths)} images...")
    
    # Extract embeddings for all images
    embeddings = []
    image_names = []
    
    model.eval()
    with torch.no_grad():
        for image_path in image_paths:
            try:
                _, image_tensor = load_and_preprocess_image(image_path, config)
                image_tensor = image_tensor.to(device)
                
                # Extract embedding
                embedding = model.get_embeddings(image_tensor)
                embeddings.append(embedding.cpu().numpy().flatten())
                image_names.append(os.path.basename(image_path))
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
    
    if len(embeddings) == 0:
        print("No valid embeddings extracted")
        return
    
    embeddings = np.array(embeddings)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize embedding space
    print("Creating embedding space visualization...")
    fig = visualize_embedding_space(embeddings, image_names)
    
    embedding_path = os.path.join(output_dir, 'embedding_space.png')
    fig.savefig(embedding_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Embedding space visualization saved to {embedding_path}")
    
    # Compute and visualize similarity matrix
    print("Creating similarity matrix...")
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        square=True,
        ax=ax
    )
    ax.set_title('Embedding Similarity Matrix')
    ax.set_xlabel('Images')
    ax.set_ylabel('Images')
    
    # Set tick labels
    if len(image_names) <= 20:  # Only show names if not too many
        ax.set_xticklabels(image_names, rotation=45, ha='right')
        ax.set_yticklabels(image_names, rotation=0)
    
    plt.tight_layout()
    
    similarity_path = os.path.join(output_dir, 'similarity_matrix.png')
    fig.savefig(similarity_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Similarity matrix saved to {similarity_path}")


def create_model_architecture_diagram(
    model,
    output_dir: str = 'visualizations'
):
    """Create a diagram of the model architecture."""
    print("Creating model architecture diagram...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model structure
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Collect model information
    model_info = {
        'Total Parameters': count_parameters(model),
        'Backbone': model.config['model']['backbone'],
        'Embedding Dim': model.config['model']['embedding_dim'],
        'Attention Dim': model.config['model']['attention_dim'],
        'Attention Heads': model.config['model']['num_attention_heads']
    }
    
    # Create architecture summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Model components
    components = [
        'Input Images',
        'Backbone\n(FaceNet/ArcFace)',
        'Feature Extractor',
        'Twin Attention\nModule',
        'Feature Fusion',
        'Similarity Head',
        'Output'
    ]
    
    # Draw architecture flow
    y_positions = np.linspace(0.8, 0.2, len(components))
    x_positions = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95]
    
    for i, (comp, x, y) in enumerate(zip(components, x_positions, y_positions)):
        # Draw component box
        bbox = dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7)
        ax.text(x, y, comp, transform=ax.transAxes, ha='center', va='center',
                bbox=bbox, fontsize=10, weight='bold')
        
        # Draw arrow to next component
        if i < len(components) - 1:
            ax.annotate('', xy=(x_positions[i+1]-0.05, y), 
                       xytext=(x_positions[i]+0.05, y),
                       transform=ax.transAxes,
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Add model information text
    info_text = '\n'.join([f'{k}: {v}' for k, v in model_info.items()])
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    ax.set_title('ATVN Model Architecture', fontsize=16, weight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save architecture diagram
    arch_path = os.path.join(output_dir, 'model_architecture.png')
    fig.savefig(arch_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Model architecture diagram saved to {arch_path}")


def generate_visualization_report(
    similarity: float,
    attention_maps: Dict[str, Any],
    output_dir: str = 'visualizations'
):
    """Generate a text report of the visualization results."""
    report_path = os.path.join(output_dir, 'visualization_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("ATVN Model Visualization Report\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Similarity Score: {similarity:.4f}\n")
        f.write(f"Prediction: {'Twins' if similarity > 0.5 else 'Not Twins'}\n")
        f.write(f"Confidence: {abs(similarity - 0.5) * 2:.4f}\n\n")
        
        if attention_maps:
            f.write("Attention Maps Available:\n")
            f.write("-" * 25 + "\n")
            for key in attention_maps.keys():
                f.write(f"- {key}\n")
        else:
            f.write("No attention maps available\n")
        
        f.write(f"\nVisualization files generated:\n")
        f.write("- attention_maps.png\n")
        f.write("- twin_comparison.png\n")
        f.write("- embedding_space.png\n")
        f.write("- similarity_matrix.png\n")
        f.write("- model_architecture.png\n")
    
    print(f"Visualization report saved to {report_path}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize ATVN model predictions and attention')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/atvn_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--image_pair', type=str, nargs=2, metavar=('IMG1', 'IMG2'),
                       help='Pair of images to visualize')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory with images for feature visualization')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--architecture', action='store_true',
                       help='Create model architecture diagram')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.model_path}...')
    model = load_model_checkpoint(args.model_path, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    similarity = None
    attention_maps = {}
    
    # Visualize twin pair if provided
    if args.image_pair:
        similarity, attention_maps = visualize_twin_pair_attention(
            model, args.image_pair[0], args.image_pair[1], 
            config, device, args.output_dir
        )
    
    # Visualize features from directory if provided
    if args.image_dir:
        # Get all image files from directory
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(args.image_dir).glob(f'*{ext}'))
            image_paths.extend(Path(args.image_dir).glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        if image_paths:
            # Limit to first 50 images for performance
            if len(image_paths) > 50:
                print(f"Found {len(image_paths)} images, using first 50 for visualization")
                image_paths = image_paths[:50]
            
            visualize_model_features(
                model, image_paths, config, device, args.output_dir
            )
        else:
            print(f"No images found in {args.image_dir}")
    
    # Create architecture diagram if requested
    if args.architecture:
        create_model_architecture_diagram(model, args.output_dir)
    
    # Generate report
    if similarity is not None:
        generate_visualization_report(similarity, attention_maps, args.output_dir)
    
    print(f"\nAll visualizations saved to {args.output_dir}/")
    
    # Print summary
    print("\nVisualization Summary:")
    if args.image_pair:
        print(f"- Twin pair attention visualization")
        print(f"- Similarity: {similarity:.4f}")
    if args.image_dir:
        print(f"- Feature space visualization")
    if args.architecture:
        print(f"- Model architecture diagram")


if __name__ == '__main__':
    main()
