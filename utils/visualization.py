import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from pathlib import Path


def visualize_attention_maps(
    model: torch.nn.Module,
    image1_path: str,
    image2_path: str,
    transform=None,
    device: str = 'cuda',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12)
):
    """
    Visualize attention maps for a pair of images.
    
    Args:
        model: Trained ATVN model
        image1_path: Path to first image
        image2_path: Path to second image
        transform: Image preprocessing transform
        device: Device to run on
        save_path: Path to save visualization
        figsize: Figure size
    """
    model.eval()
    
    # Load original images for display
    orig_img1 = Image.open(image1_path).convert('RGB')
    orig_img2 = Image.open(image2_path).convert('RGB')
    
    # Preprocess images
    if transform is not None:
        if hasattr(transform, '__call__'):
            # Albumentations transform
            img1 = transform(image=np.array(orig_img1))['image']
            img2 = transform(image=np.array(orig_img2))['image']
        else:
            # torchvision transform
            img1 = transform(orig_img1)
            img2 = transform(orig_img2)
    else:
        # Simple resize and normalize
        img1 = torch.from_numpy(np.array(orig_img1.resize((224, 224)))).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(np.array(orig_img2.resize((224, 224)))).permute(2, 0, 1).float() / 255.0
    
    # Add batch dimension and move to device
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    
    # Get attention maps
    with torch.no_grad():
        attention_info = model.get_attention_maps(img1, img2)
        output = model(img1, img2, return_attention=True)
        similarity = output['similarity'].item()
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    fig.suptitle(f'Twin Attention Visualization (Similarity: {similarity:.3f})', fontsize=16)
    
    # Original images
    axes[0, 0].imshow(orig_img1)
    axes[0, 0].set_title('Image 1', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(orig_img2)
    axes[0, 1].set_title('Image 2', fontsize=12)
    axes[0, 1].axis('off')
    
    # Similarity score visualization
    axes[0, 2].bar(['Twin', 'Non-Twin'], [similarity, 1-similarity], color=['green', 'red'])
    axes[0, 2].set_title('Prediction', fontsize=12)
    axes[0, 2].set_ylim([0, 1])
    
    # Feature difference visualization
    if 'global_diff' in output:
        global_diff = output['global_diff'].cpu().numpy().flatten()
        axes[0, 3].hist(global_diff, bins=50, alpha=0.7)
        axes[0, 3].set_title('Global Feature Differences', fontsize=12)
        axes[0, 3].set_xlabel('Difference Value')
    
    # Multi-scale attention maps
    if 'multiscale_attn1' in attention_info and attention_info['multiscale_attn1']:
        ms_attn1 = attention_info['multiscale_attn1']
        ms_attn2 = attention_info['multiscale_attn2']
        
        for i, (attn1, attn2) in enumerate(zip(ms_attn1[:2], ms_attn2[:2])):  # Show first 2 scales
            # Convert to numpy and resize for visualization
            attn1_np = attn1.squeeze().cpu().numpy()
            attn2_np = attn2.squeeze().cpu().numpy()
            
            if len(attn1_np.shape) == 3:  # Multi-channel attention
                attn1_np = attn1_np.mean(axis=0)
                attn2_np = attn2_np.mean(axis=0)
            
            # Resize to match image size for overlay
            attn1_resized = cv2.resize(attn1_np, (224, 224))
            attn2_resized = cv2.resize(attn2_np, (224, 224))
            
            # Normalize to 0-1
            attn1_resized = (attn1_resized - attn1_resized.min()) / (attn1_resized.max() - attn1_resized.min() + 1e-8)
            attn2_resized = (attn2_resized - attn2_resized.min()) / (attn2_resized.max() - attn2_resized.min() + 1e-8)
            
            # Plot attention maps
            im1 = axes[1, i*2].imshow(attn1_resized, cmap='jet', alpha=0.7)
            axes[1, i*2].set_title(f'Multi-scale Attention 1 (Scale {i+1})', fontsize=10)
            axes[1, i*2].axis('off')
            plt.colorbar(im1, ax=axes[1, i*2], fraction=0.046, pad=0.04)
            
            im2 = axes[1, i*2+1].imshow(attn2_resized, cmap='jet', alpha=0.7)
            axes[1, i*2+1].set_title(f'Multi-scale Attention 2 (Scale {i+1})', fontsize=10)
            axes[1, i*2+1].axis('off')
            plt.colorbar(im2, ax=axes[1, i*2+1], fraction=0.046, pad=0.04)
    
    # Difference attention
    if 'diff_attn' in attention_info:
        diff_attn = attention_info['diff_attn'].squeeze().cpu().numpy()
        
        if len(diff_attn.shape) == 3:  # Multi-channel attention
            diff_attn = diff_attn.mean(axis=0)
        
        # Resize and normalize
        diff_attn_resized = cv2.resize(diff_attn, (224, 224))
        diff_attn_resized = (diff_attn_resized - diff_attn_resized.min()) / (diff_attn_resized.max() - diff_attn_resized.min() + 1e-8)
        
        im3 = axes[2, 0].imshow(diff_attn_resized, cmap='hot', alpha=0.7)
        axes[2, 0].set_title('Difference Attention', fontsize=10)
        axes[2, 0].axis('off')
        plt.colorbar(im3, ax=axes[2, 0], fraction=0.046, pad=0.04)
    
    # Overlay attention on original images
    if 'multiscale_attn1' in attention_info and attention_info['multiscale_attn1']:
        # Use first scale attention for overlay
        attn1 = attention_info['multiscale_attn1'][0].squeeze().cpu().numpy()
        attn2 = attention_info['multiscale_attn2'][0].squeeze().cpu().numpy()
        
        if len(attn1.shape) == 3:
            attn1 = attn1.mean(axis=0)
            attn2 = attn2.mean(axis=0)
        
        # Create overlays
        overlay1 = create_attention_overlay(orig_img1, attn1)
        overlay2 = create_attention_overlay(orig_img2, attn2)
        
        axes[2, 1].imshow(overlay1)
        axes[2, 1].set_title('Image 1 + Attention', fontsize=10)
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(overlay2)
        axes[2, 2].set_title('Image 2 + Attention', fontsize=10)
        axes[2, 2].axis('off')
    
    # Feature similarity heatmap
    if 'embeddings1' in output and 'embeddings2' in output:
        emb1 = output['embeddings1'].cpu().numpy().flatten()
        emb2 = output['embeddings2'].cpu().numpy().flatten()
        
        # Compute element-wise similarity
        similarity_map = emb1 * emb2
        similarity_map = similarity_map.reshape(16, -1)  # Reshape for heatmap
        
        sns.heatmap(similarity_map, ax=axes[2, 3], cmap='coolwarm', center=0)
        axes[2, 3].set_title('Feature Similarity Heatmap', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def create_attention_overlay(image: Image.Image, attention_map: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Create overlay of attention map on original image.
    
    Args:
        image: Original PIL image
        attention_map: Attention map array
        alpha: Transparency of overlay
    Returns:
        Overlayed image as numpy array
    """
    # Resize image to match attention map size initially, then resize both to 224x224
    img_array = np.array(image.resize((224, 224)))
    
    # Resize attention map to match image
    if attention_map.shape != (224, 224):
        attention_map = cv2.resize(attention_map, (224, 224))
    
    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Create heatmap
    heatmap = plt.cm.jet(attention_map)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Blend with original image
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap, alpha, 0)
    
    return overlay


def plot_similarity_heatmap(
    similarities: np.ndarray,
    labels: List[str],
    title: str = "Similarity Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot similarity heatmap between images.
    
    Args:
        similarities: Similarity matrix (N, N)
        labels: List of image labels
        title: Plot title
        save_path: Path to save plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(similarities, 
                xticklabels=labels, 
                yticklabels=labels,
                annot=True, 
                fmt='.3f',
                cmap='coolwarm',
                center=0.5,
                vmin=0, 
                vmax=1)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualize embedding space using dimensionality reduction.
    
    Args:
        embeddings: Embedding vectors (N, embedding_dim)
        labels: Labels for each embedding (N,)
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        save_path: Path to save plot
        figsize: Figure size
    """
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reduce dimensionality
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(reduced_embeddings[:, 0], 
                         reduced_embeddings[:, 1], 
                         c=labels, 
                         cmap='viridis',
                         alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'Embedding Space Visualization ({method.upper()})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_prediction_gallery(
    image_pairs: List[Tuple[str, str]],
    similarities: List[float],
    labels: List[int],
    predictions: List[int],
    num_examples: int = 12,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 15)
):
    """
    Create a gallery of prediction examples.
    
    Args:
        image_pairs: List of image pair paths
        similarities: Similarity scores
        labels: Ground truth labels
        predictions: Model predictions
        num_examples: Number of examples to show
        save_path: Path to save gallery
        figsize: Figure size
    """
    # Select examples (mix of correct and incorrect predictions)
    correct_indices = [i for i, (pred, label) in enumerate(zip(predictions, labels)) if pred == label]
    incorrect_indices = [i for i, (pred, label) in enumerate(zip(predictions, labels)) if pred != label]
    
    # Balance correct and incorrect examples
    num_correct = min(num_examples // 2, len(correct_indices))
    num_incorrect = min(num_examples - num_correct, len(incorrect_indices))
    
    selected_indices = (np.random.choice(correct_indices, num_correct, replace=False).tolist() +
                       np.random.choice(incorrect_indices, num_incorrect, replace=False).tolist())
    
    # Create gallery
    rows = int(np.ceil(len(selected_indices) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(selected_indices):
        row = i // 3
        col = i % 3
        
        if row >= rows:
            break
        
        img1_path, img2_path = image_pairs[idx]
        similarity = similarities[idx]
        label = labels[idx]
        prediction = predictions[idx]
        
        # Load images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Create side-by-side image
        combined_img = Image.new('RGB', (img1.width + img2.width, max(img1.height, img2.height)))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.width, 0))
        
        # Plot
        axes[row, col].imshow(combined_img)
        
        # Create title with prediction info
        status = "✓" if prediction == label else "✗"
        title = f"{status} Sim: {similarity:.3f}\nGT: {'Twin' if label == 1 else 'Non-Twin'}, Pred: {'Twin' if prediction == 1 else 'Non-Twin'}"
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(selected_indices), rows * 3):
        row = i // 3
        col = i % 3
        if row < rows:
            axes[row, col].axis('off')
    
    plt.suptitle('Prediction Gallery', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_feature_importance(
    model: torch.nn.Module,
    image1: torch.Tensor,
    image2: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize feature importance using gradient-based methods.
    
    Args:
        model: Trained model
        image1: First image tensor (1, 3, H, W)
        image2: Second image tensor (1, 3, H, W)
        save_path: Path to save visualization
        figsize: Figure size
    """
    model.eval()
    
    # Enable gradients for input images
    image1.requires_grad_(True)
    image2.requires_grad_(True)
    
    # Forward pass
    output = model(image1, image2)
    similarity = output['similarity']
    
    # Compute gradients
    similarity.backward()
    
    # Get gradients
    grad1 = image1.grad.abs().mean(dim=1, keepdim=True)  # Average across channels
    grad2 = image2.grad.abs().mean(dim=1, keepdim=True)
    
    # Convert to numpy
    grad1_np = grad1.squeeze().detach().cpu().numpy()
    grad2_np = grad2.squeeze().detach().cpu().numpy()
    
    # Normalize
    grad1_np = (grad1_np - grad1_np.min()) / (grad1_np.max() - grad1_np.min() + 1e-8)
    grad2_np = (grad2_np - grad2_np.min()) / (grad2_np.max() - grad2_np.min() + 1e-8)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original images
    img1_np = image1.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img2_np = image2.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    
    # Normalize images for display
    img1_np = (img1_np - img1_np.min()) / (img1_np.max() - img1_np.min())
    img2_np = (img2_np - img2_np.min()) / (img2_np.max() - img2_np.min())
    
    axes[0].imshow(img1_np)
    axes[0].set_title('Image 1')
    axes[0].axis('off')
    
    axes[1].imshow(img2_np)
    axes[1].set_title('Image 2')
    axes[1].axis('off')
    
    # Feature importance (average of both gradients)
    avg_grad = (grad1_np + grad2_np) / 2
    im = axes[2].imshow(avg_grad, cmap='hot')
    axes[2].set_title('Feature Importance')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.suptitle(f'Feature Importance (Similarity: {similarity.item():.3f})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Clean up gradients
    image1.grad = None
    image2.grad = None
