import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def verify_twin_pair(
    model: torch.nn.Module,
    image1_path: str,
    image2_path: str,
    transform=None,
    device: str = 'cuda'
) -> float:
    """
    Verify if two images are of identical twins.
    
    Args:
        model: Trained ATVN model
        image1_path: Path to first image
        image2_path: Path to second image
        transform: Image preprocessing transform
        device: Device to run inference on
    Returns:
        Similarity score between 0 and 1
    """
    model.eval()
    
    # Load and preprocess images
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')
    
    if transform is not None:
        if hasattr(transform, '__call__'):
            # Albumentations transform
            image1 = transform(image=np.array(image1))['image']
            image2 = transform(image=np.array(image2))['image']
        else:
            # torchvision transform
            image1 = transform(image1)
            image2 = transform(image2)
    
    # Add batch dimension
    image1 = image1.unsqueeze(0).to(device)
    image2 = image2.unsqueeze(0).to(device)
    
    # Compute similarity
    with torch.no_grad():
        similarity = model.compute_similarity(image1, image2)
    
    return similarity.item()


def batch_inference(
    model: torch.nn.Module,
    image_pairs: List[Tuple[str, str]],
    transform=None,
    device: str = 'cuda',
    batch_size: int = 16
) -> List[float]:
    """
    Perform batch inference on multiple image pairs.
    
    Args:
        model: Trained ATVN model
        image_pairs: List of (image1_path, image2_path) tuples
        transform: Image preprocessing transform
        device: Device to run inference on
        batch_size: Batch size for inference
    Returns:
        List of similarity scores
    """
    model.eval()
    similarities = []
    
    for i in range(0, len(image_pairs), batch_size):
        batch_pairs = image_pairs[i:i + batch_size]
        batch_images1 = []
        batch_images2 = []
        
        # Load and preprocess batch
        for img1_path, img2_path in batch_pairs:
            image1 = Image.open(img1_path).convert('RGB')
            image2 = Image.open(img2_path).convert('RGB')
            
            if transform is not None:
                if hasattr(transform, '__call__'):
                    image1 = transform(image=np.array(image1))['image']
                    image2 = transform(image=np.array(image2))['image']
                else:
                    image1 = transform(image1)
                    image2 = transform(image2)
            
            batch_images1.append(image1)
            batch_images2.append(image2)
        
        # Stack and move to device
        batch_images1 = torch.stack(batch_images1).to(device)
        batch_images2 = torch.stack(batch_images2).to(device)
        
        # Compute similarities
        with torch.no_grad():
            batch_similarities = model.compute_similarity(batch_images1, batch_images2)
        
        similarities.extend(batch_similarities.cpu().numpy().tolist())
    
    return similarities


def extract_embeddings(
    model: torch.nn.Module,
    image_paths: List[str],
    transform=None,
    device: str = 'cuda',
    batch_size: int = 32
) -> np.ndarray:
    """
    Extract face embeddings for a list of images.
    
    Args:
        model: Trained ATVN model
        image_paths: List of image paths
        transform: Image preprocessing transform
        device: Device to run inference on
        batch_size: Batch size for inference
    Returns:
        Array of embeddings (N, embedding_dim)
    """
    model.eval()
    embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        # Load and preprocess batch
        for img_path in batch_paths:
            image = Image.open(img_path).convert('RGB')
            
            if transform is not None:
                if hasattr(transform, '__call__'):
                    image = transform(image=np.array(image))['image']
                else:
                    image = transform(image)
            
            batch_images.append(image)
        
        # Stack and move to device
        batch_images = torch.stack(batch_images).to(device)
        
        # Extract embeddings
        with torch.no_grad():
            batch_embeddings = model.get_embeddings(batch_images)
        
        embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)


def find_most_similar(
    query_embedding: np.ndarray,
    database_embeddings: np.ndarray,
    image_paths: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find most similar images to a query embedding.
    
    Args:
        query_embedding: Query embedding (embedding_dim,)
        database_embeddings: Database embeddings (N, embedding_dim)
        image_paths: List of image paths corresponding to database embeddings
        top_k: Number of top similar images to return
    Returns:
        List of (image_path, similarity_score) tuples
    """
    # Compute cosine similarities
    query_norm = np.linalg.norm(query_embedding)
    db_norms = np.linalg.norm(database_embeddings, axis=1)
    
    similarities = np.dot(database_embeddings, query_embedding) / (db_norms * query_norm)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return results
    results = []
    for idx in top_indices:
        results.append((image_paths[idx], similarities[idx]))
    
    return results


def create_test_pairs_from_json(
    pairs_json_path: str,
    dataset_root: str,
    num_positive_pairs: int = 100,
    num_negative_pairs: int = 100
) -> List[Tuple[str, str, int]]:
    """
    Create test pairs from JSON file.
    
    Args:
        pairs_json_path: Path to pairs.json file
        dataset_root: Root directory of dataset
        num_positive_pairs: Number of positive pairs to create
        num_negative_pairs: Number of negative pairs to create
    Returns:
        List of (image1_path, image2_path, label) tuples
    """
    import json
    import os
    import random
    
    # Load twin pairs
    with open(pairs_json_path, 'r') as f:
        twin_pairs = json.load(f)
    
    # Get all image folders and their images
    folder_images = {}
    for folder in os.listdir(dataset_root):
        folder_path = os.path.join(dataset_root, folder)
        if os.path.isdir(folder_path):
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                folder_images[folder] = images
    
    test_pairs = []
    
    # Create positive pairs (twins)
    for _ in range(num_positive_pairs):
        if not twin_pairs:
            break
        
        twin_pair = random.choice(twin_pairs)
        folder1, folder2 = twin_pair
        
        if folder1 in folder_images and folder2 in folder_images:
            img1 = random.choice(folder_images[folder1])
            img2 = random.choice(folder_images[folder2])
            
            path1 = os.path.join(dataset_root, folder1, img1)
            path2 = os.path.join(dataset_root, folder2, img2)
            
            test_pairs.append((path1, path2, 1))
    
    # Create negative pairs (non-twins)
    all_folders = list(folder_images.keys())
    twin_folders = set()
    for twin_pair in twin_pairs:
        twin_folders.update(twin_pair)
    
    for _ in range(num_negative_pairs):
        if len(all_folders) < 2:
            break
        
        folder1, folder2 = random.sample(all_folders, 2)
        
        # Ensure they're not twins
        is_twin_pair = any(set([folder1, folder2]) == set(twin_pair) 
                         for twin_pair in twin_pairs)
        if not is_twin_pair:
            img1 = random.choice(folder_images[folder1])
            img2 = random.choice(folder_images[folder2])
            
            path1 = os.path.join(dataset_root, folder1, img1)
            path2 = os.path.join(dataset_root, folder2, img2)
            
            test_pairs.append((path1, path2, 0))
    
    # Shuffle pairs
    random.shuffle(test_pairs)
    
    return test_pairs


def save_inference_results(
    image_pairs: List[Tuple[str, str]],
    similarities: List[float],
    labels: List[int],
    output_path: str,
    threshold: float = 0.5
):
    """
    Save inference results to a CSV file.
    
    Args:
        image_pairs: List of image pair paths
        similarities: List of similarity scores
        labels: List of ground truth labels
        output_path: Path to save CSV file
        threshold: Decision threshold
    """
    import pandas as pd
    
    # Create predictions
    predictions = [1 if sim >= threshold else 0 for sim in similarities]
    
    # Create DataFrame
    results = []
    for i, ((img1, img2), sim, pred, label) in enumerate(zip(image_pairs, similarities, predictions, labels)):
        results.append({
            'image1': img1,
            'image2': img2,
            'similarity': sim,
            'prediction': pred,
            'ground_truth': label,
            'correct': pred == label
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")
    print(f"Accuracy: {df['correct'].mean():.3f}")


def load_model_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    Returns:
        Loaded model
    """
    from models.atvn import ATVNModel
    
    model, epoch = ATVNModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {epoch}")
    
    return model


def preprocess_image_for_inference(image_path: str, target_size: Tuple[int, int] = (224, 224)):
    """
    Preprocess a single image for inference.
    
    Args:
        image_path: Path to image
        target_size: Target image size (H, W)
    Returns:
        Preprocessed image tensor
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Apply transform
    transformed = transform(image=image)
    image_tensor = transformed['image']
    
    return image_tensor


def benchmark_model_speed(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    num_iterations: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        input_size: Input tensor size (B, C, H, W)
        num_iterations: Number of iterations to run
        device: Device to run on
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    model.eval()
    
    # Create dummy input
    dummy_input1 = torch.randn(input_size).to(device)
    dummy_input2 = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.compute_similarity(dummy_input1, dummy_input2)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            _ = model.compute_similarity(dummy_input1, dummy_input2)
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': times.mean(),
        'std_time': times.std(),
        'min_time': times.min(),
        'max_time': times.max(),
        'fps': 1.0 / times.mean()
    }
