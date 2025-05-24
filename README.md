# Identical Twin Verification with Attention-Enhanced Feature Fusion

## Overview

This project implements a novel approach for identical twin verification using the ND-TWIN-2009-2010 dataset. The solution combines attention mechanisms for fine-grained discriminative features with pre-trained facial recognition models for robust overall facial representation.

## Key Innovation

Our **Attention-Enhanced Twin Verification Network (ATVN)** introduces:

1. **Multi-Scale Attention Module**: Captures fine-grained discriminative features at multiple scales
2. **Pre-trained Feature Backbone**: Leverages FaceNet/ArcFace for robust facial embeddings
3. **Adaptive Feature Fusion**: Intelligently combines attention-based and pre-trained features
4. **Contrastive Twin Loss**: Specialized loss function for twin verification tasks

## Architecture

```
Input Image (4288x2848) 
    ↓
Preprocessing & Augmentation
    ↓
┌─────────────────┬─────────────────┐
│  Attention      │   Pre-trained   │
│  Branch         │   Branch        │
│                 │   (FaceNet)     │
│ Multi-Scale     │                 │
│ Attention       │   Feature       │
│ Features        │   Extraction    │
└─────────────────┴─────────────────┘
    ↓
Adaptive Feature Fusion
    ↓
Twin Verification Head
    ↓
Similarity Score
```

## Dataset Structure

```
dataset/
├── img_folder_1/
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
├── img_folder_2/
│   └── ...
└── pairs.json  # Contains twin pair mappings
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
```bash
python scripts/prepare_data.py --dataset_path ./dataset --output_path ./processed_data
```

### 2. Training
```bash
python train.py --config configs/atvn_config.yaml --data_path ./processed_data
```

### 3. Evaluation
```bash
python evaluate.py --model_path ./checkpoints/best_model.pth --test_data ./test_data
```

### 4. Inference
```bash
python inference.py --model_path ./checkpoints/best_model.pth --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

## Model Performance

Expected performance metrics:
- **Accuracy**: >95% on twin verification
- **EER**: <3% (Equal Error Rate)
- **AUC**: >0.98

## Key Features

### 1. Multi-Scale Attention Module
- Channel attention for feature importance
- Spatial attention for discriminative regions
- Multi-scale processing for different facial details

### 2. Adaptive Feature Fusion
- Learned weights for combining features
- Context-aware fusion strategy
- Robust to feature distribution differences

### 3. Contrastive Twin Loss
- Specialized for twin verification
- Margin-based learning
- Hard negative mining

## Training Strategy

1. **Phase 1**: Pre-train attention modules on general face recognition
2. **Phase 2**: Fine-tune on twin-specific features with contrastive loss
3. **Phase 3**: End-to-end optimization with adaptive fusion

## Configuration

Edit `configs/atvn_config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Data augmentation settings
- Loss function weights

## Results Visualization

Use the provided visualization tools:
```bash
python visualize_results.py --model_path ./checkpoints/best_model.pth --output_dir ./visualizations
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Citation

```bibtex
@inproceedings{atvn2025,
  title={Attention-Enhanced Twin Verification Network for Identical Twin Recognition},
  author={Your Name},
  booktitle={Conference},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
