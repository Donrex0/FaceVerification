# Face Verification Deep Learning Project

A comprehensive face verification system implemented in PyTorch that compares baseline CNN and improved ResNet50 models using Siamese networks.

## 🎯 Project Overview

This project implements a complete face verification pipeline with progressive experimental workflow, comparing a baseline CNN against an improved ResNet50-based model using transfer learning.

### Key Features
- **Baseline CNN**: Simple convolutional network for face embedding generation
- **Improved Model**: ResNet50 backbone with transfer learning and attention mechanisms
- **Siamese Architecture**: Twin networks for similarity learning
- **Multiple Loss Functions**: Contrastive loss and triplet loss
- **Comprehensive Evaluation**: ROC curves, precision-recall, and detailed metrics
- **Real-time Verification**: Demo script for practical face verification

## 📁 Project Structure

```
face_verification_project/
├── data/                           # Dataset storage
│   └── lfw/                       # LFW dataset files
├── dataset/                        # Data handling modules
│   ├── download_dataset.py         # LFW dataset downloader
│   ├── dataset_loader.py          # PyTorch dataset classes
│   ├── pair_generator.py          # Face pair generation
│   └── preprocessing.py           # Image preprocessing utilities
├── models/                         # Neural network architectures
│   ├── baseline_cnn.py            # Baseline CNN model
│   ├── siamese_network.py         # Siamese network implementation
│   └── improved_model.py          # ResNet50-based improved model
├── training/                       # Training scripts
│   ├── train_baseline.py          # Baseline model training
│   └── train_improved.py          # Improved model training
├── evaluation/                     # Model evaluation and analysis
│   ├── metrics.py                 # Evaluation metrics
│   ├── evaluate_models.py         # Model evaluation script
│   ├── roc_curve.py               # ROC curve generation
│   └── compare_models.py         # Model comparison utilities
├── verification/                   # Real-time verification
│   └── verify_faces.py            # Face verification demo
├── utils/                          # Utility functions
│   ├── image_utils.py             # Image processing utilities
│   └── config.py                 # Configuration management
├── notebooks/                      # Analysis notebooks
│   └── experiment_analysis.ipynb  # Comprehensive analysis notebook
├── experiments/                    # Results and outputs
│   ├── plots/                     # Generated plots and visualizations
│   ├── baseline_results.json      # Baseline model results
│   ├── improved_results.json      # Improved model results
│   └── comparison_table.csv       # Model comparison table
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd face_verification_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

```bash
# Download LFW dataset
python dataset/download_dataset.py
```

### 3. Training Models

```bash
# Train baseline model
python training/train_baseline.py

# Train improved model
python training/train_improved.py
```

### 4. Evaluation

```bash
# Evaluate both models
python evaluation/evaluate_models.py

# Generate comparison plots
python evaluation/roc_curve.py

# Compare models
python evaluation/compare_models.py
```

### 5. Face Verification Demo

```bash
# Verify two face images
python verification/verify_faces.py image1.jpg image2.jpg --visualize

# Use improved model
python verification/verify_faces.py image1.jpg image2.jpg --model improved --threshold 0.7
```

## 📊 Experimental Workflow

### Stage 1: Dataset Preparation
- Automatically downloads LFW dataset using scikit-learn
- Preprocesses images (resize, normalize, tensor conversion)
- Generates positive and negative face pairs
- Creates balanced training and test sets

### Stage 2: Baseline Model
- Simple CNN architecture with 4 convolutional layers
- Siamese network with contrastive loss
- 128-dimensional face embeddings
- 10 epochs training with Adam optimizer

### Stage 3: Improved Model
- ResNet50 backbone with ImageNet pretrained weights
- Transfer learning with fine-tuning
- Attention mechanisms and projection heads
- Triplet loss for better similarity learning
- 15 epochs with differential learning rates

### Stage 4: Evaluation
- Comprehensive metrics: accuracy, precision, recall, F1, ROC AUC
- Threshold optimization using Youden's J statistic
- Embedding space analysis
- Statistical comparison between models

## 🔧 Configuration

The project uses a centralized configuration system in `utils/config.py`:

```python
# Example configuration updates
config_updates = {
    'training': {
        'baseline_epochs': 15,
        'improved_lr': 0.0005
    },
    'data': {
        'batch_size': 64
    }
}
```

## 📈 Expected Results

Based on the experimental design, you should observe:

- **Baseline CNN**: ~75-80% accuracy, ~0.80-0.85 ROC AUC
- **Improved ResNet50**: ~85-92% accuracy, ~0.92-0.96 ROC AUC
- **Training Time**: Baseline ~30 min, Improved ~45-60 min (GPU)
- **Inference Speed**: ~10-20ms per pair (GPU)

## 🧪 Analysis and Visualization

### Jupyter Notebook Analysis
Run the comprehensive analysis notebook:

```bash
jupyter notebooks/experiment_analysis.ipynb
```

The notebook includes:
- Training progress visualization
- Model performance comparison
- ROC curve analysis
- Embedding space exploration
- Error analysis and insights

### Generated Outputs
- Training curves (loss and accuracy)
- ROC curves and precision-recall plots
- Model comparison tables
- Performance improvement charts
- Embedding similarity distributions

## 🎯 Usage Examples

### Face Verification

```python
from verification.verify_faces import FaceVerifier

# Initialize verifier
verifier = FaceVerifier(model_type='improved')

# Verify two faces
result = verifier.verify_faces('person1.jpg', 'person2.jpg')

print(f"Similarity: {result['similarity_score']:.4f}")
print(f"Prediction: {result['result']}")
print(f"Confidence: {result['confidence']}")
```

### Batch Processing

```python
# Verify multiple pairs
image_pairs = [
    ('img1a.jpg', 'img1b.jpg'),
    ('img2a.jpg', 'img2b.jpg'),
    ('img3a.jpg', 'img3b.jpg')
]

results = verifier.batch_verify(image_pairs)
for result in results:
    print(f"Pair {result['pair_index']}: {result['result']}")
```

## 🔬 Technical Details

### Model Architectures

#### Baseline CNN
```
Input (3×160×160)
├── Conv2D(32, 3×3) + ReLU + MaxPool2D(2×2)
├── Conv2D(64, 3×3) + ReLU + MaxPool2D(2×2)
├── Conv2D(128, 3×3) + ReLU + MaxPool2D(2×2)
├── Conv2D(256, 3×3) + ReLU + MaxPool2D(2×2)
├── Flatten
├── FC(512) + ReLU + Dropout(0.5)
├── FC(256) + ReLU + Dropout(0.3)
└── FC(128) + L2 Normalization
```

#### Improved ResNet50
```
Input (3×160×160)
├── ResNet50 Backbone (pretrained)
├── Global Average Pooling
├── FC(1024) + BatchNorm + ReLU + Dropout(0.5)
├── FC(512) + BatchNorm + ReLU + Dropout(0.3)
├── FC(256) + BatchNorm + ReLU + Dropout(0.2)
└── FC(128) + L2 Normalization
```

### Loss Functions

#### Contrastive Loss
```
L = (1-y) * 0.5 * d² + y * 0.5 * max(0, margin-d)²
```

#### Triplet Loss
```
L = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
```

## 📋 Requirements

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only training
- **Recommended**: 16GB RAM, NVIDIA GPU with 4GB+ VRAM
- **Storage**: ~5GB for dataset and models

### Software Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU acceleration)
- See `requirements.txt` for complete list

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config
   python -c "from utils.config import update_config; update_config({'data': {'batch_size': 16}})"
   ```

2. **Dataset Download Fails**
   ```bash
   # Manual LFW download
   pip install scikit-learn[data]
   python dataset/download_dataset.py
   ```

3. **Model Loading Errors**
   ```bash
   # Check checkpoint paths
   ls checkpoints/
   # Re-train if necessary
   python training/train_baseline.py
   ```

### Performance Tips

1. **Use GPU acceleration**: Ensure CUDA is properly installed
2. **Optimize batch size**: Adjust based on available memory
3. **Use mixed precision**: Enable for faster training on compatible GPUs
4. **Preprocess data**: Ensure images are properly normalized

## 📚 Research Context

This project implements state-of-the-art face verification techniques:

- **Siamese Networks**: Learn similarity metrics directly from data
- **Transfer Learning**: Leverage pretrained ImageNet features
- **Metric Learning**: Optimize embedding space for face verification
- **Hard Negative Mining**: Improve training with challenging examples

### Relevant Papers
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering.
- Taigman, Y., Yang, M., Ranzato, M., & Wolf, L. (2014). DeepFace: Closing the gap to human-level performance.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LFW Dataset**: University of Massachusetts, Amherst
- **PyTorch**: Facebook AI Research
- **Scikit-learn**: For dataset utilities
- **OpenCV**: For image processing

## 📞 Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: This project is designed for educational and research purposes. For production use, consider additional security and privacy measures for face recognition systems.
