# ML Module for Vastu-AI Floor Plan Generation

This module adds machine learning capabilities to generate more realistic floor plans based on real architectural patterns learned from the CubiCasa5k dataset.

## Directory Structure

```
ml/
├── data/                       # Data loading and preprocessing
│   ├── cubicasa_loader.py     # CubiCasa5k dataset loader
│   ├── preprocessor.py        # Convert floor plans → graphs
│   └── augmentation.py        # Data augmentation
│
├── models/                     # Neural network models
│   ├── graph_encoder.py       # Layout ↔ Graph conversion
│   ├── gnn_refiner.py         # GNN layout refinement model
│   └── (pix2pix_renderer.py)  # Visual renderer (Phase 3)
│
├── training/                   # Training scripts and utilities
│   ├── train_gnn.py           # GNN training script
│   ├── losses.py              # Custom loss functions
│   ├── metrics.py             # Evaluation metrics
│   └── config.py              # Training configuration
│
├── utils/                      # Utilities
│   ├── checkpointing.py       # Model checkpointing
│   └── visualization.py       # Training visualization
│
└── inference.py               # Production inference wrappers
```

## Components

### 1. Data Pipeline

**CubiCasaDataset** (`data/cubicasa_loader.py`)
- Loads SVG floor plans from CubiCasa5k
- Parses rooms, walls, doors, windows
- Maps room types to our schema
- Filters residential plans

**FloorPlanPreprocessor** (`data/preprocessor.py`)
- Converts floor plans to graph representations
- Nodes = rooms (features: type, area, zone)
- Edges = adjacency relationships
- Normalizes coordinates

**FloorPlanAugmentation** (`data/augmentation.py`)
- Flipping, rotation, jitter
- Improves model generalization

### 2. Models

**GraphEncoder** (`models/graph_encoder.py`)
- Converts Layout → Graph for GNN processing
- Converts Graph → Layout for output
- Handles Vastu zones and room types

**GraphAttentionRefiner** (`models/gnn_refiner.py`)
- 3-layer Graph Attention Network
- Input: Constraint-solver layout
- Output: Refined positions and dimensions
- Preserves constraints while adding realism

### 3. Training

**train_gnn.py** (`training/train_gnn.py`)
- Main training script
- Handles data loading, training loop, validation
- Saves best model and checkpoints

**LayoutRefinementLoss** (`training/losses.py`)
- Multi-objective loss:
  - Position refinement (match real layouts)
  - Overlap penalty (keep rooms separate)
  - Area preservation (maintain total area)
  - Adjacency preservation (keep connections)
  - Vastu compliance (optional)

**LayoutMetrics** (`training/metrics.py`)
- Position accuracy (MSE, MAE)
- Overlap percentage
- Area preservation error
- Room size deviation

### 4. Inference

**GNNRefinerInference** (`inference.py`)
- Production-ready inference wrapper
- Loads trained models
- Graceful fallback if model unavailable
- Integrates with existing FloorPlanGenerator

## Usage

### Training

```python
# Load data
from ml.data.cubicasa_loader import CubiCasaDataset
from ml.data.preprocessor import FloorPlanPreprocessor

dataset = CubiCasaDataset("data/cubicasa5k", split="train")
preprocessor = FloorPlanPreprocessor(normalize_coords=True)

# Create model
from ml.models.gnn_refiner import GraphAttentionRefiner

model = GraphAttentionRefiner(
    node_features_dim=21,
    edge_features_dim=4,
    hidden_dim=128,
    num_gat_layers=3,
)

# Train
python backend/ml/training/train_gnn.py --data_dir data/cubicasa5k --epochs 50
```

### Inference

```python
from ml.inference import GNNRefinerInference

# Create refiner
refiner = GNNRefinerInference(
    model_path="ml/models/saved/best_model.pt",
    device="cpu",
)

# Refine layout
refined_layout = refiner.refine(constraint_layout)
```

### Integration with Generator

```python
# In FloorPlanGenerator
from ml.inference import GNNRefinerInference

class FloorPlanGenerator:
    def __init__(self):
        # ... existing init ...
        self.gnn_refiner = GNNRefinerInference(
            model_path=settings.gnn_model_path,
            enable=settings.use_gnn_refiner,
        )

    def generate(self, request):
        # ... constraint solving ...
        layout = self.constraint_solver.solve(...)

        # Refine with GNN
        if self.gnn_refiner:
            layout = self.gnn_refiner.refine(layout)

        # ... continue pipeline ...
```

## Model Architecture

### Graph Attention Network (GAT)

```
Input Graph
  ↓
Node Encoder (Linear: 21 → 128)
  ↓
GAT Layer 1 (128 → 128×4, 4 heads)
  ↓
Layer Norm + ReLU + Dropout
  ↓
GAT Layer 2 (128×4 → 128×4, 4 heads)
  ↓
Layer Norm + ReLU + Dropout
  ↓
GAT Layer 3 (128×4 → 128, 1 head)
  ↓
Layer Norm + ReLU
  ↓
Position Decoder (MLP: 128 → 64 → 4)
  ↓
Tanh (bounded adjustments)
  ↓
Output: Δx, Δy, Δw, Δh (refinements)
```

### Graph Structure

**Nodes** (rooms):
- Features: [room_type (11-dim one-hot), area (1-dim), zone (9-dim one-hot)]
- Total: 21 dimensions

**Edges** (adjacency):
- Features: [distance, dx, dy, adjacency_flag]
- Total: 4 dimensions

## Configuration

Edit `configs/gnn_config.yaml`:

```yaml
model:
  hidden_dim: 128          # Hidden layer size
  num_gat_layers: 3        # Number of GAT layers
  num_heads: 4             # Attention heads

training:
  num_epochs: 50           # Training epochs
  learning_rate: 0.001     # Learning rate
  batch_size: 8            # Batch size

loss:
  overlap_weight: 5.0      # Penalize overlaps heavily
  use_vastu_loss: false    # Include Vastu compliance
```

## Performance

- **Model size**: ~2 MB
- **Parameters**: ~500K
- **Inference time**: <1 second (CPU)
- **Training time**: 2-3 hours for 50 epochs (CPU)

## Expected Results

After training on CubiCasa5k:
- Position accuracy: <3% deviation
- Zero room overlaps (enforced)
- Vastu compliance: >90%
- More natural room arrangements

## Dependencies

See `requirements-ml.txt`:
- torch >= 2.1.0
- torch-geometric >= 2.4.0
- numpy, PIL, opencv
- wandb, tensorboard (optional, for logging)

## Future Work

### Phase 3: Visual Renderer
- Pix2Pix or Diffusion model
- Input: Simple layout (colored boxes)
- Output: Photorealistic floor plan image
- Includes: walls, textures, symbols, annotations

### Potential Improvements
- Fine-tune on Indian floor plans
- Add room relationship constraints
- Multi-floor support
- Style transfer for different architectural styles

## References

- **CubiCasa5k**: https://github.com/CubiCasa/CubiCasa5k
- **Graph Attention Networks**: Veličković et al., ICLR 2018
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

## License

Same as main Vastu-AI project. CubiCasa5k dataset: CC BY 4.0.
