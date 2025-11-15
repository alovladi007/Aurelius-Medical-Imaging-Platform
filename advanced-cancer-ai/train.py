"""
Main Training Script for Advanced Cancer Detection AI
Supports both synthetic and real medical imaging data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from pathlib import Path
import json
import argparse
from datetime import datetime

# Import our modules
from src.models.multimodal_cancer_detector import create_model
from src.training.trainer import create_trainer
from src.evaluation.metrics import CancerDetectionMetrics
from src.data.data_manager import DataManager
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.visualization import plot_training_history

def create_synthetic_multimodal_data(num_samples: int = 1000):
    """Create synthetic multimodal data for testing"""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating {num_samples} synthetic multimodal samples...")

    data = {
        'image': torch.randn(num_samples, 3, 224, 224),
        'clinical': torch.randn(num_samples, 10),
        'genomic': torch.randn(num_samples, 1000, 5),
        'cancer_type': torch.randint(0, 4, (num_samples,)),
        'cancer_stage': torch.randint(0, 5, (num_samples,)),
        'risk_score': torch.rand(num_samples)
    }

    return data


def collate_fn(batch):
    """Custom collate function for multimodal data"""
    if isinstance(batch[0], dict):
        # Multimodal data
        keys = batch[0].keys()
        collated = {}
        for key in keys:
            collated[key] = torch.stack([item[key] for item in batch])
        return collated
    else:
        # Simple tensors
        return torch.utils.data.dataloader.default_collate(batch)


def train_with_real_data(args, config):
    """Train model with real medical imaging data"""
    logger = logging.getLogger(__name__)

    # Initialize data manager
    logger.info("Initializing data manager...")
    data_manager = DataManager(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        image_size=tuple(config.model.image_size),
        train_val_split=config.data.train_val_split,
        augmentation=config.data.get('augmentation', {}).get('enabled', True),
        multimodal=config.data.multimodal,
        dataset_type=config.data.dataset_type
    )

    # Get data loaders
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = data_manager.get_data_loaders()

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating model...")
    model_config = {
        'num_classes': config.model.num_classes,
        'num_stages': config.model.num_stages,
        'image_encoder_params': {
            'vit_model': config.model.vision_model,
            'efficientnet_model': config.model.efficientnet_model,
            'num_classes': config.model.image_embedding_dim
        },
        'clinical_encoder_params': {
            'input_dim': config.model.clinical_input_dim,
            'hidden_dims': config.model.clinical_hidden_dims,
            'output_dim': config.model.clinical_embedding_dim
        },
        'genomic_encoder_params': {
            'sequence_length': config.model.genomic_sequence_length,
            'embedding_dim': config.model.genomic_embedding_dim
        },
        'fusion_params': {
            'image_dim': config.model.image_embedding_dim,
            'clinical_dim': config.model.clinical_embedding_dim,
            'genomic_dim': config.model.genomic_embedding_dim,
            'fusion_dim': config.model.fusion_dim
        },
        'dropout': config.model.dropout
    }

    model = create_model(model_config)

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() and config.deployment.use_gpu else 'cpu')
    model = model.to(device)
    logger.info(f"Using device: {device}")

    # Create trainer config
    trainer_config = {
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.weight_decay,
        'scheduler': config.training.scheduler,
        'warmup_epochs': config.training.warmup_epochs,
        'min_lr': config.training.min_lr,
        'detection_weight': config.training.detection_weight,
        'staging_weight': config.training.staging_weight,
        'risk_weight': config.training.risk_weight,
        'use_focal_loss': config.training.use_focal_loss,
        'focal_alpha': config.training.focal_alpha,
        'focal_gamma': config.training.focal_gamma,
        'gradient_clip': config.training.gradient_clip,
        'patience': config.training.early_stopping_patience,
        'min_delta': config.training.early_stopping_min_delta,
        'checkpoint_dir': config.training.checkpoint_dir
    }

    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(model, train_loader, val_loader, trainer_config)

    # Train model
    logger.info(f"Starting training for {config.training.num_epochs} epochs...")
    history = trainer.train(config.training.num_epochs)

    # Save training history
    history_file = Path(config.logging.get('log_dir', './logs')) / 'training_history.json'
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_file}")

    # Plot training history
    if config.evaluation.get('save_visualizations', True):
        viz_dir = Path(config.evaluation.get('visualization_dir', './visualizations'))
        viz_dir.mkdir(parents=True, exist_ok=True)
        plot_path = viz_dir / 'training_history.png'
        plot_training_history(history, save_path=str(plot_path))
        logger.info(f"Training plots saved to {plot_path}")

    # Evaluate on test set if available
    if test_loader:
        logger.info("Evaluating on test set...")
        evaluate_model(model, test_loader, device, config)

    return model, history


def train_with_synthetic_data(args, config):
    """Train model with synthetic data for testing"""
    logger = logging.getLogger(__name__)
    logger.info("Running in test mode with synthetic data")

    # Generate synthetic data
    train_data = create_synthetic_multimodal_data(800)
    val_data = create_synthetic_multimodal_data(200)
    test_data = create_synthetic_multimodal_data(100)

    # Create datasets
    from torch.utils.data import TensorDataset

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, data_dict):
            self.data = data_dict
            self.length = len(data_dict['cancer_type'])

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.data.items()}

    train_dataset = DictDataset(train_data)
    val_dataset = DictDataset(val_data)
    test_dataset = DictDataset(test_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create model
    logger.info("Creating model...")
    model_config = {
        'num_classes': 4,
        'num_stages': 5,
        'dropout': 0.3
    }
    model = create_model(model_config)

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Using device: {device}")

    # Create trainer config
    trainer_config = {
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'scheduler': 'cosine',
        'warmup_epochs': 2,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'patience': 5,
        'checkpoint_dir': './checkpoints'
    }

    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(model, train_loader, val_loader, trainer_config)

    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(args.epochs)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    metrics = evaluate_model(model, test_loader, device, config)

    # Save model
    save_dir = Path('./checkpoints')
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / 'synthetic_final_model.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    return model, history


def evaluate_model(model, test_loader, device, config):
    """Evaluate model on test set"""
    logger = logging.getLogger(__name__)

    model.eval()
    metrics_calculator = CancerDetectionMetrics(
        num_classes=4,
        class_names=['Lung Cancer', 'Breast Cancer', 'Prostate Cancer', 'Colorectal Cancer']
    )

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                # Multimodal data
                images = batch['image'].to(device)
                clinical = batch.get('clinical')
                genomic = batch.get('genomic')
                labels = batch['cancer_type'].to(device)

                if clinical is not None:
                    clinical = clinical.to(device)
                if genomic is not None:
                    genomic = genomic.to(device)

                # Forward pass
                outputs = model(images, clinical, genomic)
                predictions = outputs['cancer_type'].argmax(dim=1)
                probabilities = torch.softmax(outputs['cancer_type'], dim=1)

                # Update metrics
                metrics_calculator.update(
                    predictions=predictions,
                    labels=labels,
                    probabilities=probabilities,
                    stages=batch.get('cancer_stage'),
                    stage_preds=outputs.get('cancer_stage', torch.zeros_like(labels)),
                    risks=batch.get('risk_score'),
                    risk_preds=outputs.get('risk_score')
                )
            else:
                # Simple image data
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                predictions = outputs['cancer_type'].argmax(dim=1)
                probabilities = torch.softmax(outputs['cancer_type'], dim=1)

                metrics_calculator.update(predictions=predictions, labels=labels,
                                        probabilities=probabilities)

    # Compute all metrics
    metrics = metrics_calculator.compute_all_metrics()

    # Print summary
    metrics_calculator.print_summary()

    # Save visualizations if configured
    if config.evaluation.get('save_visualizations', True):
        viz_dir = Path(config.evaluation.get('visualization_dir', './visualizations'))
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        cm_path = viz_dir / 'confusion_matrix.png'
        metrics_calculator.plot_confusion_matrix(save_path=str(cm_path))

        # ROC curves
        roc_path = viz_dir / 'roc_curves.png'
        metrics_calculator.plot_roc_curves(save_path=str(roc_path))

        logger.info(f"Visualizations saved to {viz_dir}")

    # Save metrics
    metrics_file = Path(config.logging.get('log_dir', './logs')) / 'test_metrics.json'
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy values to float for JSON serialization
    metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in metrics.items() if k != 'confusion_matrix'}

    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Advanced Cancer Detection AI')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode with synthetic data')
    parser.add_argument('--data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr

    # Setup logging
    log_dir = Path(config.logging.get('log_dir', './logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name='cancer_detection',
        log_dir=str(log_dir),
        log_level=config.logging.get('level', 'INFO'),
        console=True,
        file_logging=True
    )

    logger.info("="*80)
    logger.info("Advanced Cancer Detection AI - Training")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Print configuration
    logger.info(str(config))

    # Create directories
    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.evaluation.get('visualization_dir', './visualizations')).mkdir(
        parents=True, exist_ok=True
    )

    # Train model
    if args.test_mode:
        model, history = train_with_synthetic_data(args, config)
    else:
        model, history = train_with_real_data(args, config)

    logger.info("")
    logger.info("="*80)
    logger.info("Training completed successfully!")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
