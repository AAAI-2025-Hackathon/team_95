import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import NextDayFireDataset
import numpy as np
import gc
import random
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Directory for saving checkpoints
dir_checkpoint = Path('checkpoints')
def seed_all(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # # Set global random seed
    # tf.random.set_seed(seed)
    # # Set operation-level random seeds
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'

def train_next_day_fire(
    train_data,
    val_data,
    model,
    device,
    dir_checkpoint: Path = Path('checkpoints'),
    starting_epoch: int = 1,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    pos_weight: float = 3.0,
    limit_features: list = None,
    max_train_samples: int = 10000,
    max_val_samples: int = 500,
):
    """
    Train the ResNetUNet model for next-day fire prediction.

    Args:
        train_data: Training dataset.
        val_data: Validation dataset.
        model: ResNetUNet model.
        device: Device to use for training (e.g., 'cuda' or 'cpu').
        dir_checkpoint: Directory to save checkpoints.
        starting_epoch: Starting epoch number.
        epochs: Number of epochs to train.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        val_percent: Percentage of data to use for validation.
        save_checkpoint: Whether to save checkpoints.
        img_scale: Scaling factor for images.
        pos_weight: Weight for positive class in BCE loss.
        limit_features: List of features to use.
    """
    # Create datasets
    train_set = NextDayFireDataset(
        train_data,
        limit_features_list=limit_features,
        clip_normalize=True,
        sampling_method='original',
        max_samples=max_train_samples
    )
    val_set = NextDayFireDataset(
        val_data,
        limit_features_list=limit_features,
        clip_normalize=True,
        sampling_method='original',
        max_samples=max_val_samples
    )

    n_train = len(train_set)
    n_val = len(val_set)

    # Create data loaders
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=batch_size,
    )

    logging.info(
        f'''Starting training:
        Starting epoch:  {starting_epoch}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Positive weight: {pos_weight}
        Input features:  {limit_features}
    '''
    )

    # Set up the optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    global_step = 0

    # Training loop
    for epoch in range(starting_epoch, epochs + 1):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                # Forward pass
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(images.shape[0])
                global_step += 1

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                if global_step % (n_train // (5 * batch_size)) == 0:
                    val_loss, val_dice = evaluate(
                        model, val_loader, device, criterion
                    )
                    logging.info(
                        f'Validation Loss: {val_loss}, Validation Dice: {val_dice}'
                    )

        # Save checkpoint
        if save_checkpoint and epoch % 10 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss / len(train_loader),
                },
                f'{dir_checkpoint}/{model.__class__.__name__}_checkpoint_epoch{epoch}.pth',
            )
            logging.info(f'Checkpoint {epoch} saved!')


@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    """
    Evaluate the model on the validation set.

    Args:
        model: ResNetUNet model.
        dataloader: Validation data loader.
        device: Device to use for evaluation.
        criterion: Loss function.

    Returns:
        val_loss: Average validation loss.
        val_dice: Average validation Dice score.
    """
    model.eval()
    val_loss = 0
    val_dice = 0

    for batch in dataloader:
        images, true_masks = batch

        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        # Forward pass
        masks_pred = model(images)
        loss = criterion(masks_pred, true_masks)
        val_loss += loss.item()

        # Compute Dice score
        masks_pred = torch.sigmoid(masks_pred) > 0.5
        dice = dice_coeff(masks_pred, true_masks)
        val_dice += dice.item()

    model.train()
    return val_loss / len(dataloader), val_dice / len(dataloader)


def dice_coeff(pred, target, smooth=1e-6):
    """
    Compute the Dice coefficient.

    Args:
        pred: Predicted masks.
        target: Ground truth masks.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice coefficient.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def load_checkpoint(checkpoint_file: str, model, optimizer=None,device='cpu'):
    """
    Load a checkpoint.

    Args:
        checkpoint_file: Path to the checkpoint file.
        model: Model to load the checkpoint into.
        optimizer: Optimizer to load the checkpoint into (optional).

    Returns:
        epoch: Epoch number from which to resume training.
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def result_analysis(
    model,
    test_loader,
    device,
    checkpoint_path,
    save_dir='results',
    epoch=None,
):
    """
    Analyze the model's performance on the test dataset.

    Args:
        model: The trained model.
        test_loader: DataLoader for the test dataset.
        device: Device to use for evaluation.
        checkpoint_path: Path to the checkpoint file.
        save_dir: Directory to save results and figures.
        epoch: Epoch number to load the checkpoint (optional).
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Initialize lists to store results
    y_true = []
    y_pred = []
    y_score = []

    # Evaluate on the test dataset
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.float32)

            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # Binary predictions
            scores = torch.sigmoid(outputs)  # Probabilities for AUC

            # Store results
            y_true.extend(masks.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())
            y_score.extend(scores.cpu().numpy().flatten())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)

    # Print metrics
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # Save metrics to a file
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'AUC: {auc:.4f}\n')

    print(f'Results and figures saved in {save_dir}')