import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
import torch
from train import result_analysis,visualize_fire_spread
from dataset import NextDayFireDataset
import tensorflow as tf
from torch.utils.data import DataLoader
from model.resnet import ResNetUNet

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the test dataset
test_data_file_pattern = '/s/lovelace/h/nobackup/sangmi/hackathon/AAAI-2025/data/northamerica_2012-2023/val/_ongoing_*.tfrecord'
test_data_file_names = tf.io.gfile.glob(test_data_file_pattern)
test_data = tf.data.TFRecordDataset(test_data_file_names)

test_set = NextDayFireDataset(
    test_data,
    limit_features_list=['elevation', 'th', 'sph', 'pr', 'NDVI', 'PrevFireMask'],
    clip_normalize=True,
    sampling_method='original',
    max_samples=500,
)

test_loader = DataLoader(
    test_set,
    shuffle=False,
    batch_size=1,
)

# Initialize the model
model = ResNetUNet(in_channels=6, num_classes=1)

# Path to the checkpoint
checkpoint_path = './checkpoints/ResNetUNet_checkpoint_epoch8.pth'  # Replace with the correct path

# Perform result analysis
result_analysis(
    model=model,
    test_loader=test_loader,
    device=device,
    checkpoint_path=checkpoint_path,
    save_dir='results',
    epoch=8,  # Replace with the epoch number you want to analyze
)
# Perform visualization
visualize_fire_spread(
    model=model,
    test_loader=test_loader,
    device=device,
    checkpoint_path=checkpoint_path,
    save_dir='visualizations',
    num_pairs=3,  # Number of image pairs to visualize
)