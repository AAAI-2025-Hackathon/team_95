import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
import torch
import tensorflow as tf
from dataset import *
import logging
import os
from argparse import ArgumentParser
from train import train_next_day_fire, seed_all, load_checkpoint
from model.resnet import ResNetUNet  # Import the ResNetUNet model

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Argument parser
parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dir_checkpoint', type=str, default='./checkpoints/')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

# Seed everything for reproducibility
seed_all(args.seed)

# Set parameters
limit_features = [
    'elevation',
    'th',
    'sph',
    'pr',
    'NDVI',
    'PrevFireMask',
]
n_channels = len(limit_features)
n_classes = 1
dir_checkpoint = args.dir_checkpoint
os.makedirs(dir_checkpoint, exist_ok=True)

# Load checkpoint if available
try:
    # Find the last modified checkpoint in the directory
    last_checkpoint = sorted(
        [f for f in os.listdir(dir_checkpoint) if f.find('checkpoint')],
        key=lambda f: os.path.getmtime(os.path.join(dir_checkpoint, f)),
        reverse=True,
    )[0]
    print(f'Loading {last_checkpoint}')
    load_model = f'{dir_checkpoint}/{last_checkpoint}'
except IndexError:
    print('No checkpoints found')
    load_model = None

starting_epoch = 37
epochs = args.epochs
batch_size = args.batch_size
lr = 0.01
scale = 0.5
val_percent = 20
pos_weight = 3.0

# Datasets
train_data_file_pattern = (
    '/s/lovelace/h/nobackup/sangmi/hackathon/AAAI-2025/data/northamerica_2012-2023/train/_ongoing_*.tfrecord'
)
train_data_file_names = tf.io.gfile.glob(train_data_file_pattern)
val_data_file_pattern = (
    '/s/lovelace/h/nobackup/sangmi/hackathon/AAAI-2025/data/northamerica_2012-2023/val/_ongoing_*.tfrecord'
)
val_data_file_names = tf.io.gfile.glob(val_data_file_pattern)

# Make tf datasets
train_data = tf.data.TFRecordDataset(train_data_file_names)
val_data = tf.data.TFRecordDataset(val_data_file_names)

# Logging and device setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')

# Initialize the ResNetUNet model
model = ResNetUNet(in_channels=n_channels, num_classes=n_classes)
model = model.to(memory_format=torch.channels_last)

logging.info(
    f'Network:\n'
    f'\t{model.in_channels} input channels\n'
    f'\t{model.num_classes} output channels (classes)\n'
)

# Load model checkpoint if available
if load_model:

    cehckpoint = load_checkpoint(load_model,model=model,device=device)
    #model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {load_model}')
    logging.info(f'Starting from Epoch {starting_epoch}')

model.to(device=device)
logging.info(f'Model loaded to {device}')

# Train the model
train_next_day_fire(
    starting_epoch=starting_epoch,
    train_data=train_data,
    val_data=val_data,
    model=model,
    epochs=epochs,
    dir_checkpoint=dir_checkpoint,
    save_checkpoint=True,
    batch_size=batch_size,
    learning_rate=lr,
    device=device,
    img_scale=scale,
    val_percent=val_percent / 100,
    pos_weight=pos_weight,
    limit_features=limit_features,
    max_train_samples = 10000,
    max_val_samples= 50
)

