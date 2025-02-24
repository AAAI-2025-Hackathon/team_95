import torch
import tensorflow as tf
from dataset import *
import logging
import os
from argparse import ArgumentParser
from model.asufm.asufm import ASUFM
from train import train_next_day_fire, seed_all, load_checkpoint
from configs.asufm import get_asfum_6_configs

# Enable cudnn optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dir_checkpoint', type=str, default='./checkpoints/')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
# New argument: if true, use the augmented dataset for training
parser.add_argument(
    '--use_augmented',
    type=lambda x: (str(x).lower() in ['true', '1']),
    default=False,
    help='If true uses the augmented dataset next-day-fire-2012-2023 for training; otherwise uses next-day-fire dataset'
)

args = parser.parse_args()
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
use_bilinear = True
n_channels = len(limit_features)
n_classes = 1
dir_checkpoint = args.dir_checkpoint
os.makedirs(dir_checkpoint, exist_ok=True)

try:
    # Sorting your checkpoints by modification time, so the most recent is first
    last_checkpoint = sorted(
        [f for f in os.listdir(dir_checkpoint) if f.startswith('checkpoint')],
        key=lambda f: os.path.getmtime(os.path.join(dir_checkpoint, f)),
        reverse=True,
    )[0]
    print(f'Loading {last_checkpoint}')
    load_model = os.path.join(dir_checkpoint, last_checkpoint)
    # Extract stride assuming the filename format is "checkpoint_epoch_{epoch}_{...}.pth"
    parts = last_checkpoint.split('_')
    starting_epoch = int(parts[2])  # This gets "10" from checkpoint_epoch_10_0.pth
except IndexError:
    print('No checkpoints found')
    load_model = None
    starting_epoch = 0

pos_weight = 3.0
print(f'Starting epoch: {starting_epoch}')
epochs = args.epochs
batch_size = args.batch_size
lr = 0.0001
scale = 0.5
val_percent = 20
amp = True
optimizer = 'adamw'
optimizer_state_dict = None
loss_function = 'bce'
activation = 'relu'
sampling = 'original'
skip_eval = False
use_checkpointing = False

# Choose dataset file patterns based on the --use_augmented flag
if args.use_augmented:
    train_data_file_pattern = 'next-day-fire-2012-2023/northamerica_2012-2023/train/*.tfrecord'
    val_data_file_pattern = 'next-day-fire-2012-2023/northamerica_2012-2023/val/*.tfrecord'
    print('Using augmented training dataset')
else:
    train_data_file_pattern = 'next-day-fire/next_day_wildfire_spread_train_*.tfrecord'
    val_data_file_pattern = 'next-day-fire/next_day_wildfire_spread_eval_*.tfrecord'
    print('Using original training dataset')

# Glob the files and create TFRecord Datasets
train_data_file_names = tf.io.gfile.glob(train_data_file_pattern)
val_data_file_names = tf.io.gfile.glob(val_data_file_pattern)
train_data = tf.data.TFRecordDataset(train_data_file_names)
val_data = tf.data.TFRecordDataset(val_data_file_names)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')

# Train the model
config = get_asfum_6_configs()
model = ASUFM(config=config, num_classes=n_classes)
model = model.to(memory_format=torch.channels_last)

logging.info(
    f'Network:\n'
    f'\t{model.in_chans} input channels\n'
    f'\t{model.num_classes} output channels (classes)\n'
)

if load_model:
    state_dict, optimizer_state_dict, _, _, _ = load_checkpoint(load_model)
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {load_model}')
    logging.info(f'Optimizer loaded from {load_model}')
    logging.info(f'Starting from Epoch {starting_epoch}')

model.to(device=device)
logging.info(f'Model loaded to {device}')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')

# Start training
train_next_day_fire(
    starting_epoch=starting_epoch,
    train_data=train_data,
    val_data=val_data,
    model=model,
    epochs=epochs,
    dir_checkpoint=dir_checkpoint,
    batch_size=batch_size,
    learning_rate=lr,
    device=device,
    img_scale=scale,
    val_percent=val_percent / 100,
    amp=amp,
    pos_weight=pos_weight,
    limit_features=limit_features,
    optimizer=optimizer,
    optimizer_state_dict=optimizer_state_dict,
    loss_function=loss_function,
    activation=activation,
    sampling_method=sampling,
    skip_eval=skip_eval,
    use_checkpointing=use_checkpointing,
)