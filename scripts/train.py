#! /usr/bin/python
#
# This is a script to train volumetric CNNs to identify the V1, V2, and V3
# boundaries in the Human Connectome Project.


# Step 1: Import dependencies.

from pathlib import Path
import os, sys, time, copy, argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

import neuropythy as ny
import pimms

try:
    import volcnn
except ModuleNotFoundError:
    print("ERROR: Could not import volcnn library!")
    print("       Make sure volcnn is on your PYTHONPATH.")
    sys.exit(1)
from volcnn import (
    HCPVolumeDataset,
    make_dataloaders,
    sids,
    data_cache_path,
    dice_loss,
    bce_loss,
    UNet3D)

# Go ahead and define the training and validation partition:
trn_sids = [100610, 118225, 140117, 158136, 197348, 214524, 346137, 412528,
            573249, 724446, 905147, 102311, 159239, 173334, 221319, 352738,
            429040, 725751, 826353, 910241, 102816, 145834, 162935, 175237,
            199655, 233326, 436845, 732243, 926862, 104416, 128935, 146129,
            164131, 200210, 365343, 463040, 751550, 859671, 927359, 105923,
            130114, 146432, 164636, 187345, 200311, 467351, 617748, 757764,
            942658, 108323, 130518, 165436, 191033, 200614, 249947, 381038,
            525541, 627549, 109123, 131217, 146937, 167036, 177746, 191336,
            201515, 385046, 536647, 638049, 770352, 872764, 111312, 167440,
            178142, 191841, 203418, 257845, 541943, 771354, 878776, 958976,
            111514, 132118, 169040, 178243, 393247, 547046, 654552, 878877,
            966975, 114823, 155938, 205220, 283543, 395756, 550439, 671855,
            783462, 898176, 971160, 156334, 180533, 193845, 318637, 397760,
            552241, 680957, 899885, 973770, 115825, 135124, 157336, 169747,
            181232, 401422, 562345, 690152, 814649, 901139, 995174, 116726,
            137128, 158035, 181636, 196144, 330324, 406836, 572045, 818859]
val_sids = [765864, 209228, 134829, 585256, 901442, 169444, 380036, 389357,
            581450, 198653, 115017, 782561, 176542, 246133, 185442, 601127,
            204521, 195041, 182739, 212419, 263436, 320826, 825048, 192641,
            360030, 177140, 146735, 126426, 789373, 871762, 172130, 171633]

# Step 2: Process Arguments.
# At this point we've imported volcnn and are ready to process arguments.
# Later we will want to process these parameters from the arguments. For now
# we just define most of them.

parser = argparse.ArgumentParser()
parser.add_argument(
    "tag",
    help="the unique tag used to name the model",
    type=str)
parser.add_argument(
    "inputs",
    help="either 'anat' for anatomical inputs or 'dwi' for diffusion",
    choices=['anat', 'dwi'],
    type=str)
parser.add_argument(
    "-M", "--model-path",
    dest='model_path',
    help="the directory into which to save the best model",
    type=str,
    default=".")
parser.add_argument(
    "-C", "--data-cache-path",
    dest='data_cache_path',
    help="the directory in which to cache the input data",
    type=str,
    default=None)
parser.add_argument(
    "-d", "--device",
    help="the device to use ('cpu' or 'cuda')",
    type=str,
    default='cpu',
    choices=['cpu', 'cuda'])
parser.add_argument(
    "-z", "--zoom",
    help="how much to up/down sample the data (usually 0.25 or 0.5)",
    default=0.25,
    type=float)
parser.add_argument(
    "-n", "--num-epochs",
    dest='num_epochs',
    help="the number of epochs to run (default 30)",
    default=30,
    type=int)
parser.add_argument(
    "-b", "--batch-size",
    dest='batch_size',
    help="the batch-size to use in training (default 5)",
    type=int,
    default=5)
parser.add_argument(
    "-l", "--lr",
    help="the learning rate to use in training (default 0.0075)",
    type=float,
    default=0.0075)
parser.add_argument(
    "-g", "--gamma",
    help="the learning rate decay to use in training (default 0.95)",
    type=float,
    default=0.95)
args = parser.parse_args()

# What directory are we saving the trained model to?
if args.model_path is None:
    output_path = Path(".")
else:
    output_path = Path(args.model_path)
if not output_path.is_dir():
    print("ERROR: output_path does not exist:", str(output_path))
    sys.exit(1)

# Where is the cache path for the training data?
if args.data_cache_path is not None:
    volcnn.data_cache_path = Path(args.data_cache_path)

# Zoom is how much we downsample the images; 1/4 means we take a 256
# x 256 x 256 image and turn it into a 64 x 64 x 64. A value of 1/2
# would mean that we turn the 256-cube image into a 128-cube image.
zoom = args.zoom

# The inputs we will give to the UNet:
if args.inputs == 'anat':
    inputs = ('graymask', 'T1', 'T2')
elif args.inputs == 'dwi':
    inputs = ('graymask', 'ODI', 'FA', 'FICVF')
else:
    print("ERROR: unrecognized input (must be 'anat' or 'dwi'):", inputs)
    sys.exit(1)

# The tag name we give the model.
tag = args.tag
    
# The outputs we will train the UNet to predict:
outputs = ('V1', 'V2', 'V3')

# The dtype we'll use (float speeds things up over double).
dtype = torch.float

# The device ('cpu' or 'cuda', if available).
device = args.device

# What is the batch-size (how many examples do we give the model at
# a time during training)?
batch_size = args.batch_size

# Should the dataloaders shuffle the training examples?
shuffle = True

# Additional parameters for the training.
lr = args.lr #0.0075
gamma = args.gamma #0.95
num_epochs = args.num_epochs #30


# Step 3: Load the data to use in training.

# Now, make the dataloaders; the make_dataloaders() function returns
# a tuple of the training dataloader and the validation dataloader.
# We only need to do this once, no matter how many times we train
# the model.
(trn_loader, val_loader) = make_dataloaders(
    inputs=inputs,
    outputs=outputs, 
    zoom=zoom,
    dtype=dtype,
    device=device,
    batch_size=batch_size,
    shuffle=shuffle,
    partition=(trn_sids, val_sids))
dataloaders = {'trn': trn_loader, 'val': val_loader}


# Step 4: Train the model.

# Make the model.
model = UNet3D(len(inputs), len(outputs))

# We use the Adam optimizer.
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr)
# We use the StepLR scheduler.
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=gamma)

# We're now ready to run the model, so we print an output message with the
# parameters included.
print("Running train.py with the following parameters:")
print("---")
print(f"   tag:             {tag}")
print(f"   inputs:          {inputs}")
print(f"   outputs:         {outputs}")
print(f"   model_path:      {output_path}")
print(f"   data_cache_path: {data_cache_path}")
print(f"   device:          {device}")
print(f"   zoom:            {zoom}")
print(f"   num_epochs:      {num_epochs}")
print(f"   lr:              {lr}")
print(f"   gamma:           {gamma}")
print(f"   batch_size:      {batch_size}")
print("---")

# We'll want to keep track of the losses at each step.
trn_losses = []
val_losses = []
losses = {'trn': trn_losses, 'val': val_losses}
dice_losses = {'trn': [], 'val': []}
loss_tables = {'trn': [], 'val': []}
dice_loss_tables = {'trn': [], 'val': []}

# And the best model weights.
best_weights = None
best_loss = np.inf

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1:02d}:")
    losses_epoch = {'trn': [], 'val': []}
    dice_losses_epoch = {'trn': [], 'val': []}
    since = time.time()
    wd = epoch / (num_epochs - 1)
    wb = 1 - wd
    # Each epoch has a training and validation phase
    for phase in ['trn', 'val']:
        print('  Training..' if phase == 'trn' else '  Testing..', end='')
        if phase == 'trn':
            model.train()  # Set model to training mode.
        else:
            model.eval()   # Set model to evaluate mode.
        epoch_samples = 0
        for (inputs, labels) in dataloaders[phase]:
            print('.', end="")
            # Zero the parameter gradients.
            optimizer.zero_grad()
            # Calculate the forward model and the loss.
            with torch.set_grad_enabled(phase == 'trn'):
                outputs = model(inputs)
                bloss = bce_loss(outputs, labels, logits=model.logits)
                dloss = dice_loss(outputs, labels, logits=model.logits)
                loss = bloss*wb + dloss*wd
                # backward + optimize only if in training phase
                if phase == 'trn':
                    loss.backward()
                    optimizer.step()
                # Add the loss to our trackers.
                losses_epoch[phase].append((float(loss), outputs.shape[0]))
                dice_losses_epoch[phase].append((float(dloss), outputs.shape[0]))
        print(' Done. (lr=' + str(optimizer.param_groups[0]['lr']) + ')')
    scheduler.step()
    endepoch = time.time()
    for (k,v) in losses_epoch.items():
        s = [l*s for (l,s) in v]
        losses[k].append(np.sum(s) / len(dataloaders[k].dataset))
        loss_tables[k].append([u[0] for u in v])
    for (k,v) in dice_losses_epoch.items():
        s = [l*s for (l,s) in v]
        dice_losses[k].append(np.sum(s) / len(dataloaders[k].dataset))
        dice_loss_tables[k].append([u[0] for u in v])
    if dice_losses['val'][-1] < best_loss:
        best_loss = dice_losses['val'][-1]
        best_weights = copy.deepcopy(model.state_dict())
    print(
        f"  {endepoch - since:-6.2f}s,"
        f" {trn_losses[-1]:6.4f} [{dice_losses['trn'][-1]}] trn loss,"
        f" {val_losses[-1]:6.4f} [{dice_losses['val'][-1]}] val loss.")
final_model = model
best_model = UNet3D(
    model.feature_count, model.segment_count,
    base_model=model.base_model,
    logits=model.logits)
best_model.load_state_dict(best_weights)


# Step 5: Save best model weights and loss out to a file.
torch.save(best_model.state_dict(), output_path / f"bestmodel_{tag}.pt")
print("Training complete!")
sys.exit(0)
