import os, sys, time, copy, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scipy as sp
import scipy.sparse as sps
import neuropythy as ny
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import visual_autolabel as val


parser = argparse.ArgumentParser()
parser.add_argument("tag", type=str, help="Unique tag to name the model")
parser.add_argument("-M", "--model-path", type=str, default=".", help="Directory to save the best model")
parser.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
parser.add_argument("-n", "--num-epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("-b", "--batch-size", type=int, default=5, help="Batch size")
parser.add_argument("-l", "--lr", type=float, default=0.0075, help="Initial learning rate")
parser.add_argument("-g", "--gamma", type=float, default=0.95, help="Learning rate decay factor")
args = parser.parse_args()


dataset2D_cache_path = '/data/visual-autolabel/datasets/HCP'
tx3Dto2D_cache_path = '/data/visual-autolabel/volumetric/data/tx3Dto2D'

val.image.dataset3D_cache_path = '/data/visual-autolabel/volumetric/data'
val.image.noddi_data_path = '/data/visual-autolabel/volumetric/NODDI'

inputs3D = ('graymask', 'T1', 'T2')
inputs2D = ('curvature', 'convexity', 'thickness')
outputs = ('V1', 'V2', 'V3')

zoom = 1/2
subindex = (slice(2, -2), slice(8, 256+8), slice(2, -2))
dtype = torch.float32
device = torch.device(args.device)
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr
gamma = args.gamma
tag = args.tag
shuffle = True

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


class VolumeToFlatImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sids,
                 cache_path=tx3Dto2D_cache_path):
        self.cache_path = Path(cache_path)
        self.sids = sids
        self.data = {}
    def __len__(self):
        return len(self.sids)
    def __getitem__(self, k):
        sid = self.sids[k]
        if sid in self.data:
            return self.data[sid]
        matrix = torch.load(f"{self.cache_path}/{sid}.pt", weights_only=False)
        import scipy.sparse as sps
        (row, col, val) = sps.find(matrix)
        matrix = torch.sparse_coo_tensor(
            torch.as_tensor(np.array([row, col])), 
            torch.as_tensor(val),
            matrix.shape,
            dtype=torch.float32)
        #matrix = torch.tensor(matrix, dtype=torch.float32)
        self.data[sid] = matrix
        return matrix
    
class HCPHybridDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sids,
                 inputs2D,
                 inputs3D,
                 outputs=('V1', 'V2', 'V3'),
                 cache_path2D=None,
                 cache_path3D=None,
                 transform_cache_path=tx3Dto2D_cache_path,
                 dtype=None,
                 device=None,
                 mkdir_mode=509,
                 subindex=(slice(2, -2, None), slice(8, 264, None), slice(2, -2, None)),
                 zoom=0.5,
                 ):
        self.transform_dataset = VolumeToFlatImageDataset(sids, cache_path=transform_cache_path)
        self.dataset3D = val.image.HCPDataset3D(
            sids=sids,
            inputs=inputs3D,
            outputs=outputs,
            cache_path=cache_path3D,
            dtype=dtype,
            device=device,
            mkdir_mode=0o775,
            subindex=subindex,
            zoom=zoom)
        self.dataset2D = val.benson2025.hcp.HCPDataset(
            inputs2D, 
            outputs,
            sids=sids,
            cache_path=cache_path2D)
        self.sids = sids
    def __len__(self):
        return len(self.sids)
    def __getitem__(self, k):
        inputdata3D, _ = self.dataset3D[k]
        transformdata3D = self.transform_dataset[k]
        inputdata2D, outputdata2D = self.dataset2D[k]
        return (inputdata3D, transformdata3D, inputdata2D, outputdata2D)

def hybrid_collate(batch):
    input3D, transform, input2D, labels = zip(*batch)
    return (
        torch.stack(input3D),
        list(transform),
        torch.stack(input2D),
        torch.stack(labels)
    )

train_set = HCPHybridDataset(
    sids=trn_sids,
    inputs2D=inputs2D,
    inputs3D=inputs3D,
    outputs=outputs,
    cache_path2D=dataset2D_cache_path,
    cache_path3D=val.image.dataset3D_cache_path,
    transform_cache_path=tx3Dto2D_cache_path,
    dtype=dtype,
    device=device,
    mkdir_mode=0o775,
    subindex=subindex,
    zoom=zoom
)

val_set = HCPHybridDataset(
    sids=val_sids,
    inputs2D=inputs2D,
    inputs3D=inputs3D,
    outputs=outputs,
    cache_path2D=dataset2D_cache_path,
    cache_path3D=val.image.dataset3D_cache_path,
    transform_cache_path=tx3Dto2D_cache_path,
    dtype=dtype,
    device=device,
    mkdir_mode=0o775,
    subindex=subindex,
    zoom=zoom
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=hybrid_collate)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=hybrid_collate)
dataloaders = {'trn': train_loader, 'val': val_loader}


model = val.image.UNet(
    len(inputs3D), len(inputs3D), len(inputs2D), len(outputs)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=gamma)


best_loss = np.inf
best_weights = None
print("Starting training...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1:02d}/{num_epochs}")
    losses_epoch = {'trn': [], 'val': []}
    dice_epoch = {'trn': [], 'val': []}
    wd = epoch / (num_epochs - 1)
    wb = 1 - wd

    for phase in ['trn', 'val']:
        model.train() if phase == 'trn' else model.eval()
        running_loss = 0.0
        running_dice = 0.0
        count = 0

        for inputs3D, transforms, inputs2D, labels in dataloaders[phase]:
            inputs3D, inputs2D, labels = inputs3D.to(device), inputs2D.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'trn'):
                outputs = model(inputs3D, inputs2D, transforms)
                loss_bce = val.loss.bce_loss(outputs, labels)
                loss_dice = val.loss.dice_loss(outputs, labels)
                loss = wb * loss_bce + wd * loss_dice
                if phase == 'trn':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs3D.size(0)
            running_dice += loss_dice.item() * inputs3D.size(0)
            count += inputs3D.size(0)

        epoch_loss = running_loss / count
        epoch_dice = running_dice / count
        print(f"  {phase} loss: {epoch_loss:.4f}, dice: {epoch_dice:.4f}")

        if phase == 'val' and epoch_dice < best_loss:
            best_loss = epoch_dice
            best_weights = copy.deepcopy(model.state_dict())

    scheduler.step()
    for phase in ['trn', 'val']:
        if phase == 'trn':
            trn_loss = losses_epoch[phase][-1] = epoch_loss
            trn_dice = dice_epoch[phase][-1] = epoch_dice
        else:
            val_loss = losses_epoch[phase][-1] = epoch_loss
            val_dice = dice_epoch[phase][-1] = epoch_dice
            if val_dice < best_loss:
                best_loss = val_dice
                best_weights = copy.deepcopy(model.state_dict())

    print(f"  Epoch {epoch+1:02d} complete. "
          f"trn loss: {trn_loss:.4f} [{trn_dice:.4f}], "
          f"val loss: {val_loss:.4f} [{val_dice:.4f}]")

final_model = model
best_model = val.image.UNet(
    model.feature_count,
    model.feature_count,
    model.flat_feature_count,
    model.segment_count,
    base_model=model.base_model,
    logits=model.logits
)
best_model.load_state_dict(best_weights)

output_path = Path(args.model_path)
output_path.mkdir(exist_ok=True, parents=True)
torch.save(best_model.state_dict(), output_path / f"bestmodel_{tag}.pt")

print("Training complete!")
sys.exit(0)
