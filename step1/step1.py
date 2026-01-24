# -*- coding: utf-8 -*-
#%%
import os
import numpy as np
import random
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms, datasets
from PIL import Image
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import pytorch_msssim
import argparse
import json

from AE_models import CoAtNetAE, ConvNeXtAE, AE, VaeLoss, RegNetYAE, \
                    FocalNetAE, MobileVitAE
from tqdm import tqdm
from medmnist import BreastMNIST
from data_pipeline import randomly_load

from utils import get_mask_matrix, apply_mask, calc_masked_loss, image_filter, pixel_base_mask

#%%
load_dotenv()

#Load environment variables
DATASET_INFO_PATH = os.getenv("DATASET_INFO_PATH")
DATASET_ROOT = os.getenv("DATASET_ROOT")
FINETUNE_SET = os.getenv("FINETUNE_SET")

###Argument Parser###

#if --mae is set, MAE_TRAIN is set to True
parser = argparse.ArgumentParser()
parser.add_argument("--mae", action="store_true", help="Use MAE training")
# parser.add_argument("--mix_loss", action="store_true", help="Use mixed loss")
parser.add_argument("--pixel", action="store_true", help="Use pixel based mask for MAE")
# parse --latent_dim argument
parser.add_argument("--latent_dim", type=int, help="Latent dimension of the model", required=True)
# -o as output name
parser.add_argument("-o", type=str, help="Output name", required=True)
# get SEED
parser.add_argument("--seed", type=int, help="Random seed")
# loss_type="whole", "roi" or "mix"
parser.add_argument("--loss", type=str, help="Loss type", required=True, default="whole")
# --loss_func
parser.add_argument("--loss_func", type=str, help="Loss function", default="hybrid", choices=["hybrid", "ssim", "l1"])
# target_loss_weight
parser.add_argument("--target_loss_weight", "-tlw", type=float, help="Target loss weight", default=1.0)
#--epochs
parser.add_argument("--epochs", "-e", type=int, help="Number of epochs", default=1000)
#--with_training
parser.add_argument("--with_training", action="store_true", help="Use downstream training data")
#--backbone
parser.add_argument("--backbone", type=str, help="Backbone model", default="CoAtNet", choices=["CoAtNet", "ConvNeXt", "AE", "LeViTAE", "RegNetY", "FocalNet_lrf","FocalNet_srf", "MobileViT"])
#--batch_size
parser.add_argument("--batch_size", "-bs", type=int, help="Batch size", default=128)
#--without_IN
parser.add_argument("--without_IN", action="store_true", help="Use without ImageNet")
#--fixed_IN
parser.add_argument("--fixed_IN", action="store_true", help="Use fixed ImageNet weights")

args = parser.parse_args()

MAE_TRAIN = args.mae
SEED = args.seed
OUTPUT_NAME = args.o
LOSS_MODE = args.loss
PIXEL_MASK = args.pixel
TARGET_LOSS_WEIGHT = args.target_loss_weight
EPOCHS = args.epochs
WITH_TRAINING_DATA = args.with_training
BACKBONE = args.backbone
BATCH_SIZE = args.batch_size
WITHOUT_IMAGENET = args.without_IN
FIXED_IMAGENET = args.fixed_IN
LOSS_FUNC = args.loss_func

#Check if the loss mode is valid
if LOSS_MODE not in ["whole", "roi", "mix"]:
    raise ValueError("Loss mode must be one of whole, roi or mix")

if LOSS_MODE in ["roi", "mix"] and not(MAE_TRAIN or PIXEL_MASK):
    raise NotImplementedError("ROI loss or mixed loss can only be used with MAE training")

if PIXEL_MASK and MAE_TRAIN:
    raise NotImplementedError("MAE cannot be used with pixel mask or Pixel cannot be used with MAE")

latent_dim = args.latent_dim
# MAE_TRAIN = False

print(f"{MAE_TRAIN=}")
print("SEED is set to", SEED)

rand_num = np.random.randint(0, 1000)
tmp_model_path = f"/tmp/tmp_model_{rand_num}.pth"

#Set random seed
if(SEED is not None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

#%%
####### Main Script #######

############# PARAMETER SETTING #############

### Optimizer Parameters ##
batch_size = BATCH_SIZE
learning_rate = 1e-4

### Data Loader Parameters ###
workers = 6

### Model Parameters ###
OUTPUT_DIR = f"models/{OUTPUT_NAME}"
SPLIT_CONF = f"{OUTPUT_DIR}/split_info.csv"
LOG_DIR = os.path.join(OUTPUT_DIR, "step1_log")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_PATH = f"{OUTPUT_DIR}/model.pth"

### Loss Parameters ###
kld_weight = 0.1

### Other Parameters ###
mask_ratio=0.5 #For MAE
patch_size=14 #For MAE

roi_ratio=0.25 #For pixel based mask
bg_ratio=0.5 #For pixel based mask

abn_class = "COVID-19"
normal1 = "regular"
normal2 = "Bacterial pneumonia"

data_split_ratio = (0.8, 0.1, 0.1)
covid_ratio = 0.5

print(DATASET_INFO_PATH)
data_info = randomly_load(DATASET_INFO_PATH, split_ratio=data_split_ratio, covid_ratio=covid_ratio, output_path=SPLIT_CONF)

print("Set experiment name to", OUTPUT_NAME)
print("This experiment setting latent_dim to", latent_dim, "and training for", EPOCHS, "epochs")
print("Experiment output to ", OUTPUT_DIR)
print("Model will save to", MODEL_PATH)
print("The temporary model will save to", tmp_model_path)
print("Splitting information will save to", SPLIT_CONF)
print("Using", BACKBONE, "as backbone")

#%%
######### END PARAMETER SETTING #########

# if cuda is not available, then use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#GPU Hardware Acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# assert device == "cuda", "CUDA is not available"

print(f"{device=}")

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, verbose=False, path=tmp_model_path):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.best_loss = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}. Loss: {val_loss:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} → {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss

def kld_loss(z):
    return torch.mean(0.5 * (z.pow(2)))  

class SquareCenterCropResize:
    def __init__(self, size):
        self.size = size  

    def __call__(self, img):
        w, h = img.size
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        right = left + min_side
        bottom = top + min_side
        img = img.crop((left, top, right, bottom))
        img = img.resize((self.size, self.size), Image.BILINEAR)
        return img
    
class USImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, use="train", returnId=True):
        data_frame = pd.read_csv(SPLIT_CONF)

        if use == "train":
            self.data_frame = data_frame[data_frame["Use"].isin(["Train"])]
        elif use == "val":
            self.data_frame = data_frame[data_frame["Use"].isin(["Val"])]
        elif use == "test":
            self.data_frame = data_frame[data_frame["Use"].isin(["Test"])]
        else:
            raise ValueError("Use must be one of train, val or test")
        
        self.root_dir = root_dir
        self.transform = transform
        self.image_labels = []
        self.returnId = returnId

        for _, row in self.data_frame.iterrows():
            patient_id = row["ID"]
            label = row["Label"]
            img_folder = os.path.join(self.root_dir, f"{str(patient_id)}/frames")

            # 遍历文件夹中的所有图片
            for img_file in os.listdir(img_folder):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_labels.append(
                        (os.path.join(img_folder, img_file), label, patient_id)
                    )

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path, label, patient_id = self.image_labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if(self.returnId):
            return image, label, patient_id
        else:
            label = 1 if label == abn_class else 0
            return (image, label)

class LiverDataset(Dataset):
    def __init__(self, root_dir, train=True, json_file='data/LiverTrain.json', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.data = []
        
        with open(json_file, 'r') as f:
            selected_ids = set(json.load(f))
        
        benign_dir = os.path.join(root_dir, 'Benign')
        for fname in os.listdir(benign_dir):
            if fname.startswith('B_') and fname.endswith('.png'):
                id_str = fname.split('_')[1].split('.')[0]
                id_int = int(id_str)
                if train:
                    if id_int in selected_ids:
                        self.data.append({
                            'path': os.path.join(benign_dir, fname),
                            'label': 0
                        })
                else:
                    if id_int not in selected_ids:
                        self.data.append({
                            'path': os.path.join(benign_dir, fname),
                            'label': 0
                        })

        if not train:
            malignant_dir = os.path.join(root_dir, 'Malignant')
            for fname in os.listdir(malignant_dir):
                if fname.startswith('M_') and fname.endswith('.png'):
                    self.data.append({
                        'path': os.path.join(malignant_dir, fname),
                        'label': 1
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['path']
        label = self.data[idx]['label']
        if self.transform:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        else:
            return img_path, label
        
def calc_loss(x, y, target_mask=None):
    assert LOSS_FUNC in ["hybrid", "ssim", "l1"], "Mode must be one of hybrid, ssim, l1"
    y = y.to(x.device)
    
    ssim_loss = 1 - pytorch_msssim.ssim(x, y, data_range=1, size_average=False)
    
    l1_map = F.l1_loss(x, y, reduction='none')  # (B, C, H, W)
    l1_loss = l1_map.reshape(x.shape[0], -1).mean(dim=1)  # (B,)

    if LOSS_FUNC == "ssim":
        loss_map = ssim_loss
    elif LOSS_FUNC == "l1":
        loss_map = l1_loss
    else:  
        loss_map = 0.5 * ssim_loss + 0.5 * l1_loss

    if target_mask is not None:
        target_mask = target_mask.to(x.device)
        assert target_mask.shape == loss_map.shape, f"target_mask shape {target_mask.shape} does not match loss_map shape {loss_map.shape}"
        loss_map = loss_map * (1 + (TARGET_LOSS_WEIGHT - 1) * target_mask)  

    return loss_map.mean()  

transform = transforms.Compose([
    SquareCenterCropResize(224),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class BreastMNISTWrapper(Dataset):
    def __init__(self, medmnist_dataset):
        self.dataset = medmnist_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label = int(label.item()) 
        return image, label

class SourceTrackableDataset(Dataset):
    def __init__(self, datasets: list, non_target_len: int):
        self.dataset = ConcatDataset(datasets)
        self.boundary = non_target_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        source_flag = 0 if idx < self.boundary else 1

        if isinstance(data, dict):
            data['source'] = source_flag
            return data
        elif isinstance(data, tuple):
            return (data[0], data[1], source_flag)
        else:
            raise TypeError("Unsupported data format: expected dict or tuple.")


#%%
######### Prepare Data #########
covid_dataset = USImageDataset(csv_file=SPLIT_CONF, root_dir=DATASET_ROOT, transform=transform, use="train", returnId=False)
covid_val_dataset = USImageDataset(csv_file=SPLIT_CONF, root_dir=DATASET_ROOT, transform=transform, use="val", returnId=False)

BreastMNIST_dataset1 = BreastMNISTWrapper(BreastMNIST(split="train", download=True, size=224, transform=transform))
BreastMNIST_dataset2 = BreastMNISTWrapper(BreastMNIST(split="val", download=True, size=224, transform=transform))
BreastMNIST_dataset3 = BreastMNISTWrapper(BreastMNIST(split="test", download=True, size=224, transform=transform))

finetune_dataset = datasets.ImageFolder(root=FINETUNE_SET, transform=transform)

print(f"{len(covid_dataset)=}")
# print(f"{len(liver_dataset)=}")
print(f"{len(BreastMNIST_dataset1)=}")
print(f"{len(BreastMNIST_dataset2)=}")
print(f"{len(BreastMNIST_dataset3)=}")
print(f"{len(finetune_dataset)=}")

combine_dataset = ConcatDataset([finetune_dataset, BreastMNIST_dataset1, BreastMNIST_dataset2, BreastMNIST_dataset3])

total_size = len(combine_dataset)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(combine_dataset, [train_size, val_size])

if(WITH_TRAINING_DATA):
    liver_dataset = datasets.ImageFolder(root=r"data/LiverNormal", transform=transform)

    print(f"Selected {len(liver_dataset)=}")
    
    liver_train_size = int(0.9 * len(liver_dataset))
    liver_val_size = len(liver_dataset) - liver_train_size
    liver_train_dataset, liver_val_dataset = random_split(liver_dataset, [liver_train_size, liver_val_size])
    print(f"Selected {len(liver_train_dataset)=}")
    print(f"Selected {len(liver_val_dataset)=}")


    print("This experiment will use downstream training data")
    train_dataset = SourceTrackableDataset([train_dataset, covid_dataset, liver_train_dataset], len(train_dataset))
    val_dataset = SourceTrackableDataset([val_dataset, covid_val_dataset, liver_val_dataset], len(val_dataset))
else:
    print("This experiment will not use downstream training data")
    train_dataset = SourceTrackableDataset([train_dataset], len(train_dataset))
    val_dataset = SourceTrackableDataset([val_dataset], len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers, persistent_workers=True, prefetch_factor=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers, persistent_workers=True, prefetch_factor=4)

print(F"{len(combine_dataset)=}")
print(F"{len(train_dataset)=}")
print(F"{len(val_dataset)=}")
print("Dataloader is prepared")

if(BACKBONE == "CoAtNet"):
    model = CoAtNetAE(latent_dim=latent_dim, ImageNet_pretrained=not(WITHOUT_IMAGENET)).to(device)  # autoEncoder model
elif(BACKBONE == "ConvNeXt"):
    model = ConvNeXtAE(latent_dim=latent_dim, ImageNet_pretrained=not(WITHOUT_IMAGENET)).to(device)  # autoEncoder model
elif(BACKBONE == "AE"):
    model = AE(latent_dim=latent_dim).to(device)
elif(BACKBONE == "RegNetY"):
    model = RegNetYAE(latent_dim=latent_dim, ImageNet_pretrained=not(WITHOUT_IMAGENET)).to(device)
elif(BACKBONE == "FocalNet_lrf"):
    model = FocalNetAE(latent_dim=latent_dim, ImageNet_pretrained=not(WITHOUT_IMAGENET)).to(device)
elif(BACKBONE == "FocalNet_srf"):
    model = FocalNetAE(latent_dim=latent_dim, ImageNet_pretrained=not(WITHOUT_IMAGENET), sub_type="srf").to(device)
elif(BACKBONE == "MobileViT"):
    model = MobileVitAE(latent_dim=latent_dim, ImageNet_pretrained=not(WITHOUT_IMAGENET)).to(device)

model = nn.DataParallel(model)
mask_token = None

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

if(FIXED_IMAGENET):
    print(f"Freezing backbone weights, {FIXED_IMAGENET=}")
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False

losses = []
val_losses = []
early_stopping = EarlyStopping(patience=10, verbose=True)

for epoch in range(EPOCHS):
    total_loss = 0.0
    train_total = 0

    model.train()
    for data in train_loader:
        img, label, source = data  
        batch_size = img.shape[0]
        original_img = img.clone()
        img = img.to(device)
        original_img = original_img.to(device)

        if MAE_TRAIN:
            mask = get_mask_matrix(batch_size, 224, 224, 14, mask_ratio)
            mask3 = mask.unsqueeze(1).expand(batch_size, 3, 224, 224)
            mask3 = mask3.to(device)
            mask = mask.to(device)
        
            img = apply_mask(img, mask3, mask_token)
            img = img.to(device)

        if PIXEL_MASK:
            original_image, roi_mask = image_filter(img)
            mask = pixel_base_mask(roi_mask, roi_ratio, bg_ratio)
            mask3 = mask.unsqueeze(1).expand(-1, img.shape[1], -1, -1)
            mask3 = mask3.to(device)
            mask = mask.to(device)

            img = apply_mask(img, mask3)
            img = img.to(device)

        output = model(img)

        if(LOSS_MODE == "whole"):
            if(BACKBONE == "VAE"):
                loss = VaeLoss(output, original_img, target_mask=source)
            loss = calc_loss(output, original_img, target_mask=source)
        elif(LOSS_MODE == "roi"):
            loss = calc_masked_loss(output, original_img, mask, calc_loss, target_mask=source)
        elif(LOSS_MODE == "mix"):
            loss = 0.5 * calc_loss(output, original_img, target_mask=source) + 0.5 * calc_masked_loss(output, original_img, mask, calc_loss, target_mask=source)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        train_total += batch_size

    avg_train_loss = total_loss / train_total
    scheduler.step()

    total_val_loss = 0.0
    val_total = 0

    with torch.no_grad():
        for data in val_loader:
            img, label, source = data
            batch_size = img.shape[0]
            original_img = img
            img = img.to(device)
    
            if MAE_TRAIN:
                mask = get_mask_matrix(batch_size, 224, 224, 14, mask_ratio)
                mask3 = mask.unsqueeze(1).expand(batch_size, 3, 224, 224)
                mask3 = mask3.to(device)
                mask = mask.to(device)
            
                img = apply_mask(img, mask3, mask_token)

            if PIXEL_MASK:
                original_image, roi_mask = image_filter(img)
                mask = pixel_base_mask(roi_mask, roi_ratio, bg_ratio)
                mask3 = mask.unsqueeze(1).expand(-1, img.shape[1], -1, -1)
                mask3 = mask3.to(device)
                mask = mask.to(device)

                img = apply_mask(img, mask3)
                img = img.to(device)

            output = model(img)

            if(LOSS_MODE == "whole"):
                val_loss = calc_loss(output, original_img, target_mask=source)
            elif(LOSS_MODE == "roi"):
                val_loss = calc_masked_loss(output, original_img, mask, calc_loss, target_mask=source)
            elif(LOSS_MODE == "mix"):
                val_loss = 0.5 * calc_loss(output, original_img, target_mask=source) + 0.5 * calc_masked_loss(output, original_img, mask, calc_loss, target_mask=source)

            total_val_loss += val_loss.item() * batch_size
            val_total += batch_size

        avg_val_loss = total_val_loss / val_total

        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping at epoch", epoch, "with val loss", avg_val_loss)
            model.load_state_dict(torch.load(tmp_model_path))
            break

    print("epoch [{}/{}], loss:{:.4f}, val loss:{:.4f}".format(epoch + 1, EPOCHS, avg_train_loss, avg_val_loss))
    losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    torch.save(model.state_dict(), MODEL_PATH)

results = {
    "data": {"COVID-19": [], "Bacterial pneumonia": [], "regular": []}
}
#%%

with open(f"{LOG_DIR}/loss_data.json", "w") as f:
    f.write(json.dumps({
        "training": losses,
        "validation": val_losses
    }))

plt.plot(losses)
plt.plot(val_losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Val"])
plt.savefig(f"{LOG_DIR}/loss_curve.png")
plt.clf()

print(output[0].shape)

print("Output min:", output.min().item())
print("Output max:", output.max().item())
print("Output mean:", output.mean().item())

output = (output + 1) / 2

plt.imshow(output[0].cpu().detach().numpy().transpose(1, 2, 0).astype(np.float32))
plt.savefig(f"{LOG_DIR}/A.png")
plt.clf()

img = (img + 1) /2

plt.imshow(img[0].cpu().detach().numpy().transpose(1, 2, 0).astype(np.float32))
plt.savefig(f"{LOG_DIR}/B.png")
plt.clf()

original_img = (original_img + 1) / 2

if PIXEL_MASK:
    roi_mask = roi_mask.unsqueeze(1).expand(-1, original_img.shape[1], -1, -1)
    pixel_masked_img = apply_mask(original_img, roi_mask)

    pixel_masked_img = (pixel_masked_img + 1) / 2

    plt.imshow(pixel_masked_img[1].cpu().detach().numpy().transpose(1, 2, 0).astype(np.float32))
    plt.savefig(f"{LOG_DIR}/M.png")
    plt.clf()
    
if(MAE_TRAIN or PIXEL_MASK):
    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Comparison of Original, Masked and Reconstructed Image")
    axs[0].imshow(original_img[0].cpu().detach().numpy().transpose(1, 2, 0).astype(np.float32))
    axs[1].imshow(img[0].cpu().detach().numpy().transpose(1, 2, 0).astype(np.float32))
    axs[2].imshow(output[0].cpu().detach().numpy().transpose(1, 2, 0).astype(np.float32))

    plt.savefig(f"{LOG_DIR}/C.png")
    plt.close()

else:
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Comparison of Original and Reconstructed Image")
    axs[0].imshow(img[0].cpu().detach().numpy().transpose(1, 2, 0).astype(np.float32))
    axs[1].imshow(output[0].cpu().detach().numpy().transpose(1, 2, 0).astype(np.float32))

    plt.savefig(f"{LOG_DIR}/C.png")
    plt.close()

os._exit(0)
