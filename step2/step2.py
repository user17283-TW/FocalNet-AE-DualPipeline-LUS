# -*- coding: utf-8 -*-
#%%
import os
import json
import random
import numpy as np
import pandas as pd
import torch
from common_func import PatientResult, sliding_window_mse
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from testing_related import calc_metrics
from dotenv import load_dotenv
import argparse
from AE_models import AE, CoAtNetAE, load_pretrained_weights, ConvNeXtAE, RegNetYAE, MobileVitAE, FocalNetAE
from utils import calc_profile_loss, get_mask_matrix, apply_mask
from tqdm import tqdm
# from sklearnex import patch_sklearn
from scipy.stats import gaussian_kde
from thresholding import find_local_valley_threshold

# patch_sklearn()

#%%
load_dotenv()

#Load environment variables
DATASET_ROOT = os.getenv("DATASET_ROOT")
TEMPORARY_DATA_PATH = os.getenv("TEMPORARY_DATA_PATH")
PLOT_OUTPUT_DIR = os.getenv("PLOT_OUTPUT_DIR")

parser = argparse.ArgumentParser(description="Step2")

parser.add_argument("--latent_dim", "-l", type=int, default=128, help="latent dimension")
parser.add_argument("--path", "-p", type=str, help="Step1 path", required=True)
parser.add_argument("--thres", "-t", type=str, default="unsup", choices=["unsup", "semisup"], help="Threshold type")
parser.add_argument("--seed", "-s", type=int, default=0, help="Random seed")
parser.add_argument("--profile", action="store_true", help="Use image profile")
parser.add_argument("--workers", type=int, default=64, help="num of CPU workers")
parser.add_argument("--backbone", type=str, default="CoAtNet", choices=["AE", "CoAtNet", "ConvNeXt", "RegNetY", "MobileViT", "FocalNet", "FocalNet-srf"], help="Backbone model")
parser.add_argument("--mae", action="store_true", help="Use MAE")
parser.add_argument("--qc", action="store_true", help="Use QC_mv.json")
parser.add_argument("--noise", type=float, default=0.0, help="Noise level STD")
parser.add_argument("--no_MIL", action="store_true", help="Do not use MIL")
parser.add_argument("--output", "-o", type=str, help="Output result file path", required=True)
parser.add_argument("--dynamic_thres", "-dt", action="store_true", help="Use dynamic thresholding method")
parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value")
parser.add_argument("--patch_size", "-ps", type=int, default=14, help="Patch size for MAE")
# parser.add_argument("--imagenet_only", "-IN", action="store_true", help="Use only imagenet pretrained weights")

args = parser.parse_args()

LATENT_DIM = args.latent_dim
STEP1_PATH = args.path
USE_UNSUP = args.thres == "unsup"
SEED = args.seed
USE_PROFILE = args.profile
WORKER_NUM = args.workers
BACKBONE = args.backbone
USE_MAE = args.mae
USE_QC = args.qc
NOISE_STD = args.noise
NO_MIL = args.no_MIL
OUTPUT_FILE_NAME = args.output
DYNAMIC_THRES = args.dynamic_thres
GAMMA = args.gamma
PATCH_SIZE = args.patch_size

SEED_START = SEED
SEED_END = SEED + 25

if(USE_MAE):
    print("The images will be masked")

if(USE_PROFILE):
    print("Using profile loss")

if(NO_MIL):
    print("MIL was disabled")

if(DYNAMIC_THRES):
    print("Using dynamic thresholding method")

print("Set Patch Size =", PATCH_SIZE)

if os.path.isdir(STEP1_PATH):
    Step1_results = [os.path.join(STEP1_PATH, folder) for folder in os.listdir(STEP1_PATH) if os.path.isdir(os.path.join(STEP1_PATH, folder))]
    print(Step1_results)

if os.path.isfile(os.path.join(STEP1_PATH, "model.pth")):
    Step1_results = [STEP1_PATH]
    print(Step1_results)


class GaussianNoise(object):
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

#%%
result_list = []
for seed in range(SEED_START, SEED_END):
    for folder in Step1_results:
        SEED = seed
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        print(f"Set seed to {SEED}.")

        MODEL_PATH = os.path.join(folder, "model.pth")
        TEMPORARY_DATA_PATH = os.path.join(folder, "split_info.csv")

        assert os.path.exists(MODEL_PATH), f"Model path {MODEL_PATH} does not exist"
        assert os.path.exists(TEMPORARY_DATA_PATH), f"Dataset info path {TEMPORARY_DATA_PATH} does not exist"

        #Load the model
        if(BACKBONE == "AE"):
            model = AE(latent_dim=LATENT_DIM, return_feature=True)
        elif(BACKBONE == "CoAtNet"):
            model = CoAtNetAE(latent_dim=LATENT_DIM, return_feature=True)
        elif(BACKBONE == "ConvNeXt"):
            model = ConvNeXtAE(latent_dim=LATENT_DIM, return_feature=True)
        elif(BACKBONE == "RegNetY"):
            model = RegNetYAE(latent_dim=LATENT_DIM, return_feature=True)
        elif(BACKBONE == "MobileViT"):
            model = MobileVitAE(latent_dim=LATENT_DIM, return_feature=True)
        elif(BACKBONE == "FocalNet"):
            model = FocalNetAE(latent_dim=LATENT_DIM, return_feature=True)
        elif(BACKBONE == "FocalNet-srf"):
            model = FocalNetAE(latent_dim=LATENT_DIM, return_feature=True, sub_type="srf")

        model = load_pretrained_weights(model, MODEL_PATH)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{device=}")
        model = model.to(device)

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
            def __init__(self, csv_file, root_dir, transform=None, use="train"):
                data_frame = pd.read_csv(csv_file)

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

                for _, row in self.data_frame.iterrows():
                    patient_id = row["ID"]
                    label = row["Label"]
                    img_folder = os.path.join(self.root_dir, f"{str(patient_id)}/frames")
                    tmp_list = []

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

                return image, label, patient_id

        transform = transforms.Compose(
            [
                SquareCenterCropResize(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda img: F.adjust_gamma(img, gamma=GAMMA)),
                GaussianNoise(mean=0.0, std=NOISE_STD) if NOISE_STD > 0.0 else transforms.Lambda(lambda x: x),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #[-1, 1]
            ]
        )

        def compute_upper_whisker(data: list[float]) -> float:
            data = np.array(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            return data[data <= upper_bound].max()

        def find_density_elbow_threshold(
            scores: np.ndarray,
            grid_size: int = 1000,
            bandwidth: str = 'scott'
            ) -> (float, float):

            kde = gaussian_kde(scores, bw_method=bandwidth)
            
            x_min, x_max = scores.min(), scores.max()
            grid = np.linspace(x_min, x_max, grid_size)
            
            density = kde(grid)
            density_deriv = np.gradient(density, grid)  # d(density)/d(x)
            
            idx_elbow = np.argmin(density_deriv)
            threshold = grid[idx_elbow]
            
            percentile = np.mean(scores <= threshold) * 100.0
            
            return threshold, percentile

        def find_best_threshold_wacc(known_val_data, y_true, y_score):
            best_threshold = None
            best_wacc = -np.inf
            best_percentile = None

            percentiles = np.arange(0, 101, 1) 
            thresholds = np.percentile(y_score, percentiles)

            for p, thresh in zip(percentiles, thresholds):
                y_pred = (y_score >= thresh).astype(int)

                TP = np.sum((y_true == 1) & (y_pred == 1))
                TN = np.sum((y_true == 0) & (y_pred == 0))
                FP = np.sum((y_true == 0) & (y_pred == 1))
                FN = np.sum((y_true == 1) & (y_pred == 0))

                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

                wacc = (sensitivity + specificity) / 2

                if wacc > best_wacc:
                    best_wacc = wacc
                    best_threshold = thresh
                    best_percentile = p

            return best_wacc, best_percentile, best_threshold

        csv_data = pd.read_csv(TEMPORARY_DATA_PATH)

        if(USE_QC):
            print("Data len before removing QC_mv.json: ", len(csv_data))
            with open('./QC_mv.json', 'r') as f:
                qc_mv = json.load(f)

                ids_to_remove = [int(k) for k, v in qc_mv.items() if v == 0]

                csv_data = csv_data[~csv_data['Unnamed: 0'].isin(ids_to_remove)]
            print("Data len after removing QC_mv.json: ", len(csv_data))

        val_data = csv_data[csv_data['Use'] == 'Val']
        test_data = csv_data[csv_data['Use'] == 'Test']

        combined_data = pd.concat([val_data, test_data], ignore_index=True)

        labels = combined_data['Label']

        new_val_data, new_test_data = train_test_split(
            combined_data,
            test_size=0.3,
            stratify=labels,
            random_state=SEED
        )

        covid_val = new_val_data[new_val_data['Label'] == 'COVID-19'].sample(n=2, random_state=SEED)
        other_covid_val = new_val_data[new_val_data['Label'] == 'COVID-19'].drop(covid_val.index)
        non_covid_val = new_val_data[new_val_data['Label'] != 'COVID-19']
        new_val_data = pd.concat([covid_val, non_covid_val], ignore_index=True)

        new_val_data['Use'] = 'Val'
        new_test_data['Use'] = 'Test'

        file_id = random.randint(100, 999)
    
        new_val_data.to_csv(f'/tmp/new_val_{file_id}.csv', index=False)
        new_test_data.to_csv(f'/tmp/new_test_{file_id}.csv', index=False)

        val_dataset = USImageDataset(csv_file=f'/tmp/new_val_{file_id}.csv', root_dir=DATASET_ROOT, transform=transform, use="val")
        test_dataset = USImageDataset(csv_file=f'/tmp/new_test_{file_id}.csv', root_dir=DATASET_ROOT, transform=transform, use="test")

        print(len(val_dataset), len(test_dataset))

        val_labels = val_dataset.data_frame['Label'].value_counts(normalize=False)
        test_labels = test_dataset.data_frame['Label'].value_counts(normalize=False)
        print(f"Val dataset ratio: {val_labels}")
        print(f"Test dataset ratio: {test_labels}")

        print(f"{len(val_dataset)=}")
        print(f"{len(test_dataset)=}")

        print("Dataloader is prepared")

        loss_func = nn.MSELoss()

        results = {
            "data": {"COVID-19": [], "Bacterial pneumonia": [], "regular": []}
        }

        val_dataLoader = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=WORKER_NUM, prefetch_factor=4)
        val_frame_level_raw = []
        val_frame_level_true = []
        latent_ids = []
        with torch.no_grad():
            for data in tqdm(val_dataLoader, desc="Validation Progress"):

                model.eval()  #

                img, label, patient_id = data 
                original_img = img.clone()
                original_img = original_img.to(device)
                img = img.to(device)

                if(USE_MAE):
                    batch_size = img.shape[0]
                    mask = get_mask_matrix(batch_size, 224, 224, 14, 0.5)
                    mask3 = mask.unsqueeze(1).expand(batch_size, 3, 224, 224)
                    mask3 = mask3.to(device)
                    mask = mask.to(device)
            
                    img = apply_mask(img, mask3)

                output, latent= model(img)

                if(NO_MIL):
                    loss = loss_func(output, original_img).item()
                else:
                    loss = sliding_window_mse(output, original_img, window_size=PATCH_SIZE)
                    loss = compute_upper_whisker(loss)

                output = (output + 1) / 2
                img = (img + 1) / 2

                profile = calc_profile_loss(output, img)
                
                if(USE_PROFILE):
                    results["data"][label[0]].append(profile)
                    val_frame_level_raw.append(profile)
                    latent_ids.append(patient_id[0].item())
                    val_frame_level_true.append(1 if label[0] == "COVID-19" else 0 )
                else:
                    results["data"][label[0]].append(loss)
                    val_frame_level_raw.append(loss)
                    latent_ids.append(patient_id[0].item())
                    val_frame_level_true.append(1 if label[0] == "COVID-19" else 0 )

        known_val_data = results["data"]["Bacterial pneumonia"] + results["data"]["regular"]
        target_val_data = results["data"]["COVID-19"]

        fpr, tpr, thresholds = roc_curve(val_frame_level_true, val_frame_level_raw)

        if(DYNAMIC_THRES):
            threshold, percentile = find_local_valley_threshold(np.array(known_val_data), plot_comment=SEED, draw_plot=False)
        else:
            threshold = np.percentile(known_val_data, 75)
            percentile = 75.0

        if(not(USE_UNSUP)):
            youden_index = tpr - fpr 
            best_idx = np.argmax(youden_index)
            best_threshold = thresholds[best_idx]

            new_threshold = best_threshold
            print(f"[SEMISUP MODE] New threshold: {new_threshold}, original threshold: {threshold}")
            threshold = new_threshold

        test_dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=WORKER_NUM, prefetch_factor=4)

        roc_data = []
        patient_level_results = []

        for i in range(125):
            patient_level_results.append(PatientResult(i, None))

        model.eval() 
        frame_level_raw = []
        frame_level_profile = []
        frame_level_y_true = []
        ids = []

        with torch.no_grad():
            for data in tqdm(test_dataLoader, desc="Test Progress"):
                img, label, patient_id = data  
                patient_id_int = patient_id[0].item()

                original_img = img.clone()
                original_img = original_img.to(device)
                img = img.to(device)

                if(USE_MAE):
                    batch_size = img.shape[0]
                    mask = get_mask_matrix(batch_size, 224, 224, 14, 0.5)
                    mask3 = mask.unsqueeze(1).expand(batch_size, 3, 224, 224)
                    mask3 = mask3.to(device)
                    mask = mask.to(device)
            
                    img = apply_mask(img, mask3)

                output, latent= model(img)
                if(NO_MIL):
                    loss = loss_func(output, original_img).item()
                else:
                    loss = sliding_window_mse(output, original_img, window_size=PATCH_SIZE)
                    loss = compute_upper_whisker(loss)

                output = (output + 1) / 2
                img = (img + 1) / 2
                profile = calc_profile_loss(output, original_img)

                if(USE_PROFILE):
                    patient_level_results[patient_id-1].add_frame_raw_data(profile)
                else:
                    patient_level_results[patient_id-1].add_frame_raw_data(loss)

                patient_level_results[patient_id - 1].label = 1 if label[0] == "COVID-19" else 0

                if(USE_PROFILE):
                    frame_level_raw.append(profile)
                    frame_level_y_true.append(1 if label[0] == "COVID-19" else 0)
                    ids.append(patient_id_int)
                else:
                    frame_level_raw.append(loss)
                    frame_level_y_true.append(1 if label[0] == "COVID-19" else 0)
                    ids.append(patient_id_int)

        print(set(frame_level_y_true))
        print(f"Loss range: {min(frame_level_raw)} to {max(frame_level_raw)}")

        fpr, tpr, thresholds = roc_curve(frame_level_y_true, frame_level_raw)
        print("Frame Level AUC=", roc_auc_score(frame_level_y_true, frame_level_raw))
        frame_auc = roc_auc_score(frame_level_y_true, frame_level_raw)

        frame_level_class = [1 if x > threshold else 0 for x in frame_level_raw]

        print(f"Frame Level bAcc = {accuracy_score(frame_level_y_true, frame_level_class)}")
        print(f"Frame Level Acc = {accuracy_score(frame_level_y_true, frame_level_class)}")
        print(f"Frame Level Sensitivity = {recall_score(frame_level_y_true, frame_level_class)}")

        patient_level_results = [patient for patient in patient_level_results if len(patient.frame_raw_data) > 0]

        cm = np.zeros((2, 2))
        best_acc = 0

        y_pred = []
        y_true = []

        print(threshold)
        for patient in patient_level_results:
            patient.classification(threshold, reduce=False)

            y_pred.append(patient.predict_label(threshold=0.5))
            y_true.append(patient.label)

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        accuracy = np.sum(y_pred == y_true) / len(y_true)

        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))

        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        cm = np.array([[TN, FP], [FN, TP]])

        exp_name = os.path.basename(folder)
        res = calc_metrics(y_true, y_pred, auc=frame_auc, comment=SEED)
        print(threshold, accuracy, balanced_accuracy, sensitivity, specificity)

        result_list.append(res)

        with open(f"output/Step2_{OUTPUT_FILE_NAME}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(result_list))
            f.close()

        print(best_acc, cm)
