# -*- coding: utf-8 -*-
#%%
import os
import re
import json
import random
import numpy as np
import torch
from common_func import PatientResult, sliding_window_mse
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from testing_related import calc_metrics
from dotenv import load_dotenv
import argparse
from AE_models import AE, CoAtNetAE, load_pretrained_weights, ConvNeXtAE, RegNetYAE, FocalNetAE, MobileVitAE
from utils import calc_profile_loss, get_mask_matrix, apply_mask
from tqdm import tqdm
from data_pipeline import OpenPOCUSDataset
from scipy.stats import gaussian_kde

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
parser.add_argument("--backbone", type=str, default="CoAtNet", choices=["AE", "CoAtNet", "ConvNeXt", "RegNetY", "FocalNet", "MobileViT", "FocalNet-srf"], help="Backbone model")
parser.add_argument("--mae", action="store_true", help="Use MAE")
parser.add_argument("--qc", action="store_true", help="Use QC_mv.json")
parser.add_argument("--output", "-o", type=str, default="Holdout_Result", help="Output file name prefix")
parser.add_argument("--gamma", type=float, default=1.0, help="Gamma value for brightness adjustment")

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
OUTPUT_NAME = args.output
GAMMA = args.gamma

SEED_START = SEED
SEED_END = SEED_START + 25

if(USE_MAE):
    print("The images will be masked")

if(USE_PROFILE):
    print("Using profile loss")

if os.path.isdir(STEP1_PATH):
    Step1_results = [os.path.join(STEP1_PATH, folder) for folder in os.listdir(STEP1_PATH) if os.path.isdir(os.path.join(STEP1_PATH, folder))]
    print(Step1_results)

if os.path.isfile(os.path.join(STEP1_PATH, "model.pth")):
    Step1_results = [STEP1_PATH]
    print(Step1_results)


#%%
result_list = []
for seed in range(SEED_START, SEED_END):
    for folder in Step1_results:
        #Set the seed
        SEED = seed
        # SEED=94
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
        elif(BACKBONE == "FocalNet"):
            model = FocalNetAE(latent_dim=LATENT_DIM, return_feature=True)
        elif(BACKBONE == "MobileViT"):
            model = MobileVitAE(latent_dim=LATENT_DIM, return_feature=True)
        elif(BACKBONE == "FocalNet-srf"):
            model = FocalNetAE(latent_dim=LATENT_DIM, return_feature=True, sub_type="srf")

        model = load_pretrained_weights(model, MODEL_PATH)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{device=}")
        model = model.to(device) 

        class SquareTopCropResize:
            def __init__(self, size):
                self.size = size 

            def __call__(self, img):
                w, h = img.size
                min_side = min(w, h)
                left = (w - min_side) // 2
                top = 0
                right = left + min_side
                bottom = min_side
                img = img.crop((left, top, right, bottom))
                img = img.resize((self.size, self.size), Image.BILINEAR)
                return img
    
        transform = transforms.Compose(
            [
                SquareTopCropResize(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda img: F.adjust_gamma(img, gamma=GAMMA)) if GAMMA != 1.0 else transforms.Lambda(lambda x: x),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #[-1, 1]
            ]
        )

        def remapping(patient_id):
            mapping_dict = {
                "151 A": 250,
                "151 B": 251,
                "156 A": 252,
                "156 B": 253,
                "157 A": 254,
                "157 B": 255,
                "146 A": 256,
                "146 B": 257,
                "146 C": 258,
                "160 A": 259,
                "160 B": 260,
                "160 C": 261,
                "ED1": 262,
                "ED2": 263,
                "ED3": 264,
                "ED4": 265,
                "ED5": 266,
                "ED6": 267,
                "ED7": 268,
                "ED8": 269,
                "ED9": 270,
                "ED10": 271,
                "9.19": 272,
                "14.12": 273,
                "23.4": 274,
                "27.16": 275,
                "28.17": 276,
                "29.21": 277,
                "30.22": 278,
                "31.2": 279,
                "33.16": 280,
                "35.3": 281,
                "37.2": 282,
                "38.4": 283,
                "40.2": 284,
                "41.3": 285,
                "46.15": 286,
                "49.23": 287,
                "52.17": 288,
                "53.19": 289,
                "54.18": 290,
                "72.18": 291,
                "73.2": 292,
                "80.12": 293,
                "91.31": 294,
                "93.21": 295,
                "95.2": 296,
                "97.2": 297,
                "98.22": 298,
                "107.4": 299,               
            }

            if re.match(r"^\d+$", patient_id):
                return int(patient_id) + 1
            else:
                return mapping_dict[patient_id]
        
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

        def find_local_valley_threshold(
            scores: np.ndarray,
            grid_size: int = 1000,
            bandwidth: str = 'scott',
            min_percentile: float = 25.0,
            max_percentile: float = 75.0,
            smooth_window: int = 5,
            draw_plot: bool = True
        ) -> (float, float):
            kde = gaussian_kde(scores, bw_method=bandwidth)
            grid = np.linspace(scores.min(), scores.max(), grid_size)
            density = kde(grid)

            if smooth_window % 2 == 0:
                smooth_window += 1
            from scipy.signal import savgol_filter  
            density_raw = density  
            poly_order = 2  
            density = savgol_filter(density_raw, smooth_window, poly_order)

            lower = np.percentile(scores, min_percentile)
            upper = np.percentile(scores, max_percentile)

            mask = (grid >= lower) & (grid <= upper)
            valid_idxs = np.where(mask)[0]
            if valid_idxs.size == 0:
                threshold = upper
            else:
                segment = density[valid_idxs]
                local_min_idx = valid_idxs[np.argmin(segment)]
                threshold = grid[local_min_idx]

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

        val_dataset = OpenPOCUSDataset(json_path="data/OpenPOCUS/openPOCUS_filtered.json", root_dir="data/OpenPOCUS/Lung", use="val", transform=transform)
        test_dataset = OpenPOCUSDataset(json_path="data/OpenPOCUS/openPOCUS_filtered.json",root_dir="data/OpenPOCUS/Lung", use="test", transform=transform)

        print(f"{len(val_dataset)=}")
        print(f"{len(test_dataset)=}")

        print("Dataloader is prepared")

        loss_func = nn.MSELoss()

        results = {
            "data": {"0": [], "1": []}
        }

        val_dataLoader = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=WORKER_NUM, prefetch_factor=4)
        val_frame_level_raw = []
        val_frame_level_true = []
        latent_ids = []
        with torch.no_grad():
            for data in tqdm(val_dataLoader, desc="Validation Progress"):

                model.eval() 

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
                loss = sliding_window_mse(output, original_img, window_size=14)

                loss = compute_upper_whisker(loss)

                output = (output + 1) / 2
                img = (img + 1) / 2

                profile = calc_profile_loss(output, img)
                
                if(USE_PROFILE):
                    results["data"][label[0]].append(profile)
                    val_frame_level_raw.append(profile)
                    latent_ids.append(patient_id[0].item())
                    val_frame_level_true.append(int(label.item()))
                else:
                    results["data"][str(label.item())].append(loss)
                    val_frame_level_raw.append(loss)
                    latent_ids.append(patient_id)
                    val_frame_level_true.append(int(label.item()))

        known_val_data = results["data"]["0"]
        target_val_data = results["data"]["1"]

        fpr, tpr, thresholds = roc_curve(val_frame_level_true, val_frame_level_raw)

        threshold = np.percentile(known_val_data, 75)
        percentile = 75.0
        
        q1 = np.percentile(known_val_data, 25)
        q2 = np.percentile(known_val_data, 50)
        q3 = np.percentile(known_val_data, 75)
        iqr = q3 - q1
        
        print(f"Q2: {q2}, Automatic threshold: {threshold}, percentile: {percentile}")
        # threshold = 0.005

        if(not(USE_UNSUP)):
            youden_index = tpr - fpr

            best_idx = np.argmax(youden_index)
            best_threshold = thresholds[best_idx]

            new_threshold = best_threshold
            print(f"[SEMISUP MODE] New threshold: {new_threshold}, original threshold: {threshold}")
            threshold = new_threshold

        ######## Test ########
        test_dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=WORKER_NUM, prefetch_factor=4)

        roc_data = []
        patient_level_results = []

        for i in range(500):
            patient_level_results.append(PatientResult(i, None))

        model.eval() 
        frame_level_raw = []
        frame_level_profile = []
        frame_level_y_true = []
        ids = []

        with torch.no_grad():
            for data in tqdm(test_dataLoader, desc="Test Progress"):
                img, label, pt_id = data
                pt_id_str = pt_id[0]
                patient_id_int = remapping(pt_id_str)
                label = label.item()

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
                loss = sliding_window_mse(output, original_img, window_size=14)

                loss = compute_upper_whisker(loss)

                output = (output + 1) / 2
                img = (img + 1) / 2
                profile = calc_profile_loss(output, original_img)

                if(USE_PROFILE):
                    patient_level_results[patient_id_int-1].add_frame_raw_data(profile)
                else:
                    patient_level_results[patient_id_int - 1].add_frame_raw_data(loss)

                patient_level_results[patient_id_int - 1].label = int(label)

                if(USE_PROFILE):
                    frame_level_raw.append(profile)
                    frame_level_y_true.append(int(label))
                    ids.append(patient_id_int)
                else:
                    frame_level_raw.append(loss)
                    frame_level_y_true.append(int(label))
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

        print(f"Patient level results: {len(patient_level_results)}")


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

        #senstivity and specificity
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

        with open(f"output/Step2_{OUTPUT_NAME}_Holdout.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(result_list))
            f.close()

        print(best_acc, cm)
