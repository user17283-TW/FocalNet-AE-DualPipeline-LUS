# -*- coding: utf-8 -*-
import os
import sys
import re
import json
import random
import argparse
from tqdm import tqdm
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from PIL import Image
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# --- UMAP 視覺化相關導入 ---
import umap
import seaborn as sns
from matplotlib import pyplot as plt

# --- 使用者自定義模組 ---
from common_func import PatientResult
from testing_related import calc_metrics
from AE_models import CoAtNetAE, ConvNeXtAE, RegNetYAE, MobileVitAE, FocalNetAE, load_pretrained_weights
from data_pipeline import OpenPOCUSDataset_new

# --- 全域設定 ---
load_dotenv()
DATASET_ROOT = os.getenv("DATASET_ROOT")

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Refactored Step2 with Dim Reduction")
parser.add_argument("--latent_dim", "-l", type=int, default=128, help="latent dimension")
parser.add_argument("--path", "-p", type=str, help="Step1 path", required=True)
parser.add_argument("--workers", type=int, default=16, help="num of CPU workers")
parser.add_argument("--backbone", type=str, default="CoAtNet", choices=["CoAtNet", "ConvNeXt", "RegNetY", "MobileViT", "FocalNet", "ResNet50", "FocalNet-srf"], help="Backbone model")
parser.add_argument("--dim_reduction", "-d", type=str, default="None", choices=["PCA", "LDA", "None"], help="Dimensionality reduction method")
parser.add_argument("--imagenet", action="store_true", help="Load imagenet weights only (for non-AE models)")
parser.add_argument("--output_name", "-o", type=str, help="Output result file name", required=True)
parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value")
parser.add_argument("--seed", type=int, default=542, help="Random seed for reproducibility")

args = parser.parse_args()

LATENT_DIM = args.latent_dim
STEP1_PATH = args.path
WORKER_NUM = args.workers
BACKBONE = args.backbone
DIM_REDUCTION = args.dim_reduction
IMAGENET_ONLY = args.imagenet
OUTPUT_NAME = args.output_name
GAMMA = args.gamma
SEED = args.seed
SEED_START = SEED
SEED_END = SEED + 25

print(f"Using Dimensionality Reduction: {DIM_REDUCTION}")

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        y = self.fc(x)
        return y

class SquareTopCropResize:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        w, h = img.size
        min_side = min(w, h)
        left, top = (w - min_side) // 2, 0
        img = img.crop((left, top, left + min_side, top + min_side))
        return img.resize((self.size, self.size), Image.BILINEAR)

def remapping(patient_id):
    mapping_dict = {"151 A": 250,"151 B": 251,"156 A": 252,"156 B": 253,"157 A": 254,"157 B": 255,"146 A": 256,"146 B": 257,"146 C": 258,"160 A": 259,"160 B": 260,"160 C": 261,"ED1": 262,"ED2": 263,"ED3": 264,"ED4": 265,"ED5": 266,"ED6": 267,"ED7": 268,"ED8": 269,"ED9": 270,"ED10": 271,"9.19": 272,"14.12": 273,"23.4": 274,"27.16": 275,"28.17": 276,"29.21": 277,"30.22": 278,"31.2": 279,"33.16": 280,"35.3": 281,"37.2": 282,"38.4": 283,"40.2": 284,"41.3": 285,"46.15": 286,"49.23": 287,"52.17": 288,"53.19": 289,"54.18": 290,"72.18": 291,"73.2": 292,"80.12": 293,"91.31": 294,"93.21": 295,"95.2": 296,"97.2": 297,"98.22": 298,"107.4": 299}
    if isinstance(patient_id, str) and patient_id.isdigit():
        return int(patient_id)
    elif isinstance(patient_id, str):
        return mapping_dict.get(patient_id, -1)
    return int(patient_id)

def split_patient_ids(
    json_path: str, 
    data_limit: int, 
    seed: int
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_normal_patients = data['normal']
    all_abnormal_patients = data['abnormal']

    if data_limit > len(all_normal_patients):
        raise ValueError(
            f"DATA_LIMIT ({data_limit}) 不能大於 normal 病患的總數 ({len(all_normal_patients)})"
        )
    if data_limit > len(all_abnormal_patients):
        raise ValueError(
            f"DATA_LIMIT ({data_limit}) 不能大於 abnormal 病患的總數 ({len(all_abnormal_patients)})"
        )

    random.seed(seed)

    train_normal_patients = random.sample(all_normal_patients, data_limit)
    train_abnormal_patients = random.sample(all_abnormal_patients, data_limit)

    train_normal_ids = {p['id'] for p in train_normal_patients}
    train_abnormal_ids = {p['id'] for p in train_abnormal_patients}

    test_normal_patients = [p for p in all_normal_patients if p['id'] not in train_normal_ids]
    test_abnormal_patients = [p for p in all_abnormal_patients if p['id'] not in train_abnormal_ids]
    
    train_set = {
        "normal": sorted(train_normal_patients, key=lambda p: str(p['id'])),
        "abnormal": sorted(train_abnormal_patients, key=lambda p: str(p['id']))
    }
    
    test_set = {
        "normal": sorted(test_normal_patients, key=lambda p: str(p['id'])),
        "abnormal": sorted(test_abnormal_patients, key=lambda p: str(p['id']))
    }

    return train_set, test_set

folder = STEP1_PATH
for DATA_LIMIT in [5, 10, 15, 20, 25, 30]:
    result_list = []
    for seed in range(SEED_START, SEED_END):
        SEED = seed
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        print(f"\n--- Starting run for DataLimit={DATA_LIMIT}, Seed={SEED} ---")

        MODEL_PATH = os.path.join(folder, "model.pth")
        assert os.path.exists(MODEL_PATH), f"Model path {MODEL_PATH} does not exist"

        if BACKBONE == "CoAtNet":
            feature_extractor = CoAtNetAE(latent_dim=LATENT_DIM, return_feature=True)
            if not IMAGENET_ONLY: feature_extractor = load_pretrained_weights(feature_extractor, MODEL_PATH)
        elif BACKBONE == "ConvNeXt":
            feature_extractor = ConvNeXtAE(latent_dim=LATENT_DIM, return_feature=True)
            if not IMAGENET_ONLY: feature_extractor = load_pretrained_weights(feature_extractor, MODEL_PATH)
        elif BACKBONE == "FocalNet":
            feature_extractor = FocalNetAE(latent_dim=LATENT_DIM, return_feature=True)
            if not IMAGENET_ONLY: feature_extractor = load_pretrained_weights(feature_extractor, MODEL_PATH)
        elif BACKBONE == "MobileViT":
            feature_extractor = MobileVitAE(latent_dim=LATENT_DIM, return_feature=True)
            if not IMAGENET_ONLY: feature_extractor = load_pretrained_weights(feature_extractor, MODEL_PATH)
        elif BACKBONE == "FocalNet-srf":
            feature_extractor = FocalNetAE(latent_dim=LATENT_DIM, sub_type="srf", return_feature=True)
            if not IMAGENET_ONLY: feature_extractor = load_pretrained_weights(feature_extractor, MODEL_PATH)
        elif BACKBONE == "ResNet50":
            base_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            feature_extractor = nn.Sequential(*list(base_resnet.children())[:-1])
        else:
            raise ValueError(f"Backbone {BACKBONE} not supported in this script.")

        for param in feature_extractor.parameters():
            param.requires_grad = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

        transform = transforms.Compose([SquareTopCropResize(224), transforms.ToTensor(), transforms.Lambda(lambda img: F.adjust_gamma(img, gamma=GAMMA)) if GAMMA != 1.0 else transforms.Lambda(lambda x: x), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_ids, test_ids = split_patient_ids(json_path="data/OpenPOCUS/openPOCUS_filtered.json", data_limit=DATA_LIMIT, seed=SEED)

        train_dataset = OpenPOCUSDataset_new(train_ids,root_dir="data/OpenPOCUS/Lung", transform=transform)
        test_dataset = OpenPOCUSDataset_new(test_ids,root_dir="data/OpenPOCUS/Lung", transform=transform)

        print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

        print("Stage 1: Extracting features from training data...")
        train_feature_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=WORKER_NUM)
        all_train_features, all_train_labels = [], []
        with torch.no_grad():
            for img, label, _ in tqdm(train_feature_loader, desc="Extracting Train Features"):
                img = img.to(device)
                features = feature_extractor(img)[1] if "AE" in feature_extractor.__class__.__name__ else feature_extractor(img)
                all_train_features.append(features.reshape(features.size(0), -1).cpu().numpy())
                all_train_labels.extend(label.cpu().numpy())

        X_train_features = np.concatenate(all_train_features, axis=0)
        y_train_labels = np.array(all_train_labels)

        print(f"\nStage 2: Fitting StandardScaler and {DIM_REDUCTION}...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        
        if DIM_REDUCTION == "PCA":
            dimension_reducer = PCA(n_components=0.95)
            X_train_transformed = dimension_reducer.fit_transform(X_train_scaled)
        elif DIM_REDUCTION == "LDA":
            dimension_reducer = LDA(n_components=1)
            X_train_transformed = dimension_reducer.fit_transform(X_train_scaled, y_train_labels)
        else:  # None
            dimension_reducer = None
            X_train_transformed = X_train_scaled
        

        n_components = X_train_transformed.shape[1]
        
        if dimension_reducer is not None:
            print(f"{DIM_REDUCTION} fitted. Explained variance: {dimension_reducer.explained_variance_ratio_.sum():.4f}, Dimensions: {n_components}")


        print("\nStage 3: Training the classification head...")
        classification_head = ClassificationHead(input_dim=n_components).to(device)
        train_transformed_dataset = TensorDataset(torch.from_numpy(X_train_transformed).float(), torch.from_numpy(y_train_labels).long())
        train_head_loader = DataLoader(train_transformed_dataset, batch_size=16, shuffle=True)
        
        optimizer = torch.optim.AdamW(classification_head.parameters(), lr=1e-3, weight_decay=1e-4)
        ce_loss = nn.CrossEntropyLoss()
        NUM_EPOCH = 3
        for epoch in range(1, NUM_EPOCH + 1):
            classification_head.train()
            for features, labels in train_head_loader:
                features, labels = features.to(device), labels.to(device)
                output = classification_head(features)
                loss = ce_loss(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"    Head Training Epoch {epoch}/{NUM_EPOCH} Done.")
        print("Head training finished")

        print("\nStage 4: Running inference on the test set...")
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=WORKER_NUM)
        classification_head.eval()

        all_test_features_high_dim, all_test_labels, all_pt_ids = [], [], []
        with torch.no_grad():
            for img, label, pt_ids_str in tqdm(test_loader, desc="Extracting Test Features"):
                img = img.to(device)
                features = feature_extractor(img)[1] if "AE" in feature_extractor.__class__.__name__ else feature_extractor(img)
                all_test_features_high_dim.append(features.reshape(features.size(0), -1).cpu().numpy())
                all_test_labels.extend(label.tolist())
                all_pt_ids.extend([remapping(pid) for pid in pt_ids_str])

        X_test_features = np.concatenate(all_test_features_high_dim, axis=0)
        X_test_scaled = scaler.transform(X_test_features)
        if dimension_reducer is not None:
            X_test_transformed = dimension_reducer.transform(X_test_scaled)
        else:
            X_test_transformed = X_test_scaled
        
        with torch.no_grad():
            logits = classification_head(torch.from_numpy(X_test_transformed).float().to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            positive_scores = probs[:, 1].numpy()
            predictions = logits.argmax(dim=1).cpu().numpy()

        y_true_test = np.array(all_test_labels)

        print("\nCalculating final metrics...")
        frame_auc = roc_auc_score(y_true_test, positive_scores)
        print(f"Frame Level AUC = {frame_auc:.4f}")
        print(f"Frame Level Balanced Acc = {balanced_accuracy_score(y_true_test, predictions):.4f}")

        patient_level_results = {}
        for i in range(len(all_pt_ids)):
            pt_id = all_pt_ids[i]
            if pt_id != -1:
                if pt_id not in patient_level_results:
                    patient_level_results[pt_id] = {'scores': [], 'label': all_test_labels[i]}
                patient_level_results[pt_id]['scores'].append(positive_scores[i])

        y_pred_patient, y_true_patient = [], []
        for pt_id, data in sorted(patient_level_results.items()):
            prediction = 1 if np.mean(data['scores']) > 0.5 else 0
            y_pred_patient.append(prediction)
            y_true_patient.append(data['label'])

        res = calc_metrics(np.array(y_true_patient), np.array(y_pred_patient), auc=frame_auc, comment=SEED)
        result_list.append(res)
        print("Patient-level Metrics:", res)

        with open(f"output/Step2_{OUTPUT_NAME}_Fewshot_Holdout_{DATA_LIMIT}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(result_list))
            f.close()
