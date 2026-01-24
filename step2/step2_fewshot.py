# -*- coding: utf-8 -*-
#%%
import os
import json
import random
import numpy as np
import pandas as pd
import torch
from common_func import PatientResult
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.inception import InceptionOutputs

import torchvision.transforms.functional as F
from testing_related import calc_metrics
from dotenv import load_dotenv
import argparse
from AE_models import CoAtNetAE, load_pretrained_weights, ConvNeXtAE, RegNetYAE, MobileVitAE, FocalNetAE
from tqdm import tqdm

# patch_sklearn()

#%%
load_dotenv()

#Load environment variables
DATASET_ROOT = os.getenv("DATASET_ROOT")
TEMPORARY_DATA_PATH = os.getenv("TEMPORARY_DATA_PATH")
PLOT_OUTPUT_DIR = os.getenv("PLOT_OUTPUT_DIR")

parser = argparse.ArgumentParser(description="Step2")

parser.add_argument("--latent_dim", "-l", type=int, default=32, help="latent dimension")
parser.add_argument("--path", "-p", type=str, help="Step1 path", required=True)
parser.add_argument("--thres", "-t", type=str, default="unsup", choices=["unsup", "semisup"], help="Threshold type")
parser.add_argument("--seed", "-s", type=int, default=0, help="Random seed")
parser.add_argument("--profile", action="store_true", help="Use image profile")
parser.add_argument("--workers", type=int, default=64, help="num of CPU workers")
parser.add_argument("--backbone", type=str, default="CoAtNet", choices=["CoAtNet", "ConvNeXt", "RegNetY", "MobileViT", "FocalNet", "ResNet50", "FocalNet-srf", "VGG19", "InceptionV3"], help="Backbone model")
parser.add_argument("--imagenet", action="store_true", help="Use IN")
parser.add_argument("--qc", action="store_true", help="Use QC_mv.json")
parser.add_argument("--output", "-o", type=str, default="step2_results", help="Output file name", required=True)
parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value")


args = parser.parse_args()

LATENT_DIM = args.latent_dim
STEP1_PATH = args.path
USE_UNSUP = args.thres == "unsup"
SEED = args.seed
USE_PROFILE = args.profile
WORKER_NUM = args.workers
BACKBONE = args.backbone
IMAGE_NET_ONLY = args.imagenet
USE_QC = args.qc
OUTPUT_NAME = args.output
GAMMA = args.gamma
SEED_START = SEED
SEED_END = SEED + 25

if(IMAGE_NET_ONLY):
    print("Using ImageNet as pretrained weights")

if(USE_PROFILE):
    print("Using profile loss")

#Find all folder in the path
if os.path.isdir(STEP1_PATH):
    Step1_results = [os.path.join(STEP1_PATH, folder) for folder in os.listdir(STEP1_PATH) if os.path.isdir(os.path.join(STEP1_PATH, folder))]
    print(Step1_results)

if os.path.isfile(os.path.join(STEP1_PATH, "model.pth")):
    Step1_results = [STEP1_PATH]
    print(Step1_results)


class Classifier(nn.Module):
    def __init__(self, base_model):
        super(Classifier, self).__init__()
        self.base_model = base_model
        #Fix the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Linear(1568, 2)

        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x, feature_map = self.base_model(x)
        #flatten the output
        flattenned = feature_map.reshape(feature_map.size(0), -1)
        y = self.fc(flattenned)

        return y, feature_map

class ResNet50_Modified(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith('fc')

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        return self.model(x), None
    
class VGG19_Modified(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        last_fc = self.model.classifier[-1]
        in_features = last_fc.in_features

        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x), None

class InceptionV3_Modified(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.model.fc.parameters():
            p.requires_grad = True

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, InceptionOutputs):
            out = out.logits
        return out, None
#%%
folder = Step1_results[0]

for DATA_LIMIT in [2, 4, 6, 8, 10, 12]:
    result_list = []
    for seed in range(SEED_START, SEED_END):
        SEED = seed
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        print(f"Set seed to {SEED}.")

        #Load the step1 model
        MODEL_PATH = os.path.join(folder, "model.pth")
        TEMPORARY_DATA_PATH = os.path.join(folder, "split_info.csv")

        #Check if the model path exists
        assert os.path.exists(MODEL_PATH), f"Model path {MODEL_PATH} does not exist"
        assert os.path.exists(TEMPORARY_DATA_PATH), f"Dataset info path {TEMPORARY_DATA_PATH} does not exist"

        #Load the model
        if(BACKBONE == "CoAtNet"):
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
            model = FocalNetAE(latent_dim=LATENT_DIM, sub_type="srf", return_feature=True)

        if BACKBONE == "ResNet50":
            model = ResNet50_Modified()
        elif BACKBONE == "VGG19":
            model = VGG19_Modified()
        elif BACKBONE == "InceptionV3":
            model = InceptionV3_Modified()

        else: 
            if(not(IMAGE_NET_ONLY)):
                model = load_pretrained_weights(model, MODEL_PATH)
            model = Classifier(model)

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
                SquareCenterCropResize(299 if BACKBONE=="InceptionV3" else 224),
                transforms.ToTensor(),
                transforms.Lambda(lambda img: F.adjust_gamma(img, gamma=GAMMA)) if GAMMA != 1.0 else transforms.Lambda(lambda x: x),
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

        print("Val Data Label Counts:")
        print(new_val_data['Label'].value_counts())
        print("Test Data Label Counts:")
        print(new_test_data['Label'].value_counts())

        covid_val = new_val_data[new_val_data['Label'] == 'COVID-19'].sample(n=DATA_LIMIT, random_state=SEED)
        other_covid_val = new_val_data[new_val_data['Label'] == 'COVID-19'].drop(covid_val.index)
        reg_val = new_val_data[new_val_data['Label'] == 'regular']
        bac_val = new_val_data[new_val_data['Label'] == 'Bacterial pneumonia']

        non_covid_val = pd.concat([reg_val, bac_val]).sample(n=DATA_LIMIT, random_state=SEED)
        new_val_data = pd.concat([covid_val, non_covid_val], ignore_index=True)

        new_test_data = pd.concat([new_test_data, other_covid_val], ignore_index=True)
    
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

        ce = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-4
        )

        train_loader = DataLoader(
            val_dataset,                   
            batch_size=16,                 
            shuffle=True,
            num_workers=WORKER_NUM,
            pin_memory=True,
            prefetch_factor=4,
            drop_last= True if BACKBONE=="InceptionV3" else False  
        )

        NUM_EPOCH = 3
        for epoch in range(1, NUM_EPOCH + 1):
            model.train()
            running_loss, running_correct, running_total = 0.0, 0, 0

            for img, label, _ in tqdm(train_loader, desc=f"[Epoch {epoch}/{NUM_EPOCH}]"):
                img = img.to(device)

                label_idx = torch.tensor(
                            [1 if l == "COVID-19" else 0 for l in label],   # List → 0/1
                            dtype=torch.long,
                            device=device
                        )

                output, _ = model(img)            
                loss = ce(output, label_idx)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                running_loss += loss.item() * img.size(0)
                running_correct += (output.argmax(dim=1) == label_idx).sum().item()
                running_total += img.size(0)

            epoch_loss = running_loss / running_total
            epoch_acc  = running_correct / running_total
            print(f"    Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        print("Training finished")

        test_dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=WORKER_NUM, prefetch_factor=4)

        roc_data = []
        patient_level_results = []

        for i in range(125):
            patient_level_results.append(PatientResult(i, None))

        model.eval() 
        frame_level_raw = []
        frame_level_class = []
        frame_level_y_true = []
        ids = []
        frame_feature = []

        with torch.no_grad():
            for data in tqdm(test_dataLoader, desc="Test Progress"):
                img, label, patient_id = data 
                patient_id_int = patient_id[0].item()

                original_img = img.clone()
                original_img = original_img.to(device)
                img = img.to(device)

                logits, feature_map = model(img)

                pred_labels = logits.argmax(dim=1).cpu().item()
                probs = torch.softmax(logits, dim=1).cpu()
                positive_scores = probs[:, 1]

                patient_level_results[patient_id-1].add_frame_raw_data(pred_labels)
                patient_level_results[patient_id - 1].label = 1 if label[0] == "COVID-19" else 0

                frame_level_class.append(pred_labels)
                frame_level_raw.append(positive_scores)
                frame_level_y_true.append(1 if label[0] == "COVID-19" else 0)
                ids.append(patient_id_int)


        fpr, tpr, thresholds = roc_curve(frame_level_y_true, frame_level_raw)
        print("Frame Level AUC=", roc_auc_score(frame_level_y_true, frame_level_raw))
        frame_auc = roc_auc_score(frame_level_y_true, frame_level_raw)

        print(f"Frame Level bAcc = {accuracy_score(frame_level_y_true, frame_level_class)}")
        print(f"Frame Level Acc = {accuracy_score(frame_level_y_true, frame_level_class)}")
        print(f"Frame Level Sensitivity = {recall_score(frame_level_y_true, frame_level_class)}")

        patient_level_results = [patient for patient in patient_level_results if len(patient.frame_raw_data) > 0]

        cm = np.zeros((2, 2))
        best_acc = 0

        y_pred = []
        y_true = []

        threshold = 0.7
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
        
        output_path = f"output/Step2_{OUTPUT_NAME}_Fewshot_{DATA_LIMIT}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result_list))
            f.close()

        print(best_acc, cm)
