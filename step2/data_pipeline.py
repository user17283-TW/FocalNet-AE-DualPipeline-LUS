import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random
import os
import json
from torchvision import transforms
from typing import Union, List, Tuple, Dict
import torch

def split_OpenPOCUS(
    json_path: str, 
    data_limit: int, 
    seed: int
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:

    # 1. 讀取 JSON 檔案
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_normal_patients = data['normal']
    all_abnormal_patients = data['abnormal']

    # 檢查 DATA_LIMIT 是否超出範圍
    if data_limit > len(all_normal_patients):
        raise ValueError(
            f"DATA_LIMIT ({data_limit}) 不能大於 normal 病患的總數 ({len(all_normal_patients)})"
        )
    if data_limit > len(all_abnormal_patients):
        raise ValueError(
            f"DATA_LIMIT ({data_limit}) 不能大於 abnormal 病患的總數 ({len(all_abnormal_patients)})"
        )

    # 2. 設定隨機種子
    random.seed(seed)

    # 3. 隨機抽樣完整的病患物件，建立訓練集
    train_normal_patients = random.sample(all_normal_patients, data_limit)
    train_abnormal_patients = random.sample(all_abnormal_patients, data_limit)

    # 4. 找出剩餘的病患物件，建立測試集
    # 首先，獲取訓練集中所有病患的 ID
    train_normal_ids = {p['id'] for p in train_normal_patients}
    train_abnormal_ids = {p['id'] for p in train_abnormal_patients}

    # 然後，遍歷完整列表，將不在訓練集 ID 中的病患加入測試集
    test_normal_patients = [p for p in all_normal_patients if p['id'] not in train_normal_ids]
    test_abnormal_patients = [p for p in all_abnormal_patients if p['id'] not in train_abnormal_ids]
    
    # 5. 組織並返回結果，排序時指定 key
    # 使用 lambda 函數，告訴 sorted() 根據字典中的 'id' 欄位來排序
    # 將 id 轉為 str 是為了避免混合型別比較的錯誤
    train_set = {
        "normal": sorted(train_normal_patients, key=lambda p: str(p['id'])),
        "abnormal": sorted(train_abnormal_patients, key=lambda p: str(p['id']))
    }
    
    test_set = {
        "normal": sorted(test_normal_patients, key=lambda p: str(p['id'])),
        "abnormal": sorted(test_abnormal_patients, key=lambda p: str(p['id']))
    }

    return train_set, test_set

def load_data(data_path: str) -> pd.DataFrame:

    # 重新加载数据集
    data = pd.read_csv(data_path)

    # 过滤出需要的标签
    required_labels_data = data[
        data["Label"].isin(["Bacterial pneumonia", "regular", "COVID-19"])
    ]

    # 计算每个标签的不同部分所需的数量
    num_total_bacterial = len(
        required_labels_data[required_labels_data["Label"] == "Bacterial pneumonia"]
    )
    num_total_regular = len(
        required_labels_data[required_labels_data["Label"] == "regular"]
    )
    num_total_covid19 = len(
        required_labels_data[required_labels_data["Label"] == "COVID-19"]
    )

    num_train_bacterial = int(num_total_bacterial * 0.7)
    num_val_bacterial = int(num_total_bacterial * 0.1)

    num_train_regular = int(num_total_regular * 0.7)
    num_val_regular = int(num_total_regular * 0.1)

    num_test_covid19 = int(num_total_covid19 * 0.5)

    # 分配训练、验证和测试集
    data.loc[data["Label"] == "Bacterial pneumonia", "Use"] = [
        (
            "Train"
            if i < num_train_bacterial
            else "Val" if i < num_train_bacterial + num_val_bacterial else "Test"
        )
        for i in range(num_total_bacterial)
    ]
    data.loc[data["Label"] == "regular", "Use"] = [
        (
            "Train"
            if i < num_train_regular
            else "Val" if i < num_train_regular + num_val_regular else "Test"
        )
        for i in range(num_total_regular)
    ]
    data.loc[data["Label"] == "COVID-19", "Use"] = [
        "Test" if i < num_test_covid19 else "Val" for i in range(num_total_covid19)
    ]

    # 保存修改后的数据集
    modified_data_path_with_covid = "processed_data_modified.csv"
    data.to_csv(modified_data_path_with_covid, index=False)

    print(modified_data_path_with_covid)

class OpenPOCUSDataset_new(Dataset):
    def __init__(
        self,
        ids: List[str],
        root_dir: str = "../data/OpenPOCUS/Lung",
        transform=None,
        verbose: bool = False,
    ):
        # 設定隨機種子（若提供）
        # if seed is not None:
        #     random.seed(seed)

        self.verbose = verbose
        normal_list = ids['normal']
        abnormal_list = ids['abnormal']

        # 依照 ID 與其類別，組成 (影像路徑, label) 清單
        self.samples: List[Tuple[str, int]] = []

        for entry in normal_list:
            self._gather_images(entry, root_dir, label=0)
        for entry in abnormal_list:
            self._gather_images(entry, root_dir, label=1)

        self.transform = transform
# 
    def _gather_images(self, entry: dict, root_dir: str, label: int):
        """
        依 entry 的 id 與 pos，在對應資料夾下遞迴收集所有影像檔。
        """
        id_str = str(entry['id'])
        for pos in entry.get('pos', []):
            dir_path = os.path.join(root_dir, id_str, pos)
            if not os.path.isdir(dir_path):
                if(self.verbose):
                    print(f"[Warning] Directory {dir_path} does not exist. label={label} Skipping...")
                continue
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                        full_path = os.path.join(root, fname)
                        self.samples.append((full_path, label, id_str))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label, pt_id = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, pt_id
    
class OpenPOCUSDataset(Dataset):
    def __init__(
        self,
        json_path: str = "../data/OpenPOCUS/openPOCUS.json",
        root_dir: str = "../data/OpenPOCUS/Lung",
        use: str = 'val',
        split_ratio: float = 0.7,
        seed: Union[int, None] = None,
        transform=None,
        verbose: bool = False,
    ):
        if seed is not None:
            random.seed(seed)

        self.verbose = verbose
        
        # 讀取 JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f) 
        normal_list = data['normal']
        abnormal_list = data['abnormal']

        random.shuffle(normal_list)
        random.shuffle(abnormal_list)
        n_norm = int(len(normal_list) * split_ratio)
        n_abnorm = int(len(abnormal_list) * split_ratio)
        norm_part1, norm_part2 = normal_list[:n_norm], normal_list[n_norm:]
        abnorm_part1, abnorm_part2 = abnormal_list[:n_abnorm], abnormal_list[n_abnorm:]

        if use == 'val':
            selected_norm = norm_part1
            selected_abnorm = []
            print(f"Validation set: {len(selected_norm)} normal, {len(selected_abnorm)} abnormal")
        elif use == 'test':
            selected_norm = norm_part2
            selected_abnorm = abnorm_part1
            print(f"Test set: {len(selected_norm)} normal, {len(selected_abnorm)} abnormal")
        else:
            raise ValueError(f"Unknown use='{use}'. Choose 'val' or 'test'.")

        self.samples: List[Tuple[str, int]] = []

        for entry in selected_norm:
            self._gather_images(entry, root_dir, label=0)
        for entry in selected_abnorm:
            self._gather_images(entry, root_dir, label=1)

        self.transform = transform
        # print({"normal": selected_norm, "abnormal": selected_abnorm})
# 
    def _gather_images(self, entry: dict, root_dir: str, label: int):

        id_str = str(entry['id'])
        for pos in entry.get('pos', []):
            dir_path = os.path.join(root_dir, id_str, pos)
            if not os.path.isdir(dir_path):
                if(self.verbose):
                    print(f"[Warning] Directory {dir_path} does not exist. label={label} Skipping...")
                continue
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                        full_path = os.path.join(root, fname)
                        self.samples.append((full_path, label, id_str))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label, pt_id = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, pt_id


def randomly_load(
    data_path: str, split_ratio: tuple = (0.5, 0.1, 0.4), covid_ratio: float = 0.5, use_other=False, output_path="temporary_data.csv"
) -> dict:

    assert (
        sum(split_ratio) == 1
    ), f"The sum of the split ratio must be equal to 1 , Got {sum(split_ratio)}"

    data = pd.read_csv(data_path)
    regular_df = data[data["Label"] == "regular"]
    pneumonia_df = data[(data["Label"] == "Bacterial pneumonia") | (data["Label"] == "Pneumonia")]
    covid_df = data[data["Label"] == "COVID-19"]

    # Training set contains 50% of Bacterial pneumonia and regular images
    # Validation set contains 10% of Bacterial pneumonia and regular images
    # Test set contains 40% of Bacterial pneumonia and regular images and 50% of COVID-19 images
    bacterial_training = pneumonia_df.sample(frac=split_ratio[0])
    pneumonia_df = pneumonia_df.drop(bacterial_training.index)
    bacterial_validation = pneumonia_df.sample(
        frac=split_ratio[1] / (1 - split_ratio[0])
    )
    pneumonia_df = pneumonia_df.drop(bacterial_validation.index)
    bacterial_test = pneumonia_df

    regular_training = regular_df.sample(frac=split_ratio[0])
    regular_df = regular_df.drop(regular_training.index)
    regular_validation = regular_df.sample(frac=split_ratio[1] / (1 - split_ratio[0]))
    regular_df = regular_df.drop(regular_validation.index)
    regular_test = regular_df

    if(use_other):
        other_df = data[data["Label"] == "Other"]
        other_training = other_df.sample(frac=split_ratio[0])
        other_df = other_df.drop(other_training.index)
        other_validation = other_df.sample(frac=split_ratio[1] / (1 - split_ratio[0]))
        other_df = other_df.drop(other_validation.index)
        other_test = other_df

    covid_test = covid_df.sample(frac=covid_ratio)
    covid_validation = covid_df.drop(covid_test.index)


    if(use_other):
        training_data = pd.concat([bacterial_training, regular_training, other_training])
        validation_data = pd.concat(
            [bacterial_validation, regular_validation, other_validation, covid_validation]
        )
        test_data = pd.concat([bacterial_test, regular_test, other_test, covid_test])
    else:
        training_data = pd.concat([bacterial_training, regular_training])
        validation_data = pd.concat(
            [bacterial_validation, regular_validation, covid_validation]
        )
        test_data = pd.concat([bacterial_test, regular_test, covid_test])

    # Add a new column to the data to indicate the use of the image
    training_data["Use"] = "Train"
    validation_data["Use"] = "Val"
    test_data["Use"] = "Test"

    # Save the modified data
    modified_data_path = output_path
    pd.concat([training_data, validation_data, test_data]).to_csv(
        modified_data_path, index=False
    )

    # print(test_data.value_counts("Label").to_dict())
    # Get ids of the images in the training, validation and test set and sort them
    training_ids = sorted(training_data["ID"].tolist())
    validation_ids = sorted(validation_data["ID"].tolist())
    test_ids = sorted(test_data["ID"].tolist())

    #Get absolute path of output path
    output_path = os.path.abspath(output_path)

    return {
        "file": output_path,
        "training": training_ids,
        "validation": validation_ids,
        "testing": test_ids,
        "ratios": {
            "training": split_ratio[0],
            "validation": split_ratio[1],
            "testing": split_ratio[2],
            "covid-testing": covid_ratio,
        },
        "counts": {
            "training": training_data.value_counts("Label").to_dict(),
            "validation": validation_data.value_counts("Label").to_dict(),
            "testing": test_data.value_counts("Label").to_dict(),
        }
    }