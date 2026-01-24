import os
import torch
import numpy as np
from torch import nn
import hashlib
import datetime
from typing import Callable, List, Optional

def add_noise_tf(tensors, std):
    import tensorflow as tf
    
    def add_noise_to_tensor(tensor):
        noise = tf.random.normal(shape=tf.shape(tensor), mean=0.0, stddev=std, dtype=tf.float32)
        return tensor + noise
    
    if isinstance(tensors, list):
        return [add_noise_to_tensor(tensor) for tensor in tensors]
    else:
        return add_noise_to_tensor(tensors)
    
def add_noise(_tensors: torch.Tensor, _std: float, _mean=0, use_gpu=True):
    def add_noise_to_tensor(_tensor):
        device = _tensor.device if use_gpu else 'cpu'
        noise = torch.randn(_tensor.size(), device=device) * _std + _mean
        return _tensor + noise
    
    if isinstance(_tensors, list):
        return [add_noise_to_tensor(tensor) for tensor in _tensors]
    else:
        return add_noise_to_tensor(_tensors)

def generate_current_hash():
    def to_base36(num):
        # Convert a number to base36
        chars = '0123456789abcdefghijklmnopqrstuvwxyz'
        result = ''
        while num:
            num, i = divmod(num, 36)
            result = chars[i] + result
        return result

    # Get the current time and format it
    current_time = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    # Use SHA-256 hash function
    hash_object = hashlib.sha256(current_time.encode())
    # Get the hexadecimal digest of the hash
    hex_dig = hash_object.hexdigest()
    # Convert it to an integer
    num = int(hex_dig, 16)
    # Convert to base36
    base36_hash = to_base36(num)
    # Keep only the first 5 characters
    return base36_hash[:5]

def decide_threshold(data, precent):
    assert precent >= 0 and precent <= 100, "precent should be in [0, 100]"
    
    return np.percentile(data, precent)

# def decide_threshold(data, precent):
#     return 0.04

def thresholding(x, threshold):
    return int(x > threshold)

def thresholding_clf(x, clf):
    x = np.array(x).reshape(-1, 1)
    return clf.predict(x)

class EarlyStopping:
    def __init__(self, patience=7, chk_point_file = "checkpoint.pth", verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_epoch = 0
        self.delta = delta
        
        if(os.path.isdir(chk_point_file)):
            self.chk_point_file = os.path.join(chk_point_file, "checkpoint.pth")
        else:
            self.chk_point_file = chk_point_file

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch + 1 # 0-based to 1-based
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model.state_dict(), self.chk_point_file)  # 這裡保存模型的最佳狀態
        self.val_loss_min = val_loss
    def load_checkpoint(self, model):

        #Check if the checkpoint file exists
        try:
            model.load_state_dict(torch.load(self.chk_point_file))
        except FileNotFoundError:
            print(f"File {self.chk_point_file} not found.")
            return None
        return model

class PatientResult():
    def __init__(self, patient_id, label):
        self.patient_id = patient_id
        self.label = label
        self.frame_raw_data = []
        self.frame_classes_data = []
        self.predicted_label = None
        self.borderline_flag = False

    def compute_upper_whisker(self, data: list[float]) -> float:
        data = np.array(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        return data[data <= upper_bound].max()


    def add_frame_raw_data(self, frame_data):
        if(type(frame_data) != list):
            self.frame_raw_data.append(frame_data)
        else:
            #flatten the frame_data DO NOT USE np.flatten()!
            for item in frame_data:
                self.frame_raw_data.append(item)

    def classification(self, clf, reduce=False):
        assert len(self.frame_raw_data) > 0, "frame_raw_data is empty, run add_frame_raw_data() first."

        if reduce:
            self.frame_raw_data = [np.median(self.frame_raw_data)]

        if isinstance(clf, (int, float)):  # threshold
            self.frame_classes_data = [thresholding(frame_data, clf) for frame_data in self.frame_raw_data]
        else:  # classifier
            self.frame_classes_data = [thresholding_clf(frame_data, clf) for frame_data in self.frame_raw_data]

        return self.frame_classes_data
    
    def predict_label(self, threshold=0.5):
        assert len(self.frame_classes_data) > 0, "frame_classes_data is empty, run get_frame_classes_data() first."

        #majority vote for frames_classes_data
        average = sum(self.frame_classes_data) / len(self.frame_classes_data)

        if(average >0.3 and average < 0.7):
            self.borderline_flag = True

        self.predicted_label = 1 if average > threshold else 0
        return self.predicted_label
    
    def override_pred(self, label):
        self.predicted_label = label

    def __str__(self):
        return f"Patient {self.patient_id}, label: {self.label}"

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

def load_step1(path: str, latent_dim: int = None):

    MODEL = os.path.join(path, "model.pth")
    DATA_SPLIT_FILE = os.path.join(path, "split_info.csv")

    #Check if the model file exists
    if not os.path.isfile(MODEL):
        print(f"Model file {MODEL} does not exist.")
        return None, None, None

    #Check if the data split file exists
    if not os.path.isfile(DATA_SPLIT_FILE):
        print(f"Data split file {DATA_SPLIT_FILE} does not exist.")
        return None, None, None

    return MODEL, DATA_SPLIT_FILE, int(latent_dim)

def sliding_window_losses(
    output: torch.Tensor,
    img: torch.Tensor,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    window_size: int = 14,
    stride: Optional[int] = None
) -> List[float]:
    
    # 檢查 shape 與 dtype
    assert output.shape == img.shape, "output 與 img 的 shape 必須相同"
    assert output.ndim == 4, "輸入張量必須為 4D (B, C, H, W)"
    assert output.dtype == torch.float32 and img.dtype == torch.float32, \
        "張量 dtype 必須為 torch.float32"

    if stride is None:
        stride = window_size

    B, C, H, W = output.shape
    losses: List[float] = []

    for b in range(B):
        for top in range(0, H - window_size + 1, stride):
            for left in range(0, W - window_size + 1, stride):
                patch_out = output[b, :, top:top+window_size, left:left+window_size]
                patch_img = img[b, :, top:top+window_size, left:left+window_size]
                l = loss_func(patch_out, patch_img)
                losses.append(l.item())

    return losses

def sliding_window_mse(
    output: torch.Tensor,
    img: torch.Tensor,
    window_size: int = 14,
    stride: int = None
) -> List[float]:

    assert output.shape == img.shape, "output 與 img 必須同 shape"
    assert output.ndim == 4, "輸入張量需為 4D (B, C, H, W)"
    assert output.dtype == torch.float32, "dtype 必須是 torch.float32"

    if stride is None:
        stride = window_size

    B, C, H, W = output.shape

    unfold = torch.nn.Unfold(kernel_size=window_size, stride=stride)
    patches_out = unfold(output)  # [B, F, L]
    patches_img = unfold(img)

    diffs = patches_out - patches_img
    mse_per_patch = diffs.pow(2).mean(dim=1)

    return mse_per_patch.reshape(-1).tolist()