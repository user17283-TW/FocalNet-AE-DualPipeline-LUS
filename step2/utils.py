#%%
import numpy as np
from torch.nn import functional as F
import torch
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn

class PatchLossCalculator:
    def __init__(self, step_l, step_w, loss_func=nn.MSELoss(reduction='none')):
        self.step_l = step_l
        self.step_w = step_w
        self.loss_func = loss_func
        self.patch_losses = None
        self.positions = None
        self.batch_size = None

    def calc_loss(self, image, reconstruct_img):
        if image.shape != reconstruct_img.shape:
            raise ValueError("兩張圖片的大小必須相等")
        
        B, C, H, W = image.shape
        if H % self.step_l != 0 or W % self.step_w != 0:
            raise ValueError("圖片尺寸必須可被 step 整除")

        self.batch_size = B
        unfold = nn.Unfold(kernel_size=(self.step_l, self.step_w),
                            stride=(self.step_l, self.step_w))
        
        patches_img = unfold(image)     
        patches_rec = unfold(reconstruct_img)
        
        L = patches_img.shape[-1] 
        D = patches_img.shape[1] 

        patches_img = patches_img.transpose(1, 2).reshape(B * L, D)
        patches_rec = patches_rec.transpose(1, 2).reshape(B * L, D)

        loss_tensor = self.loss_func(patches_img, patches_rec)
        patch_loss = loss_tensor.mean(dim=1) if loss_tensor.dim() == 2 else loss_tensor

        self.patch_losses = patch_loss.view(B, L)

        n_h = H // self.step_l
        n_w = W // self.step_w
        self.positions = [(i * self.step_l, j * self.step_w)
                        for i in range(n_h)
                        for j in range(n_w)]

    def max_area(self):
        if self.patch_losses is None:
            raise ValueError("請先執行 calc_loss")

        B, L = self.patch_losses.shape
        flat_loss = self.patch_losses.view(-1)
        max_loss, idx = flat_loss.max(dim=0)
        batch_idx = idx // L
        patch_idx = idx % L
        return max_loss.item(), (batch_idx.item(), self.positions[patch_idx])

def get_mask_matrix(batch_size, W, H, patch_size, mask_ratio):
    assert H % patch_size == 0 and W % patch_size == 0, "H 和 W 必須能被 patch_size 整除"

    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    total_patches = n_patches_h * n_patches_w
    num_mask_patches = int(total_patches * mask_ratio)

    masks = torch.ones((batch_size, H, W))

    for b in range(batch_size):
        mask_indices = np.random.choice(total_patches, size=num_mask_patches, replace=False)

        for idx in mask_indices:
            i = idx // n_patches_w 
            j = idx % n_patches_w 
            masks[b, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = 0

    masks = masks.clone().detach().float()
    return masks

def apply_mask(image, mask):
    assert isinstance(image, torch.Tensor), "image 必須是 torch.Tensor"
    assert isinstance(mask, torch.Tensor), "mask 必須是 torch.Tensor"

    # 確保 mask 形狀為 (H, W)，擴展到 (1, H, W)，然後複製到所有通道 (C, H, W)
    # mask = mask.unsqueeze(0).expand(image.shape)  # (C, H, W)

    # Apply mask, 0 for masked regions, DO NOT MODIFY ON unmasked regions
    masked_image = torch.where(mask == 0, image.clone().detach().fill_(0.0), image)

    return masked_image

def calc_masked_loss(image1, image2, mask, loss_func):
    assert image1.shape == image2.shape
    assert mask.dim() == 3

    device = image1.device
    mask_expanded = mask.unsqueeze(1).expand_as(image1).to(device)
    valid_area = (mask_expanded == 0).float().to(device)
    image2 = image2.to(device)

    masked_image1 = image1 * valid_area
    masked_image2 = image2 * valid_area

    try:
        loss_map = loss_func(masked_image1, masked_image2, reduction='none')
        loss_value = (loss_map * valid_area).sum() / valid_area.sum()
    except TypeError:
        loss_value = loss_func(masked_image1, masked_image2)

    return loss_value

def pixel_base_mask(input_mask, roi_ratio, bg_ratio):
    original_mask = input_mask.clone()
    new_mask = input_mask.clone()

    roi_indices = torch.nonzero(original_mask == 255, as_tuple=True)
    num_roi = roi_indices[0].numel()
    num_to_zero = int(roi_ratio * num_roi)
    if num_to_zero > 0:
        perm = torch.randperm(num_roi)[:num_to_zero]
        new_mask[roi_indices[0][perm], roi_indices[1][perm]] = 0

    bg_indices = torch.nonzero(original_mask == 0, as_tuple=True)
    num_bg = bg_indices[0].numel()
    num_to_one = int((1 - bg_ratio) * num_bg)

    if num_to_one > 0:
        perm = torch.randperm(num_bg)[:num_to_one]
        new_mask[bg_indices[0][perm], bg_indices[1][perm]] = 255

    return new_mask

def image_filter(image_tensor, remove_small_regions=True, regions_limit=5):
    batch_size = image_tensor.shape[0]

    if image_tensor.dim() == 4 and image_tensor.shape[1] == 1:
        image_tensor = image_tensor.squeeze(1)  

    if image_tensor.dim() == 4 and image_tensor.shape[1] == 3:
        image_tensor = image_tensor.permute(0, 2, 3, 1).cpu().numpy()
        image_tensor = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in image_tensor])
        image_tensor = torch.tensor(image_tensor)

    percentile_masks = torch.zeros_like(image_tensor)

    for i in range(batch_size):
        image_np = image_tensor[i].cpu().numpy().astype(np.uint8)

        non_black_mask = image_np > 5
        valid_pixels = image_np[non_black_mask]

        if len(valid_pixels) > 0:
            p10 = np.percentile(valid_pixels, 20) 
            p90 = np.percentile(valid_pixels, 80) 

            percentile_mask = np.zeros_like(image_np)
            percentile_mask[non_black_mask & ((image_np <= p10) | (image_np >= p90))] = 255

            if remove_small_regions:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(percentile_mask, connectivity=8)
                
                new_mask = np.zeros_like(percentile_mask)
                for j in range(1, num_labels): 
                    area = stats[j, cv2.CC_STAT_AREA]
                    if area >= 25:  
                        new_mask[labels == j] = 255

                percentile_mask = new_mask
        else:
            percentile_mask = np.zeros_like(image_np)  

        percentile_masks[i] = torch.tensor(percentile_mask, dtype=torch.uint8)

    return image_tensor, percentile_masks

def image_filter_gpu(image_tensor, dilation_iterations=2):
    if image_tensor.dim() == 4 and image_tensor.shape[1] == 1:
        image_tensor = image_tensor.squeeze(1)
    
    if image_tensor.dim() == 4 and image_tensor.shape[1] == 3:
        if not image_tensor.dtype.is_floating_point:
            image_tensor = image_tensor.float()
        r = image_tensor[:, 0, :, :]
        g = image_tensor[:, 1, :, :]
        b = image_tensor[:, 2, :, :]
        image_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b

    if not image_tensor.dtype.is_floating_point:
        image_tensor = image_tensor.float()

    original_images = image_tensor.clone()
    batch_size = image_tensor.shape[0]
    otsu_thresh_images_list = []



def profile_scalar(img):
    arr = np.array(img)  

    var_r_x = np.var(arr[:, :, 0], axis=0)
    var_g_x = np.var(arr[:, :, 1], axis=0)
    var_b_x = np.var(arr[:, :, 2], axis=0)

    var_r_y = np.var(arr[:, :, 0], axis=1)
    var_g_y= np.var(arr[:, :, 1], axis=1)
    var_b_y = np.var(arr[:, :, 2], axis=1)

    avg_profile_x = np.mean([var_r_x, var_g_x, var_b_x], axis=0)
    avg_profile_y = np.mean([var_r_y, var_g_y, var_b_y], axis=0)

    return avg_profile_x, avg_profile_y

def calc_profile_loss(img1, img2, mode="mean"):
    assert img1.shape == img2.shape, "兩張圖片的大小必須相等"
    assert mode in ["mean", "var", "sum"], "mode 必須是 'mean' 或 'var'"

    # 將 Tensor 轉為 NumPy（格式: CxHxW）
    loss_profile_x_1, loss_profile_y_1 = profile_scalar(img1.cpu().numpy())
    loss_profile_x_2, loss_profile_y_2 = profile_scalar(img2.cpu().numpy())

    if mode == "mean":
        loss_profile = \
            float(np.mean(np.abs(loss_profile_x_1 - loss_profile_x_2))) + \
            float(np.mean(np.abs(loss_profile_y_1 - loss_profile_y_2)))

    elif mode == "var":
        loss_profile = \
            float(np.var(np.abs(loss_profile_x_1 - loss_profile_x_2))) + \
            float(np.var(np.abs(loss_profile_y_1 - loss_profile_y_2)))
        
    elif mode == "sum":
        loss_profile = \
            float(np.sum(np.abs(loss_profile_x_1 - loss_profile_x_2))) + \
            float(np.sum(np.abs(loss_profile_y_1 - loss_profile_y_2)))

    return loss_profile