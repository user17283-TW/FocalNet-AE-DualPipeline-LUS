#%%
import numpy as np
from torch.nn import functional as F
from typing import Optional
import torch
import torchvision.transforms.functional as TF
import cv2
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


def generate_brightness_variants(img_batch, bright_factor=1.2, dark_factor=0.8):
    x_orig = img_batch
    x_bright = torch.stack([TF.adjust_brightness(img, bright_factor) for img in img_batch])
    x_dark = torch.stack([TF.adjust_brightness(img, dark_factor) for img in img_batch])
    return x_orig, x_bright, x_dark

def simple_adjust_brightness(img_batch, bright_factor=1.2, dark_factor=0.8):
    x_orig = img_batch
    x_bright = torch.clamp(x_orig * bright_factor, -1, 1)
    x_dark = torch.clamp(x_orig * dark_factor, -1, 1)
    return x_orig, x_bright, x_dark

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

def apply_mask(image: torch.Tensor, mask: torch.Tensor, mask_token: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert image.dim() == 4, f"image shape 必須是 (B, C, H, W)，但得到 {image.shape}"
    B, C, H, W = image.shape

    if mask.shape[1] == 1:
        mask_expanded = mask.expand(-1, C, -1, -1)
    elif mask.shape[1] == C:
        mask_expanded = mask
    else:
        raise ValueError(f"mask shape 不符合: {mask.shape} (預期 (B, 1, H, W) 或 (B, C, H, W))")

    mask_expanded = mask_expanded.to(dtype=image.dtype, device=image.device)

    if mask_token is not None:
        if mask_token.dim() == 1 and mask_token.shape[0] == C:
            mask_tok = mask_token.view(1, C, 1, 1).expand(B, C, H, W)
        elif mask_token.dim() == 3 and mask_token.shape[0] == C:
            mask_tok = mask_token.view(1, C, 1, 1).expand(B, C, H, W)
        elif mask_token.dim() == 4 and mask_token.shape[1] == C:
            mask_tok = mask_token.expand(B, C, H, W)
        else:
            raise ValueError(f"mask_token shape 不符合: {mask_token.shape} (預期 (C,), (C,1,1), (1,C,1,1))")

        mask_tok = mask_tok.to(dtype=image.dtype, device=image.device)
    else:
        mask_tok = torch.zeros((B, C, H, W), dtype=image.dtype, device=image.device)

    masked_image = image * mask_expanded + mask_tok * (1 - mask_expanded)

    return masked_image

def calc_masked_loss(image1, image2, mask, loss_func, target_mask=None):
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
        loss_value = loss_func(masked_image1, masked_image2, target_mask=target_mask)

    return loss_value

def pixel_base_mask(input_mask, roi_ratio, bg_ratio):
    original_mask = input_mask.clone()
    new_mask = input_mask.clone()

    roi_indices = torch.nonzero(original_mask.cpu() == 255, as_tuple=True)

    num_roi = roi_indices[0].numel()
    num_to_zero = int(roi_ratio * num_roi)
    if num_to_zero > 0:
        perm = torch.randperm(num_roi)[:num_to_zero]
        new_mask[roi_indices[0][perm], roi_indices[1][perm]] = 0

    bg_indices = torch.nonzero(original_mask.cpu() == 0, as_tuple=True)
    bg_indices = tuple(idx.to(original_mask.device) for idx in bg_indices)
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