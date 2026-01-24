import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CoAtNetAE(nn.Module):
    def __init__(self, img_size=224, latent_dim=512, return_feature=False, ImageNet_pretrained=True):
        super(CoAtNetAE, self).__init__()

        self.return_feature = return_feature

        # Use FastViT-8T as Encoder basis
        base_model = timm.create_model("coatnet_0_rw_224", pretrained=ImageNet_pretrained)
    
        # Extract the backbone only (without classification head)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        
        # Calculate feature map size at encoder output
        # FastViT-T8 has 32x downsampling to feature maps
        feature_size = img_size // 32
        
        # Get channel count from FastViT-T8 (should be 512 for t8 variant)
        # You might need to adjust this based on actual model output
        encoder_channels = 768  # Adjust if needed after checking

        # Create a proper bottleneck
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((feature_size, feature_size)),
            nn.Conv2d(encoder_channels, latent_dim, kernel_size=1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU()
        )
        
        # Improved decoder with skip connections
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Using Tanh instead of Sigmoid for better gradients
        )

    def forward(self, x,):
        # Extract features through encoder
        features = self.encoder(x)
        # print(features.shape)

        # Apply bottleneck
        latent = self.bottleneck(features)
        
        # Decode
        x = self.decoder_block1(latent)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)
        x = self.decoder_block4(x)
        reconstructed = self.final_layer(x)
        
        if(self.return_feature):
            return reconstructed, latent

        return reconstructed

class MobileVitAE(nn.Module):
    def __init__(self, img_size=224, latent_dim=512, return_feature=False, ImageNet_pretrained=True):
        super(MobileVitAE, self).__init__()

        self.return_feature = return_feature

        # Use FastViT-8T as Encoder basis
        base_model = timm.create_model("mobilevitv2_200", pretrained=ImageNet_pretrained)
    
        # Extract the backbone only (without classification head)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        
        # Calculate feature map size at encoder output
        # FastViT-T8 has 32x downsampling to feature maps
        feature_size = 7
        
        # Get channel count from FastViT-T8 (should be 512 for t8 variant)
        # You might need to adjust this based on actual model output
        encoder_channels = 1024  # Adjust if needed after checking

        # Create a proper bottleneck
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((feature_size, feature_size)),
            nn.Conv2d(encoder_channels, latent_dim, kernel_size=1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU()
        )
        
        # Improved decoder with skip connections
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Using Tanh instead of Sigmoid for better gradients
        )

    def forward(self, x):
        # Extract features through encoder
        features = self.encoder(x)
        # print(features.shape)

        # Apply bottleneck
        latent = self.bottleneck(features)
        
        # Decode
        x = self.decoder_block1(latent)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)
        x = self.decoder_block4(x)
        reconstructed = self.final_layer(x)
        
        if(self.return_feature):
            return reconstructed, latent

        return reconstructed
    
class FocalNetAE(nn.Module):
    def __init__(self, img_size=224, latent_dim=512, return_feature=False, ImageNet_pretrained=True, sub_type="lrf"):
        super(FocalNetAE, self).__init__()

        assert sub_type in ["lrf", "srf"], "rf must be either 'lrf' or 'srf'"
        self.return_feature = return_feature

        # Use FastViT-8T as Encoder basis
        base_model = timm.create_model(f"focalnet_tiny_{sub_type}", pretrained=ImageNet_pretrained)
    
        # Extract the backbone only (without classification head)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        
        # Calculate feature map size at encoder output
        # FastViT-T8 has 32x downsampling to feature maps
        feature_size = img_size // 32
        
        # Get channel count from FastViT-T8 (should be 512 for t8 variant)
        # You might need to adjust this based on actual model output
        encoder_channels = 768  # Adjust if needed after checking

        # Create a proper bottleneck
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((feature_size, feature_size)),
            nn.Conv2d(encoder_channels, latent_dim, kernel_size=1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU()
        )
        
        # Improved decoder with skip connections
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Using Tanh instead of Sigmoid for better gradients
        )

    def forward(self, x):
        # Extract features through encoder
        features = self.encoder(x)
        # print(features.shape)

        # Apply bottleneck
        latent = self.bottleneck(features)
        
        # Decode
        x = self.decoder_block1(latent)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)
        x = self.decoder_block4(x)
        reconstructed = self.final_layer(x)
        
        if(self.return_feature):
            return reconstructed, latent

        return reconstructed
    
class RegNetYAE(nn.Module):
    def __init__(self, img_size=224, latent_dim=512, return_feature=False, ImageNet_pretrained=True):
        super(RegNetYAE, self).__init__()

        self.return_feature = return_feature

        # Use FastViT-8T as Encoder basis
        base_model = timm.create_model("regnety_032", pretrained=ImageNet_pretrained)
    
        # Extract the backbone only (without classification head)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        
        # Calculate feature map size at encoder output
        # FastViT-T8 has 32x downsampling to feature maps
        feature_size = img_size // 32
        
        # Get channel count from FastViT-T8 (should be 512 for t8 variant)
        # You might need to adjust this based on actual model output
        encoder_channels = 1512  # Adjust if needed after checking

        # Create a proper bottleneck
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((feature_size, feature_size)),
            nn.Conv2d(encoder_channels, latent_dim, kernel_size=1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU()
        )
        
        # Improved decoder with skip connections
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Using Tanh instead of Sigmoid for better gradients
        )

    def forward(self, x):
        # Extract features through encoder
        features = self.encoder(x)
        # print(features.shape)

        # Apply bottleneck
        latent = self.bottleneck(features)

        #Merge brightness prediction with latent
        # brightness = brightness.view(-1, 1, 1, 1).expand(-1, 1, 7, 7)
        # latent = torch.cat((latent, brightness), dim=1)
        
        # Decode
        x = self.decoder_block1(latent)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)
        x = self.decoder_block4(x)
        reconstructed = self.final_layer(x)
        
        if(self.return_feature):
            return reconstructed, latent

        return reconstructed
    
class ConvNeXtAE(nn.Module):
    def __init__(self, img_size=224, latent_dim=512, return_feature=False, ImageNet_pretrained=True):
        super(ConvNeXtAE, self).__init__()

        self.return_feature = return_feature

        # Use FastViT-8T as Encoder basis
        base_model = timm.create_model("convnextv2_tiny", pretrained=ImageNet_pretrained)
    
        # Extract the backbone only (without classification head)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        
        # Calculate feature map size at encoder output
        # FastViT-T8 has 32x downsampling to feature maps
        feature_size = img_size // 32
        
        # Get channel count from FastViT-T8 (should be 512 for t8 variant)
        # You might need to adjust this based on actual model output
        encoder_channels = 768  # Adjust if needed after checking

        # Create a proper bottleneck
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((feature_size, feature_size)),
            nn.Conv2d(encoder_channels, latent_dim, kernel_size=1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU()
        )
        
        # Improved decoder with skip connections
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Using Tanh instead of Sigmoid for better gradients
        )

    def forward(self, x):
        # Extract features through encoder
        features = self.encoder(x)
        # print(features.shape)

        # Apply bottleneck
        latent = self.bottleneck(features)
        
        # Decode
        x = self.decoder_block1(latent)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)
        x = self.decoder_block4(x)
        reconstructed = self.final_layer(x)
        
        if(self.return_feature):
            return reconstructed, latent

        return reconstructed


    def __init__(self, img_size=224, latent_dim=512, return_feature=False, ImageNet_pretrained=True):
        super(SwinAE, self).__init__()

        self.return_feature = return_feature

        # 使用 features_only 來取得 CNN-style 特徵圖（[B, C, H, W]）
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=ImageNet_pretrained,
            features_only=True
        )

        # 根據 Swin-Tiny 輸出特徵圖維度
        # 最後一層輸出為 [B, 768, 7, 7]
        encoder_channels = self.encoder.feature_info[-1]['num_chs']
        feature_size = self.encoder.feature_info[-1]['reduction']  # 32 → 224/32=7

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # 強制保留 spatial 結構
            nn.Conv2d(encoder_channels, latent_dim, kernel_size=1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU()
        )

        # Decoder blocks
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.decoder_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 取最後一層 feature（最語意層）
        features = self.encoder(x)
        x = features[-1]  # shape: [B, 768, 7, 7]
        x = x.permute(0, 3, 1, 2) if x.shape[-1] == 768 else x 

        latent = self.bottleneck(x)

        # Decode
        x = self.decoder_block1(latent)  # [B, 256, 14, 14]
        x = self.decoder_block2(x)       # [B, 128, 28, 28]
        x = self.decoder_block3(x)       # [B, 64, 56, 56]
        x = self.decoder_block4(x)       # [B, 32, 112, 112]
        reconstructed = self.final_layer(x)  # [B, 3, 224, 224]

        if self.return_feature:
            return reconstructed, latent

        return reconstructed
    
class AE(nn.Module):
        def __init__(self, latent_dim=172, sec_dim=405, dropout_rate=0):
            super(AE, self).__init__()

            self.encoder = nn.Sequential(
                nn.Linear(3 * 224 * 224, 2048),  # Adjust the input size
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                nn.Linear(1024, sec_dim),  # Adjust the input size
                nn.ReLU(True),
                # nn.Linear(64,8),
            )

            self.bottleneck = nn.Linear(sec_dim, latent_dim)

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, sec_dim),
                nn.ReLU(True),
                nn.Linear(sec_dim, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 3 * 224 * 224),
                nn.Tanh(),
            )  

        def forward(self, x):
            # x = x.view(-1, 3, 64, 64)
            x = x.view(x.size(0), -1)
            latent_space = self.encode(x)
            y = self.decode(latent_space)

            return y
        
        def encode(self, x):
            x = self.encoder(x)
            return self.bottleneck(x)
        
        def decode(self, x: torch.Tensor) -> torch.Tensor:
            x = self.decoder(x)
            #reshape x to img = 3, 224, 224
            x = x.view(x.size(0), 3, 224, 224)
            return x
            
def load_pretrained_weights(model, weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    
    model.load_state_dict(checkpoint)
    return model

class VaeLoss(nn.Module):
    def __init__(self, kld_weight: int = 1):
        super(VaeLoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor, mu: torch.Tensor, logsigma: torch.Tensor, mse=None) -> torch.Tensor:
        if(mse is None):
            mse = F.mse_loss(x, y, reduction='mean')
        # reconstruct_loss = F.binary_cross_entropy(output, img_flatten,reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp()) / x.shape[0]
        loss = mse + self.kld_weight * kld_loss

        return loss