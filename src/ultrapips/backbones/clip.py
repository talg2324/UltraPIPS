import torch
import open_clip

from .utils import ImageNormalizer
from .base import UltrasoundImageEncoder


class CLIPBackbone(UltrasoundImageEncoder):
    """
    Backbone wrapper for CLIP-based models (Standard CLIP, BiomedCLIP, Ultrasound-CLIP).
    Expects ViT-B-16 architecture.
    """

    def __init__(self, model_type: str, weights_path: str | None = None):
        super().__init__(weights_path)
        self.model_type = model_type.lower()

        if self.model_type == "clip":
            # Standard OpenAI CLIP
            model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
            self.model = model.visual
            self.model_type_internal = "transformer"
            
            # Normalization for OpenAI CLIP
            self.normalizer = ImageNormalizer(
                img_size=224,
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
                num_output_channels=3
            )
        elif self.model_type in ["biomedclip", "ultrasound_clip"]:
            # BiomedCLIP or its derivative Ultrasound-CLIP
            model, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            self.model = model.visual
            self.model_type_internal = "timm"

            if self.model_type == "ultrasound_clip":
                if weights_path is None:
                    raise ValueError("Ultrasound-CLIP requires weights_path (checkpoint).")
                
                ckpt = torch.load(weights_path, map_location="cpu")
                state_dict = ckpt["state_dict"]
                
                # Extract base_clip.visual weights
                # The state dict has keys like 'base_clip.visual.trunk.blocks.0.norm1.weight'
                # While self.model (model.visual) expects 'trunk.blocks.0.norm1.weight'
                visual_state_dict = {
                    k.replace("base_clip.visual.", ""): v 
                    for k, v in state_dict.items() 
                    if k.startswith("base_clip.visual.")
                }
                
                if not visual_state_dict:
                    # Try alternate mapping if base_clip.visual prefix is missing
                    visual_state_dict = {
                        k: v for k, v in state_dict.items() if "trunk" in k or "head" in k
                    }

                msg = self.model.load_state_dict(visual_state_dict, strict=False)
                print(f"Loaded Ultrasound-CLIP weights from {weights_path}. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")

            # Normalization for BiomedCLIP / Ultrasound-CLIP
            # Note: Verified via open_clip.create_model_and_transforms that BiomedCLIP 
            # uses standard OpenAI CLIP normalization constants.
            self.normalizer = ImageNormalizer(
                img_size=224,
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
                num_output_channels=3
            )
        else:
            raise ValueError(f"Unknown CLIP model type: {model_type}")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.normalizer(x)
        features = []

        if self.model_type_internal == "transformer":
            # VisionTransformer (Standard CLIP from open-clip)
            # x shape: (B, C, H, W)
            x = self.model.conv1(x)  # shape = [B, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [B, width, L]
            x = x.permute(0, 2, 1)  # shape = [B, L, width]
            
            # Add cls token
            cls_token = self.model.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            
            # Add pos embedding
            x = x + self.model.positional_embedding.to(x.dtype)
            x = self.model.ln_pre(x)

            for block in self.model.transformer.resblocks:
                x = block(x)
                # Remove cls token and reshape to (B, C, H, W)
                f = x[:, 1:, :]
                B, L, C = f.shape
                H = W = int(L**0.5)
                f = f.permute(0, 2, 1).reshape(B, C, H, W)
                features.append(f)

        elif self.model_type_internal == "timm":
            # TimmModel (BiomedCLIP / Ultrasound-CLIP)
            trunk = self.model.trunk
            
            # Forward patches
            x = trunk.patch_embed(x)
            x = trunk._pos_embed(x)
            x = trunk.patch_drop(x)
            x = trunk.norm_pre(x)

            for block in trunk.blocks:
                x = block(x)
                # Remove prefix tokens (cls) and reshape
                f = x[:, trunk.num_prefix_tokens:, :]
                B, L, C = f.shape
                H = W = int(L**0.5)
                f = f.permute(0, 2, 1).reshape(B, C, H, W)
                features.append(f)

        return features
