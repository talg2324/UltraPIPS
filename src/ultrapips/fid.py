import torch
from tqdm import tqdm
from monai.metrics.fid import get_fid_score
from torch.utils.data import Dataset, DataLoader

from .loss import UltraPIPS, BackboneType


class FID(UltraPIPS):
    def __init__(
        self,
        backbone: str | BackboneType = BackboneType.TUSA_VIT,
        model_path: str | None = None,
    ):
        # We don't need a specific reduction for FID since it returns a single score
        super().__init__(backbone=backbone, model_path=model_path, reduction="mean")

    def _extract_features(self, loader: DataLoader, device: torch.device, desc: str) -> torch.Tensor:
        all_feats = []
        for batch in tqdm(loader, desc=desc, leave=False):
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            elif isinstance(batch, dict):
                # Fallback for dict-based datasets (common in monai/huggingface)
                x = batch.get('image', batch.get('img', next(iter(batch.values()))))
            else:
                x = batch

            x = x.to(device)

            with torch.no_grad():
                feats = self.encoder(x)
                final_feats = feats[-1]

                # Spatial mean pooling if it's a spatial feature map
                if final_feats.ndim == 4:
                    final_feats = final_feats.mean(dim=[2, 3])

                all_feats.append(final_feats.cpu())

        return torch.cat(all_feats, dim=0)

    def forward(self, dataset_real: Dataset, dataset_fake: Dataset, batch_size: int = 32) -> torch.Tensor:
        """
        Compute the Fréchet Inception Distance between two datasets.

        Args:
            dataset_real: Dataset containing real images
            dataset_fake: Dataset containing generated/fake images
            batch_size: Batch size for feature extraction
        """

        device = next(self.parameters()).device

        loader_real = DataLoader(dataset_real, batch_size=batch_size, shuffle=False)
        loader_fake = DataLoader(dataset_fake, batch_size=batch_size, shuffle=False)

        self.eval()

        real_features = self._extract_features(loader_real, device, desc="Extracting Real Features")
        fake_features = self._extract_features(loader_fake, device, desc="Extracting Fake Features")

        # get_fid_score uses double precision and will fallback to scipy CPU logic when necessary
        fid_score = get_fid_score(y_pred=fake_features, y=real_features)

        # Return on same device as the model
        return fid_score.to(device)
