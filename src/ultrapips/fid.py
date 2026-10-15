from typing import Iterable, NamedTuple

import torch
import torch.nn as nn

from .loss import BackboneType, build_encoder


class FIDStats(NamedTuple):
    """
    Gaussian fit of an image set's embeddings: mean (D,) and covariance (D, D).

    Embedding is the expensive part of FID. When one set is compared against many others,
    compute its stats once with `FID.stats` and pass them in place of the images.
    Stats are only comparable with stats from the same backbone. Save/load with torch.save / torch.load.
    """
    mu: torch.Tensor
    sigma: torch.Tensor


class FID(nn.Module):
    """
    Fréchet distance between two image sets in a backbone's embedding space.

    Example:
        fid = FID(backbone="usfm", model_path="weights/USFM.pt")
        score = fid(DataLoader(set_a, batch_size=32), DataLoader(set_b, batch_size=32))

        # Reuse one set's stats across comparisons
        stats_a = fid.stats(DataLoader(set_a, batch_size=32))
        scores = [fid(stats_a, DataLoader(s, batch_size=32)) for s in other_sets]
    """

    def __init__(self, backbone: str | BackboneType, model_path: str | None = None):
        super().__init__()
        self.encoder = build_encoder(backbone, model_path)

    @torch.inference_mode()
    def stats(self, batches: Iterable[torch.Tensor]) -> FIDStats:
        """
        Args:
            batches: iterable of (B, C, H, W) image batches (e.g. a DataLoader), same format as UltraPIPS.
                Needs more images in total than embedding dims.
        """
        from tqdm import tqdm

        self.eval()
        device = next(self.parameters()).device
        feats = torch.cat([self.encoder.embed(x.to(device)).double().cpu() for x in tqdm(batches, leave=False)])

        # The covariance estimate is rank-deficient unless there are more samples than feature dims
        n, d = feats.shape
        if n <= d:
            raise ValueError(f"FID needs more samples than embedding dims, got {n} samples, {d} dims")

        return FIDStats(feats.mean(dim=0), torch.cov(feats.T))

    def forward(self, x: Iterable[torch.Tensor] | FIDStats, y: Iterable[torch.Tensor] | FIDStats) -> float:
        """
        Args:
            x, y: iterables of image batches (see `stats`) or precomputed FIDStats.
        """
        x = x if isinstance(x, FIDStats) else self.stats(x)
        y = y if isinstance(y, FIDStats) else self.stats(y)

        # Tr(sqrtm(Sx @ Sy)) as the sum of sqrt eigenvalues of Sx @ Sy, as in torchmetrics' _compute_fid
        tr_covmean = torch.linalg.eigvals(x.sigma @ y.sigma).sqrt().real.sum()
        return ((x.mu - y.mu).square().sum() + x.sigma.trace() + y.sigma.trace() - 2 * tr_covmean).item()
