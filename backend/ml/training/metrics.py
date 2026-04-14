"""
Evaluation metrics for floor plan generation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


class LayoutMetrics:
    """
    Metrics for evaluating layout quality

    Metrics:
    1. Position accuracy (MSE)
    2. Overlap percentage
    3. Area preservation
    4. Room count accuracy
    """

    @staticmethod
    def compute_metrics(
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        batch: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        Compute all metrics

        Args:
            pred_positions: (N, 4) predicted positions
            target_positions: (N, 4) target positions
            batch: (N,) batch assignment

        Returns:
            Dict of metric name -> value
        """
        metrics = {}

        # 1. Position MSE
        metrics['position_mse'] = F.mse_loss(pred_positions, target_positions).item()

        # 2. Position MAE
        metrics['position_mae'] = F.l1_loss(pred_positions, target_positions).item()

        # 3. Overlap percentage
        metrics['overlap_pct'] = LayoutMetrics._compute_overlap_pct(pred_positions, batch)

        # 4. Area preservation
        metrics['area_error'] = LayoutMetrics._compute_area_error(pred_positions, target_positions)

        # 5. Average room size deviation
        metrics['size_deviation'] = LayoutMetrics._compute_size_deviation(pred_positions, target_positions)

        return metrics

    @staticmethod
    def _compute_overlap_pct(
        positions: torch.Tensor,
        batch: torch.Tensor = None
    ) -> float:
        """
        Compute percentage of room pairs that overlap

        Returns:
            Overlap percentage [0, 100]
        """
        N = len(positions)
        if N < 2:
            return 0.0

        if batch is None:
            batch = torch.zeros(N, dtype=torch.long)

        overlaps = 0
        n_pairs = 0

        for i in range(N):
            for j in range(i + 1, N):
                if batch[i] != batch[j]:
                    continue

                if LayoutMetrics._boxes_overlap(positions[i], positions[j]):
                    overlaps += 1

                n_pairs += 1

        if n_pairs == 0:
            return 0.0

        return (overlaps / n_pairs) * 100

    @staticmethod
    def _boxes_overlap(box1: torch.Tensor, box2: torch.Tensor) -> bool:
        """Check if two boxes overlap"""
        x1, y1, w1, h1 = box1.tolist()
        x2, y2, w2, h2 = box2.tolist()

        # Check if boxes intersect
        return not (
            x1 + w1 <= x2 or  # box1 is to the left of box2
            x2 + w2 <= x1 or  # box2 is to the left of box1
            y1 + h1 <= y2 or  # box1 is above box2
            y2 + h2 <= y1     # box2 is above box1
        )

    @staticmethod
    def _compute_area_error(
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor
    ) -> float:
        """Compute relative error in total area"""
        pred_areas = pred_positions[:, 2] * pred_positions[:, 3]
        target_areas = target_positions[:, 2] * target_positions[:, 3]

        pred_total = pred_areas.sum().item()
        target_total = target_areas.sum().item()

        if target_total == 0:
            return 0.0

        return abs(pred_total - target_total) / target_total * 100

    @staticmethod
    def _compute_size_deviation(
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor
    ) -> float:
        """Compute average deviation in room sizes"""
        pred_areas = pred_positions[:, 2] * pred_positions[:, 3]
        target_areas = target_positions[:, 2] * target_positions[:, 3]

        deviations = torch.abs(pred_areas - target_areas) / (target_areas + 1e-6)
        return deviations.mean().item() * 100


class VastuMetrics:
    """
    Metrics for evaluating Vastu compliance

    These are heuristic metrics that check basic Vastu rules.
    """

    @staticmethod
    def compute_vastu_score(
        positions: torch.Tensor,
        node_features: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute Vastu compliance metrics

        Args:
            positions: (N, 4) room positions
            node_features: (N, F) node features (includes room types)

        Returns:
            Dict of metric name -> score [0, 1]
        """
        N = len(positions)
        if N == 0:
            return {'total': 0.0}

        # Extract room types
        room_type_idx = torch.argmax(node_features[:, :11], dim=1)

        scores = []

        for i in range(N):
            x, y, w, h = positions[i].tolist()
            cx, cy = x + w/2, y + h/2  # Centroid

            room_type = room_type_idx[i].item()

            # Kitchen in southeast (x > 0.5, y > 0.5)
            if room_type == 2:  # kitchen
                score = 1.0 if (cx > 0.5 and cy > 0.5) else 0.0
                scores.append(('kitchen_placement', score))

            # Puja room in northeast (x > 0.5, y < 0.5)
            if room_type == 5:  # puja_room
                score = 1.0 if (cx > 0.5 and cy < 0.5) else 0.0
                scores.append(('puja_placement', score))

            # Bathroom not in northeast
            if room_type == 3:  # bathroom
                score = 1.0 if not (cx > 0.5 and cy < 0.5) else 0.0
                scores.append(('bathroom_placement', score))

        if not scores:
            return {'total': 1.0}

        total_score = sum(s for _, s in scores) / len(scores)

        result = {'total': total_score}
        for name, score in scores:
            result[name] = score

        return result


class ImageQualityMetrics:
    """
    Metrics for evaluating image quality in Pix2Pix rendering

    Metrics:
    1. FID (Fréchet Inception Distance) - measures similarity to real images
    2. LPIPS (Learned Perceptual Image Patch Similarity) - perceptual similarity
    3. SSIM (Structural Similarity Index) - structural similarity
    4. L1 distance - pixel-wise reconstruction error
    """

    @staticmethod
    def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Compute Fréchet Inception Distance (FID)

        Lower is better. Measures distribution similarity between
        real and generated images.

        Args:
            real_images: (N, 3, H, W) real images in [-1, 1]
            fake_images: (N, 3, H, W) generated images in [-1, 1]

        Returns:
            FID score (lower is better, typically 0-300)

        Note: This is a simplified implementation. For production,
              use torchmetrics.FrechetInceptionDistance
        """
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance

            fid = FrechetInceptionDistance(feature=2048, normalize=True)

            # Convert from [-1, 1] to [0, 1]
            real_images = (real_images + 1) / 2
            fake_images = (fake_images + 1) / 2

            # Convert to uint8 [0, 255]
            real_images = (real_images * 255).to(torch.uint8)
            fake_images = (fake_images * 255).to(torch.uint8)

            fid.update(real_images, real=True)
            fid.update(fake_images, real=False)

            score = fid.compute()
            return score.item()
        except ImportError:
            print("Warning: torchmetrics not installed, cannot compute FID")
            return -1.0

    @staticmethod
    def compute_lpips(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity)

        Lower is better. Measures perceptual similarity using
        deep features from pretrained networks.

        Args:
            real_images: (N, 3, H, W) real images in [-1, 1]
            fake_images: (N, 3, H, W) generated images in [-1, 1]

        Returns:
            LPIPS score (lower is better, typically 0-1)

        Note: Requires lpips package: pip install lpips
        """
        try:
            import lpips

            # Create LPIPS model (uses VGG by default)
            loss_fn = lpips.LPIPS(net='alex')

            # Compute LPIPS
            with torch.no_grad():
                distances = loss_fn(real_images, fake_images)

            # Average across batch
            return distances.mean().item()
        except ImportError:
            print("Warning: lpips package not installed, cannot compute LPIPS")
            return -1.0

    @staticmethod
    def compute_ssim(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Compute SSIM (Structural Similarity Index)

        Higher is better. Measures structural similarity between images.

        Args:
            real_images: (N, 3, H, W) real images in [-1, 1] or [0, 1]
            fake_images: (N, 3, H, W) generated images in [-1, 1] or [0, 1]

        Returns:
            SSIM score (higher is better, range 0-1)
        """
        # Convert from [-1, 1] to [0, 1] if needed
        if real_images.min() < 0:
            real_images = (real_images + 1) / 2
        if fake_images.min() < 0:
            fake_images = (fake_images + 1) / 2

        # Clamp to [0, 1]
        real_images = torch.clamp(real_images, 0, 1)
        fake_images = torch.clamp(fake_images, 0, 1)

        # Compute SSIM per image
        ssim_values = []
        for i in range(real_images.size(0)):
            ssim_val = ImageQualityMetrics._ssim_single(
                real_images[i:i+1],
                fake_images[i:i+1]
            )
            ssim_values.append(ssim_val)

        return sum(ssim_values) / len(ssim_values)

    @staticmethod
    def _ssim_single(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        size_average: bool = True
    ) -> float:
        """
        Compute SSIM for a single image pair

        Implementation based on:
        Wang et al. "Image quality assessment: from error visibility to
        structural similarity" IEEE Transactions on Image Processing, 2004.
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Create Gaussian window
        channel = img1.size(1)
        window = ImageQualityMetrics._create_gaussian_window(window_size, channel)
        window = window.to(img1.device).type_as(img1)

        # Compute means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean().item()
        else:
            return ssim_map.mean(1).mean(1).mean(1).item()

    @staticmethod
    def _create_gaussian_window(window_size: int, channel: int):
        """Create Gaussian window for SSIM computation"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                np.exp(-(x - window_size//2)**2/float(2*sigma**2))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    @staticmethod
    def compute_l1_distance(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Compute pixel-wise L1 distance

        Args:
            real_images: (N, 3, H, W) real images
            fake_images: (N, 3, H, W) generated images

        Returns:
            L1 distance (lower is better)
        """
        return F.l1_loss(fake_images, real_images).item()


def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """Convenience function for computing FID"""
    return ImageQualityMetrics.compute_fid(real_images, fake_images)


def compute_lpips(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """Convenience function for computing LPIPS"""
    return ImageQualityMetrics.compute_lpips(real_images, fake_images)


def compute_ssim(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """Convenience function for computing SSIM"""
    return ImageQualityMetrics.compute_ssim(real_images, fake_images)


# Import after class definitions to avoid circular import
import torch.nn.functional as F
