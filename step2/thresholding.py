from scipy.signal import savgol_filter  # Import savgol_filter
import numpy as np
from scipy.stats import gaussian_kde

def find_local_valley_threshold(
    scores: np.ndarray,
    grid_size: int = 1000,
    bandwidth: str = 'scott',
    min_percentile: float = 25.0,
    max_percentile: float = 75.0,
    smooth_window: int = 5,
    draw_plot: bool = True,
    plot_comment: str = "",
) -> (float, float):
    
    kde = gaussian_kde(scores, bw_method=bandwidth)
    grid = np.linspace(scores.min(), scores.max(), grid_size)
    density = kde(grid)

    if smooth_window % 2 == 0:
        smooth_window += 1
    
    density_raw = density 
    poly_order = 2
    density = savgol_filter(density_raw, smooth_window, poly_order)

    lower = np.percentile(scores, min_percentile)
    iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
    upper = np.percentile(scores, 75) + 1.5 * iqr

    mask = (grid >= lower) & (grid <= upper)
    valid_idxs = np.where(mask)[0]
    if valid_idxs.size == 0:
        threshold = upper
    else:
        segment = density[valid_idxs]
        local_min_idx = valid_idxs[np.argmin(segment)]
        threshold = grid[local_min_idx]

    percentile = np.mean(scores <= threshold) * 100.0
    return threshold, percentile