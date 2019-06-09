from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from datetime import datetime
import os
import glob


def load_folder(path):
    files = glob.glob(os.path.join(path, "*.csv"))
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    return df


def get_timestamp():
    now = datetime.now()
    timestamp = "{}-{}-{}_{}-{}-{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    return timestamp


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_results_dir(data_path, model_type="rf", title_suffix=""):
    timestamp = get_timestamp()
    dataset_name = os.path.basename(data_path).split(".")[0]
    result_dir = os.path.join("results", *[model_type, dataset_name, "{}{}".format(title_suffix, timestamp)])
    model_dir = os.path.join(result_dir, "model")
    log_dir = os.path.join(result_dir, "logs")

    make_dir(model_dir)
    make_dir(log_dir)

    return model_dir, log_dir


def load_fits(filepath):
    info_headers = ["LIB3", "LIB2", "LIB1", "DISTANCE", "JD", "ALPHA", "PEDESTAL", "PHASE", "ALBEDO"]
    with fits.open(filepath, memmap=False) as f:
        img = f[0].data
        info = [f[0].header[header] for header in info_headers]
    return img, (info, info_headers)


def img_to_patches(img, patch_shape):
    img_h, img_w = np.shape(img)
    patch_h, patch_w = patch_shape

    grid_x = np.concatenate([np.arange(0, img_w, patch_w), [img_w]])
    grid_y = np.concatenate([np.arange(0, img_h, patch_h), [img_h]])
    grid_width = len(grid_x) - 1
    grid_height = len(grid_y) - 1

    patches = np.zeros((grid_height, grid_width, patch_h, patch_w), dtype=img.dtype)

    for i in range(grid_height):
        for j in range(grid_width):
            a = np.repeat(list(range(grid_y[i], grid_y[i + 1])), repeats=patch_shape[1])
            b = np.tile(list(range(grid_x[j], grid_x[j + 1])), reps=patch_shape[0])
            patches[i, j] = img[(a, b)].reshape(patch_shape)

    return patches


def patches_means(patches):
    num_features = patches.shape[0] * patches.shape[1]
    features = np.zeros(num_features)

    k = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch_mean = np.mean(patches[i, j])
            features[k] = patch_mean
            k += 1

    return features


def plot_patches(patches, figsize=(10,10), wspace=0.1, hspace=0.1, border_width=2, patch_scores=None, save_to=None,
                 title=None
                 ):
    if patch_scores is not None and patches.shape[0]*patches.shape[1] != len(patch_scores):
        raise ValueError("Number of patch scores must be the same as number of patches")

    imin, imax = patches.min(), patches.max()

    fig, axes = plt.subplots(patches.shape[0], patches.shape[1],
                             figsize=figsize,
                             gridspec_kw={"wspace": wspace, "hspace": hspace}
                             )

    if patch_scores is not None:
        color_scaler = 1 - patch_scores / patch_scores.max()
    index = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            axes[i, j].imshow(patches[i, j], vmin=imin, vmax=imax, cmap="gray")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            axes[i, j].spines['bottom'].set_linewidth(border_width)
            axes[i, j].spines['top'].set_linewidth(border_width)
            axes[i, j].spines['right'].set_linewidth(border_width)
            axes[i, j].spines['left'].set_linewidth(border_width)


            if patch_scores is not None:
                #axes[i, j].text(patches.shape[2] // 2, patches.shape[3] // 2, patch_scores[index], color="red")
                ax_color = (1, color_scaler[index], color_scaler[index])
                axes[i, j].spines['bottom'].set_color(ax_color)
                axes[i, j].spines['top'].set_color(ax_color)
                axes[i, j].spines['right'].set_color(ax_color)
                axes[i, j].spines['left'].set_color(ax_color)

            axes[i, j].set_aspect("equal")

            index += 1

    if title is not None:
        #plt.title(title, fontsize=20)
        fig.suptitle(title, fontsize=20)
    if save_to is not None:
        plt.savefig(save_to)
    plt.show()


def bin_column(df, column, n_bins=10):
    values, bins = np.histogram(df[column].values, bins=n_bins)

    binned_dfs = []
    for i in range(len(bins) - 1):
        binmin, binmax = bins[i:i+2]
        binned_dfs.append(df[(df[column] > binmin) & (df[column] < binmax)])

    return binned_dfs, bins


class MoonContamination:
    def __init__(self):
        self._noise_file = "noise_and_offsets/noise_images.npz"
        self._offset_file = "noise_and_offsets/offsets.txt"
        self.offsets = pd.read_csv(self._offset_file).values
        self.noise_images = np.load(self._noise_file)["images"]

    def add_random_moon_noise(self, img, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if np.shape(img) != (512, 512):
            raise ValueError("Invalid image shape {}. Image shape must be (512, 512)".format(np.shape(img)))
        random_noise_file_index = np.random.randint(0, self.noise_images.shape[0])
        noise = self.noise_images[random_noise_file_index]
        return np.clip(img + noise, a_min=0, a_max=55000)

    def add_random_offset(self, img, seed=None):
        if seed is not None:
            np.random.seed(seed)
        rand_offset_idx = np.random.randint(0, self.offsets.shape[0])
        offset = self.offsets[rand_offset_idx]
        img = shift(img, (offset[1], offset[0]), mode="nearest")
        img = np.clip(img, a_min=0, a_max=55000)
        return img
