from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.ndimage import shift
import glob
import os
from .utils import load_fits, img_to_patches, patches_means, MoonContamination
np.random.seed(42)

def extract_features_from_fits(path, save_path, patch_shape, with_noise=False, with_offset=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = glob.glob(os.path.join(path, "*.fits"))
    np.random.shuffle(files)

    # infer number of features by processing first file
    img, (info, info_headers) = load_fits(files[0])
    patches = img_to_patches(img, patch_shape)
    features = patches_means(patches)

    # define data matrix from inferred number of features
    data_mat = np.zeros((len(files), len(features)+len(info)))
    data_mat[0] = np.concatenate([features, info])

    # process the remaining files
    header = ["mu" + str(i) for i in range(len(features))]
    header.extend(info_headers)

    contaminator = MoonContamination()
    suffix = "{}_{}x{}".format(os.path.basename(path), patch_shape[0], patch_shape[1])
    if with_noise:
        suffix += "_n"
    if with_offset:
        suffix += "_o"
    suffix += ".csv"

    save_file = os.path.join(save_path,
                             suffix)
    for i in tqdm(range(1, len(files))):
        # load image and info
        img, (info, info_headers) = load_fits(files[i])

        # add noise to image
        if with_noise:
            contaminator.add_random_moon_noise(img)

        # add offset to image
        if with_offset:
            img = contaminator.add_random_offset(img)

        # split image into patches
        patches = img_to_patches(img, patch_shape)

        # extract features
        features = patches_means(patches)
        data_mat[i] = np.concatenate([features, info])

        if i%500 == 0:
            df = pd.DataFrame(data_mat, columns=header)
            df.to_csv(save_file, index=False)

    df = pd.DataFrame(data_mat, columns=header)
    df.to_csv(save_file, index=False)

