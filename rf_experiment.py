import numpy as np
import os
import glob
import json
from moonlib import utils
from moonlib.experiment import run_rf_experiment

np.random.seed(42)


#% load data
files = glob.glob("/media/ch/Seagate Expansion Drive/MOON/processed_mean_4x4-128x128/*")
for file in files:
    print(file)
    data_path = file
    patch_size_s = os.path.basename(data_path).split("_")[1].split("x")
    patch_size = (int(patch_size_s[0]), int(patch_size_s[1]))
    df = utils.load_folder(data_path)

    #
    im_indices = [i for i, col in enumerate(df.columns) if col[:2] == "mu"]
    X_cols, y_col = np.array([*df.columns[im_indices]]), "ALBEDO"

    model_params = {"n_estimators": 250,
                    "n_jobs": 7,
                    "verbose": 2
                    }

    model_dir, log_dir = utils.make_results_dir(data_path)

    config = {**model_params,
              "features": X_cols.tolist(),
              "target": y_col,
              }
    config_file = os.path.join(log_dir, "config.json")
    with open(config_file, "w") as f:
        f.write(json.dumps(config, sort_keys=True, indent=4))

    run_rf_experiment(df=df,
                      X_cols=X_cols,
                      y_col=y_col,
                      model_params=model_params,
                      model_dir=model_dir,
                      log_dir=log_dir,
                      show=False
                      )



    feat_file = os.path.join(log_dir, "feature_importances.npz")
    feat = np.load(feat_file)["image_scores"]

    img, info = utils.load_fits('../plot_images/single_phase/2455890.1730884_1.64449716_0.05676983_0.21432364.fits')
    patches = utils.img_to_patches(np.log(img), patch_size)

    utils.plot_patches(patches, wspace=0.1, hspace=0.1, border_width=2, patch_scores=feat,
                           save_to=os.path.join(log_dir, "score_overlay.png"),
                           title="Feature importance overlay", show=False)