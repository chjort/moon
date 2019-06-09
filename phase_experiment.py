import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import os
from moonlib.experiment import run_rf_experiment
from moonlib import utils
import shutil


#%
files = glob.glob("/media/ch/Seagate Expansion Drive/MOON/processed_mean_4x4-128x128/*")
for file in files[:2]:
    print(file)
    data_path = file
    patch_size_s = os.path.basename(data_path).split("_")[1].split("x")
    patch_size = (int(patch_size_s[0]), int(patch_size_s[1]))

    df = utils.load_folder(data_path)
    im_indices = [i for i, col in enumerate(df.columns) if col[:2] == "mu"]
    X_cols, y_col = np.array([*df.columns[im_indices]]), "ALBEDO"

    n_bins = 10
    bdf, bins = utils.bin_column(df, "PHASE", n_bins=n_bins)

    # create experiment directory
    timestamp = utils.get_timestamp()
    dataset_name = os.path.basename(data_path).split(".")[0]
    exp_dir = os.path.join("results", *["rf", dataset_name, "binned_{}".format(timestamp)])
    overlay_dir = os.path.join(exp_dir, "score_overlays")
    feature_dir = os.path.join(exp_dir, "feature_importances")
    utils.make_dir(exp_dir)
    utils.make_dir(overlay_dir)
    utils.make_dir(feature_dir)

    #% Plot Albedo across bins

    vals = np.zeros((n_bins, 20))
    for i in range(n_bins):
        val, bin = np.histogram(bdf[i]["ALBEDO"], bins=20)
        vals[i] = val

    mean_hist = vals.mean(axis=0)
    var_hist = np.sum((vals - mean_hist)**2, axis=0)
    std_hist = np.sqrt(var_hist)

    plt.bar(range(20), mean_hist, label="Mean histogram")
    plt.bar(range(20), std_hist, label="Histogram standard deviation")
    plt.title("Mean albedo histogram across bins")
    plt.xlabel("Albedo")
    plt.ylabel("Count")
    plt.xticks(np.arange(len(bin)) - 0.5,
               np.linspace(0.1, 0.5, 21).round(2),
               rotation=45
               )
    plt.legend()
    plt.savefig(os.path.join(exp_dir, "albedo_across_bins_fig.png"))
    #plt.show()


    #%% Plot the bins
    vals, bins = np.histogram(df["PHASE"].values, bins=n_bins)

    plt.bar(range(n_bins), vals)
    plt.xlabel("Phase [degrees]")
    plt.ylabel("Count")
    plt.title("Phase bins")
    plt.xticks(np.arange(len(bins)) - 0.5, bins.round(0))
    plt.savefig(os.path.join(exp_dir, "phase_bins_count_fig.png"))
    #plt.show()

    #%% Find an image from each bin.
    # files = glob.glob("/home/ch/workspace/moon/raw_tar/shard*/*.fits")
    #
    # binned_imgs = {}
    # found_bins = set()
    # for file in files:
    #     img, (info, info_header) = utils.load_fits(file)
    #
    #     for j in range(len(bins) - 1):
    #         binmin, binmax = bins[j:j+2]
    #         if info[7] < binmax and info[7] > binmin and (binmin, binmax) not in found_bins:
    #             binned_imgs[j] = {"img_path": file, "min": binmin, "max": binmax}
    #             found_bins.add((binmin, binmax))
    #             print(binmin, binmax, len(found_bins))
    #
    #     if len(found_bins) == n_bins:
    #         break

    img_files = sorted(glob.glob("/home/ch/Dropbox/DTU/Research/Revealing Climate Change from Moon Images/plot_images/phases/*"))
    binned_imgs = {i:{"img_path":ifile} for i, ifile in enumerate(img_files)}

    #%% Train separate models on each bin

    model_params = {"n_estimators": 250,
                    "n_jobs": 7,
                    "verbose": 2
                    }
    contaminator = utils.MoonContamination()
    for bin_i in range(n_bins):
        print("**** {} ****".format(bin_i))
        df = bdf[bin_i]

        # create experiment directories
        result_dir = os.path.join(exp_dir, "bin{}".format(bin_i))
        model_dir = os.path.join(result_dir, "model")
        log_dir = os.path.join(result_dir, "logs")
        utils.make_dir(model_dir)
        utils.make_dir(log_dir)

        config = {**model_params,
                  "features": X_cols.tolist(),
                  "target": y_col,
                  "bin": [bins[bin_i].round(3), bins[bin_i + 1].round(3)]
                  }
        config_file = os.path.join(log_dir, "config.json")
        with open(config_file, "w") as f:
            f.write(json.dumps(config, sort_keys=True, indent=4))

        # run experiment
        run_rf_experiment(df=df,
                          X_cols=X_cols,
                          y_col=y_col,
                          model_params=model_params,
                          model_dir=model_dir,
                          log_dir=log_dir,
                          show=False
                          )

        # visualize feature importance
        feat_file = os.path.join(log_dir, "feature_importances.npz")
        phase_img_file = binned_imgs[bin_i]["img_path"]
        img = utils.load_fits(phase_img_file)[0]
        if "_n" in data_path:
            img = contaminator.add_random_moon_noise(img)
        if "_o" in data_path:
            img = contaminator.add_random_offset(img)

        feat = np.load(feat_file)["image_scores"]
        patches = utils.img_to_patches(np.log(img + 1), patch_size)
        utils.plot_patches(patches, wspace=0.1, hspace=0.1, border_width=2, patch_scores=feat,
                           save_to=os.path.join(log_dir, "score_overlay.png"),
                           title="Phase: ({}, {})".format(bins[bin_i].round(3), bins[bin_i + 1].round(3)), show=False)
        shutil.copyfile(os.path.join(log_dir, "score_overlay.png"), os.path.join(overlay_dir, "bin{}.png".format(bin_i)))
        shutil.copyfile(os.path.join(log_dir, "image_feature_importances.png"), os.path.join(feature_dir, "bin{}.png".format(bin_i)))