import numpy as np
import matplotlib.pyplot as plt
import os
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from moonlib.metrics import mean_average_error_percentage

np.random.seed(42)


def run_rf_experiment(df, X_cols, y_col, model_params, model_dir, log_dir, show=True):
    X, y = df[X_cols].values, df[y_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)


    model = RandomForestRegressor(**model_params)

    # training
    print("Training... ", end="", flush=True)
    model.fit(X_train, y_train)
    print("done")

    print("Saving model... ", end="", flush=True)
    model_file = os.path.join(model_dir, "rf.joblib")
    dump(model, model_file)
    print("done")

    # evaluating
    y_baseline = np.full_like(y_test, y_test.mean())
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mape_file = os.path.join(log_dir, "MAPE.txt")
    with open(mape_file, "w") as f:
        mape_baseline = mean_average_error_percentage(y_test, y_baseline)
        mape_train = mean_average_error_percentage(y_train, y_train_pred)
        mape_test = mean_average_error_percentage(y_test, y_test_pred)
        f.write("Baseline: {}\n".format(mape_baseline))
        f.write("Model:\n")
        f.write("\tTrain: {}\n".format(mape_train))
        f.write("\tTest: {}\n".format(mape_test))
        print("Baseline:", mape_baseline)
        print("Model:")
        print("\tTrain:", mape_train)
        print("\tTest:", mape_test)

    # feature importance
    feature_scores = model.feature_importances_

    # get scores for image features
    im_indices = [i for i, col in enumerate(X_cols) if col[:2] == "mu"]
    im_scores = feature_scores[im_indices]
    im_score_names = X_cols[im_indices]

    # get scores for info features
    info_indices = [i for i, col in enumerate(X_cols) if col[:2] != "mu"]
    info_scores = feature_scores[info_indices]
    info_score_names = X_cols[info_indices]

    # save scores
    np.savez(os.path.join(log_dir, "feature_importances.npz"),
             image_scores=im_scores,
             image_score_names=im_score_names,
             info_scores=info_scores,
             info_score_names=info_score_names
             )

    plt.bar(range(len(im_scores)), im_scores)
    plt.title("Image feature scores")
    plt.xlabel("Feature index")
    plt.ylabel("Score")
    plt.savefig(os.path.join(log_dir, "image_feature_importances.png"))
    if show:
        plt.show()

    plt.bar(range(len(info_scores)), info_scores)
    plt.title("Info feature scores")
    plt.xlabel("Feature index")
    plt.ylabel("Score")
    plt.savefig(os.path.join(log_dir, "info_feature_importances.png"))
    if show:
        plt.show()