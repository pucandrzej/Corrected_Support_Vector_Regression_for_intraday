from processes_generators.arx_processes import generate_ARX_multiexog
from tools.generate_mc_steps_seeds import generate_mc_steps_seeds
from prod_mc_config import prod_mc_config
from processes_predictors.predict import calc_interm_kernel
from wait_for_low_cpu_load import wait_for_low_cpu

from multiprocessing import Pool
import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.metrics import adjusted_rand_score

# Parameters
n_features = 60  # Number of random normal features
test_window = 30


# Function for kernel target alignment
def kernel_target_alignment(K, y):
    y_expanded = np.expand_dims(y, axis=1)
    numerator = (y_expanded.T @ K @ y_expanded)[0, 0]
    denominator = (
        np.sum(y**2) * np.linalg.norm(K, ord="fro")
    )  # TODO confirm that this was an error (sqrt on y) and fix everywhere else in case it was indeed
    return numerator / denominator


# generalized sf
def generalized_pf(coeffs, ranking):
    D = np.abs(coeffs[:, None] - coeffs)
    f = 0
    for i in range(len(coeffs)):
        w_0 = np.sum(
            coeffs[: i + 1] * D[i, : i + 1]
        )  # important: the shift by 1 to account for python indexing and make it <=
        w_1 = []
        for j in range(len(coeffs)):
            if ranking[j] <= ranking[i]:
                w_1.append(coeffs[j] * D[i, j])
        f += coeffs[i] * np.abs(w_0 - np.sum(w_1))
    return f


def mc_step(params):
    corr_config, alpha, sample_len, step_seed, idx = params

    # Parameters for the multidimensional distributions
    mean_1 = np.zeros(int(n_features / 3))  # Mean vector for the first distribution
    mean_2 = np.zeros(int(n_features / 3))  # Mean vector for the second distribution
    mean_3 = np.zeros(int(n_features / 3))  # Mean vector for the third distribution

    cov_1 = (
        np.eye(int(n_features / 3)) * 1
    )  # Covariance matrix for the first distribution, 3 is the variance TODO vary the correlation coeffs
    cov_1[cov_1 == 0] = corr_config[0]
    cov_2 = (
        np.eye(int(n_features / 3)) * 1
    )  # Covariance matrix for the second distribution
    cov_2[cov_2 == 0] = corr_config[1]
    cov_3 = (
        np.eye(int(n_features / 3)) * 1
    )  # Covariance matrix for the third distribution
    cov_3[cov_3 == 0] = corr_config[2]

    # Generate multidimensional samples
    X_features = [
        np.random.multivariate_normal(mean_1, cov_1, sample_len).T,
        np.random.multivariate_normal(mean_2, cov_2, sample_len).T,
        np.random.multivariate_normal(mean_3, cov_3, sample_len).T,
    ]

    X_labels = [
        0 * np.ones(int(n_features / 3)),
        np.ones(int(n_features / 3)),
        2 * np.ones(int(n_features / 3)),
    ]

    # Flatten the list to have all features in a single list
    X_features = [x for group in X_features for x in group]
    X_labels = [x for group in X_labels for x in group]
    # Generate ARX series
    rng = np.random.default_rng(seed=step_seed)
    exog_coeff = rng.uniform(-1, 1, n_features)
    Y = generate_ARX_multiexog(
        [alpha], exog_coeff, 1, X_features, sample_len, step_seed
    )

    # Standardize exogenous variables and Y
    exog_standardised_vars = [(var - np.mean(var)) / np.std(var) for var in X_features]
    Y = np.exp((Y - np.mean(Y)) / np.std(Y))
    Y = (Y - np.mean(Y)) / np.std(Y)

    # Prepare training and test matrix
    X = np.vstack((Y[:-1], *[exog_var[1:] for exog_var in exog_standardised_vars])).T
    Y_train = Y[1:]

    X_calib = X[:-test_window, :]
    Y_calib = Y_train[:-test_window]

    # Compute kernel alignments and correlations for each single variable on the calibration window
    kernel_alignments = []
    corrs = []
    for col in X_calib.T:
        grid_alignments = []
        grid_corrs = []

        for q_kernel in prod_mc_config["param_grid_slide_on_median"]["q_kernel"]:
            for q_data in prod_mc_config["param_grid_slide_on_median"]["q_data"]:
                plain_kernel_X = np.abs(col[:, None] - col)
                width = np.log(2 - 2 * q_kernel) / np.quantile(plain_kernel_X, q_data)
                training_matrix = np.exp(width * plain_kernel_X)
                kernel_alignment = kernel_target_alignment(training_matrix, Y_calib)
                grid_alignments.append(kernel_alignment)
                grid_corrs.append(np.corrcoef(col, Y_calib)[0, 1])

        kernel_alignments.append(max(grid_alignments))
        corrs.append(max(grid_corrs))

    # cluster data and then compute the kernel alignment for each cluster on the calibration window
    clustering = AgglomerativeClustering(n_clusters=3).fit(X_calib[:, 1:].T)

    # get the clustering accuracy (1 means that the clustering is perfect)
    classification_rand_score = adjusted_rand_score(X_labels, clustering.labels_)

    cluster_kernel_aligns = []
    cluster_argmaxes = []
    cluster_mean_coeffs = []
    clusters_replaced_by_one_variable = []
    clusters_replaced_by_avg_variable = []
    two_clusters_replaced_by_vars = []
    two_clusters_replaced_by_avgs = []
    all_clusters_replaced_by_vars = []
    all_clusters_replaced_by_avgs = []
    for label in np.unique(clustering.labels_):
        # first calc the align for the whole cluster
        X_cluster = X[:, 1:][:, clustering.labels_ == label]
        X_cluster_calib = X_calib[:, 1:][:, clustering.labels_ == label]
        cluster_mean_coeffs.append(
            np.mean(np.abs(exog_coeff[clustering.labels_ == label]))
        )
        grid_cluster_kernel_aligns = []
        for q_kernel in prod_mc_config["param_grid_slide_on_median"]["q_kernel"]:
            for q_data in prod_mc_config["param_grid_slide_on_median"]["q_data"]:
                plain_kernel_X = calc_interm_kernel(X_cluster_calib)
                width = np.log(2 - 2 * q_kernel) / np.quantile(plain_kernel_X, q_data)
                training_matrix = np.exp(width * plain_kernel_X)
                kernel_alignment = kernel_target_alignment(training_matrix, Y_calib)
                grid_cluster_kernel_aligns.append(kernel_alignment)
        cluster_kernel_aligns.append(max(grid_cluster_kernel_aligns))

        # then calculate align for each variable in the cluster, keeping only the argmax one so we keep it as info from this cluster
        inter_kernel_alignments = []
        for col in X_cluster_calib.T:
            grid_alignments = []

            for q_kernel in prod_mc_config["param_grid_slide_on_median"]["q_kernel"]:
                for q_data in prod_mc_config["param_grid_slide_on_median"]["q_data"]:
                    plain_kernel_X = np.abs(col[:, None] - col)
                    width = np.log(2 - 2 * q_kernel) / np.quantile(
                        plain_kernel_X, q_data
                    )
                    training_matrix = np.exp(width * plain_kernel_X)
                    kernel_alignment = kernel_target_alignment(training_matrix, Y_calib)
                    grid_alignments.append(kernel_alignment)

            inter_kernel_alignments.append(max(grid_alignments))
        cluster_argmaxes.append(np.argmax(inter_kernel_alignments))

        clusters_replaced_by_one_variable.append(
            np.hstack(
                [
                    X[:, 1:][:, clustering.labels_ != label],
                    X_cluster[:, np.argmax(inter_kernel_alignments), None],
                ]
            )
        )
        clusters_replaced_by_avg_variable.append(
            np.hstack(
                [
                    X[:, 1:][:, clustering.labels_ != label],
                    np.mean(X_cluster, axis=1)[:, None],
                ]
            )
        )
        all_clusters_replaced_by_vars.append(
            X_cluster[:, np.argmax(inter_kernel_alignments)]
        )
        all_clusters_replaced_by_avgs.append(np.mean(X_cluster, axis=1))

    all_clusters_replaced_by_vars = np.vstack(all_clusters_replaced_by_vars).T
    # prepare replacements of 2 clusters by variables
    for label_idx, label in enumerate(np.unique(clustering.labels_)):
        two_clusters_replaced_by_vars.append(
            np.hstack(
                [
                    X[:, 1:][:, clustering.labels_ == label],
                    np.delete(all_clusters_replaced_by_vars, label_idx, axis=1),
                ]
            )
        )

    all_clusters_replaced_by_avgs = np.vstack(all_clusters_replaced_by_avgs).T
    # prepare replacements of 2 clusters by variables
    for label_idx, label in enumerate(np.unique(clustering.labels_)):
        two_clusters_replaced_by_avgs.append(
            np.hstack(
                [
                    X[:, 1:][:, clustering.labels_ == label],
                    np.delete(all_clusters_replaced_by_avgs, label_idx, axis=1),
                ]
            )
        )

    subsets_maes = []
    subsets_aligns = []
    subsets_max_aligns = []

    for subset_idx, variables_subset in enumerate(
        [X[:, 1:]]
        + clusters_replaced_by_one_variable.copy()
        + two_clusters_replaced_by_vars.copy()
        + [all_clusters_replaced_by_vars.copy()]
        + clusters_replaced_by_avg_variable.copy()
        + two_clusters_replaced_by_avgs.copy()
        + [all_clusters_replaced_by_avgs.copy()]
    ):
        variables_subset = np.hstack(
            [variables_subset, X[:, 0, None]]
        )  # add the AR part back
        tscv = TimeSeriesSplit(n_splits=test_window, test_size=1)

        aligns = []
        scores = []
        for q_kernel in prod_mc_config["param_grid_slide_on_median"]["q_kernel"]:
            for q_data in prod_mc_config["param_grid_slide_on_median"]["q_data"]:
                for C in prod_mc_config["param_grid_slide_on_median"]["C"]:
                    for epsilon in prod_mc_config["param_grid_slide_on_median"][
                        "epsilon"
                    ]:
                        intermediate_scores = []
                        intermediate_aligns = []
                        for train_idx, test_idx in tscv.split(variables_subset):
                            X_train_cv, X_test_cv = (
                                variables_subset[train_idx],
                                variables_subset[test_idx],
                            )
                            y_train_cv, y_test_cv = (
                                Y_train[train_idx],
                                Y_train[test_idx],
                            )

                            plain_kernel_X = calc_interm_kernel(X_train_cv)
                            width = np.log(2 - 2 * q_kernel) / np.quantile(
                                plain_kernel_X, q_data
                            )
                            training_matrix = np.exp(-np.abs(width) * plain_kernel_X)
                            reg = SVR(kernel="precomputed", epsilon=epsilon, C=C)
                            kernel_alignment = kernel_target_alignment(
                                training_matrix, y_train_cv
                            )
                            reg.fit(training_matrix, y_train_cv)

                            plain_kernel_X = np.sqrt(
                                np.sum((X_train_cv - X_test_cv) ** 2, axis=1)
                            )
                            width = np.log(2 - 2 * q_kernel) / np.quantile(
                                plain_kernel_X, q_data
                            )
                            test_matrix = np.exp(-np.abs(width) * plain_kernel_X)

                            forecast = reg.predict(test_matrix.reshape(1, -1))[0]

                            intermediate_aligns.append(kernel_alignment)
                            intermediate_scores.append(np.abs(forecast - y_test_cv[0]))
                        aligns.append(np.mean(intermediate_aligns))
                        scores.append(np.mean(intermediate_scores))

        subsets_aligns.append(aligns[np.argmin(scores)])
        subsets_max_aligns.append(max(aligns))

        subsets_maes.append(np.min(scores))

    # Return the results
    cor_matrix = np.vstack([corrs, np.abs(np.array([alpha] + list(exog_coeff)))]).T
    cor_matrix = cor_matrix[cor_matrix[:, 0].argsort()]  # sort by corr
    cor_matrix[:, 0] = range(1, len(cor_matrix) + 1)
    cor_matrix = cor_matrix[cor_matrix[:, 1].argsort()]  # sort by coeffs
    corr_generalized_spearmans_footrol = generalized_pf(
        cor_matrix[:, 1], cor_matrix[:, 0]
    )

    align_matrix = np.vstack(
        [kernel_alignments, np.abs(np.array([alpha] + list(exog_coeff)))]
    ).T
    align_matrix = align_matrix[align_matrix[:, 0].argsort()]  # sort by align
    align_matrix[:, 0] = range(1, len(align_matrix) + 1)
    align_matrix = align_matrix[align_matrix[:, 1].argsort()]  # sort by coeffs
    align_generalized_spearmans_footrol = generalized_pf(
        align_matrix[:, 1], align_matrix[:, 0]
    )

    dropping_results = pd.DataFrame()
    dropping_results["Meta"] = [
        "_",
        "0",
        "1",
        "2",
        "12",
        "02",
        "01",
        "012",
        "0 avg",
        "1 avg",
        "2 avg",
        "12 avg",
        "02 avg",
        "01 avg",
        "012 avg",
    ]
    dropping_results["Post_Drop_Alignments"] = subsets_aligns
    dropping_results["Max_Post_Drop_Alignments"] = subsets_max_aligns
    dropping_results["Post_Drop_MAE"] = subsets_maes
    dropping_results["Mean_abs_coeff"] = [None] + cluster_mean_coeffs + 11 * [None]

    return {
        "corr_generalized_spearmans_footrol": corr_generalized_spearmans_footrol,
        "align_generalized_spearmans_footrol": align_generalized_spearmans_footrol,
        "dropping_results": dropping_results,
        "classification_rand_score": classification_rand_score,
        "cluster_alignments": cluster_kernel_aligns,
        "labels": clustering.labels_,
    }


if __name__ == "__main__":
    print("Waiting till the cpu load is small...")
    # wait_for_low_cpu()
    print("Running the simulation.")

    start = time.time()

    mc_config = prod_mc_config

    mc_steps = mc_config["mc_steps_features_sens"]
    seeds = generate_mc_steps_seeds(mc_steps)

    sample_lengths = [101]

    correlations = [
        # [0.9, 0.9, 0.9], # all 3 groups highly correlated
        # [0.1, 0.1, 0.1], # all 3 groups slightly correlated
        [0, 0, 0],  # all 3 groups not correlated
        [0.9, 0.1, 0.1],  # 1 group highly correlated
        [0.9, 0.9, 0.1],  # 2 groups highly correlated
    ]

    for corr_config in correlations:
        print(corr_config)
        for alpha in [0.1, 0.9]:
            if corr_config[0] == 0 and alpha == 0.1:
                continue
            print(alpha)
            for sample_length in sample_lengths:
                params = [
                    (corr_config, alpha, sample_length, seeds[idx], idx)
                    for idx in range(mc_steps)
                ]
                with Pool(processes=min(30, mc_steps)) as p:
                    results = p.map(mc_step, params.copy())
                dir_name = f"{corr_config}_{alpha}_{sample_length}_features_sensitivity_analysis"
                os.makedirs(os.path.join("MC_RESULTS", dir_name), exist_ok=True)
                current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                pickle.dump(
                    results,
                    open(
                        os.path.join(
                            "MC_RESULTS", dir_name, current_datetime + "_results.pickle"
                        ),
                        "wb",
                    ),
                )
