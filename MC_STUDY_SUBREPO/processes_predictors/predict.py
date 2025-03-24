import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import scipy


def kernel_target_alignment(K, y):
    return (np.expand_dims(y, axis=(1)).T @ K @ np.expand_dims(y, axis=(1)))[0][0] / (
        np.sqrt(np.sum(y**2)) * np.linalg.norm(K, ord="fro")
    )


def calc_interm_kernel(interm_data, norm: int = 2):
    plain_kernel = np.zeros((len(interm_data), len(interm_data)))
    for i in range(len(interm_data)):
        for j in range(len(interm_data)):
            if i < j:
                if norm == 2:
                    plain_kernel[i, j] = np.sqrt(
                        np.sum((interm_data[i] - interm_data[j]) ** 2)
                    )  # L2 distance
                elif norm == 1:
                    plain_kernel[i, j] = np.sum(
                        np.abs(interm_data[i] - interm_data[j])
                    )  # L1 distance
    plain_kernel = plain_kernel + plain_kernel.T
    return plain_kernel


def predict(
    model,
    X_train,
    Y_train,
    X_test,
    q_kernel=None,
    q_data=None,
    naive_forecasts=None,
    naive_forecast=None,
    q_kernel_naive=None,
    q_data_naive=None,
    q_kernel_div=None,
    q_data_div=None,
    div_impact=None,
    epsilon=None,
    C=None,
    insample_mae=True,
):
    # dependence of laplace quantile func on quantile arg
    if q_kernel is not None:
        if q_kernel >= 0.5:
            laplace_quantile = np.log(2 - 2 * q_kernel)
        else:
            laplace_quantile = np.log(2 * q_kernel)

    if q_kernel_div is not None:
        if q_kernel_div >= 0.5:
            laplace_quantile_div = np.log(2 - 2 * q_kernel_div)
        else:
            laplace_quantile_div = np.log(2 * q_kernel_div)

    if not insample_mae:
        insample_fit = Y_train

    if model in [
        "AR",
        "AR_exp",
        "ARX",
        "ARX_exp",
        "AR_noshock",
        "AR_exp_noshock",
        "ARX_noshock",
        "ARX_exp_noshock",
    ]:
        reg = LinearRegression(fit_intercept=False).fit(X_train, Y_train)
        insample_fit = reg.predict(X_train)
        forecast = reg.predict(X_test)[0]
        kernel_alignment = None

    elif model == "SVR":
        plain_kernel_X = calc_interm_kernel(X_train)
        width = laplace_quantile / np.quantile(plain_kernel_X, q_data)
        training_matrix = np.exp(-np.abs(width) * plain_kernel_X)
        reg = SVR(kernel="precomputed", epsilon=epsilon, C=C)
        kernel_alignment = kernel_target_alignment(training_matrix, Y_train)
        reg.fit(training_matrix, Y_train)

        plain_kernel_X = np.sqrt(np.sum((X_train - X_test) ** 2, axis=1))
        width = laplace_quantile / np.quantile(plain_kernel_X, q_data)
        test_matrix = np.exp(-np.abs(width) * plain_kernel_X)

        forecast = reg.predict(test_matrix.reshape(1, -1))[0]
        if insample_mae:
            insample_fit = reg.predict(training_matrix)

    elif model == "cSVR":
        plain_kernel_X = calc_interm_kernel(X_train)
        plain_kernel_X_naive = (
            np.abs(
                np.tile(naive_forecasts, [len(naive_forecasts), 1])
                - np.swapaxes(np.tile(naive_forecasts, [len(naive_forecasts), 1]), 0, 1)
            )
            ** 2
        )
        sigma = np.quantile(plain_kernel_X_naive, q_data_naive) / scipy.stats.norm.ppf(
            q_kernel_naive, loc=0, scale=1
        )
        width = laplace_quantile / np.quantile(plain_kernel_X, q_data)
        training_matrix = np.exp(
            -np.abs(width) * plain_kernel_X - 1 / (2 * sigma**2) * plain_kernel_X_naive
        )
        reg = SVR(kernel="precomputed", epsilon=epsilon, C=C)
        kernel_alignment = kernel_target_alignment(training_matrix, Y_train)
        reg.fit(training_matrix, Y_train)

        plain_kernel_X = np.sqrt(np.sum((X_train - X_test) ** 2, axis=1))
        plain_kernel_X_naive = np.abs(naive_forecasts - naive_forecast) ** 2
        sigma = np.quantile(plain_kernel_X_naive, q_data_naive) / scipy.stats.norm.ppf(
            q_kernel_naive, loc=0, scale=1
        )
        width = laplace_quantile / np.quantile(plain_kernel_X, q_data)
        test_matrix = np.exp(
            -np.abs(width) * plain_kernel_X - 1 / (2 * sigma**2) * plain_kernel_X_naive
        )

        forecast = reg.predict(test_matrix.reshape(1, -1))[0]
        if insample_mae:
            insample_fit = reg.predict(training_matrix)

    elif model == "ccSVR":
        plain_kernel_X = calc_interm_kernel(X_train)
        width = laplace_quantile / np.quantile(plain_kernel_X, q_data)
        plain_kernel_X_naive = (
            np.abs(
                np.tile(naive_forecasts, [len(naive_forecasts), 1])
                - np.swapaxes(np.tile(naive_forecasts, [len(naive_forecasts), 1]), 0, 1)
            )
            ** 2
        )
        sigma = np.quantile(plain_kernel_X_naive, q_data_naive) / scipy.stats.norm.ppf(
            q_kernel_naive, loc=0, scale=1
        )
        plain_kernel_X_naive_div = np.abs(
            np.tile(naive_forecasts, [len(naive_forecasts), 1])
            - np.swapaxes(np.tile(naive_forecasts, [len(naive_forecasts), 1]), 0, 1)
        )
        plain_kernel_X_naive_div_fore = np.abs(naive_forecasts - naive_forecast)
        while (
            np.quantile(plain_kernel_X_naive_div, q_data_div) == 0
            or np.quantile(plain_kernel_X_naive_div_fore, q_data_div) == 0
        ):
            q_data_div += 0.01
        width_div = laplace_quantile_div / np.quantile(
            plain_kernel_X_naive_div, q_data_div
        )
        intermediate_div_kernel = np.exp(-np.abs(width_div) * plain_kernel_X_naive_div)
        div_kernel = 1 - div_impact * intermediate_div_kernel
        nominator = np.exp(
            -np.abs(width) * plain_kernel_X - 1 / (2 * sigma**2) * plain_kernel_X_naive
        )
        div_kernel[
            ~(
                (nominator != 1)
                & (intermediate_div_kernel < (1 / div_impact) * (1 - nominator))
            )
        ] = nominator[
            ~(
                (nominator != 1)
                & (intermediate_div_kernel < (1 / div_impact) * (1 - nominator))
            )
        ]
        training_matrix_cor_naive_div = nominator / div_kernel

        reg_corr_naive = SVR(kernel="precomputed", epsilon=epsilon, C=C)
        kernel_alignment = kernel_target_alignment(
            training_matrix_cor_naive_div, Y_train
        )
        reg_corr_naive.fit(training_matrix_cor_naive_div, Y_train)

        plain_kernel_X = np.sqrt(np.sum((X_train - X_test) ** 2, axis=1))
        width = laplace_quantile / np.quantile(plain_kernel_X, q_data)
        plain_kernel_X_naive = np.abs(naive_forecasts - naive_forecast) ** 2
        sigma = np.quantile(plain_kernel_X_naive, q_data_naive) / scipy.stats.norm.ppf(
            q_kernel_naive, loc=0, scale=1
        )
        plain_kernel_X_naive_div = np.abs(naive_forecasts - naive_forecast)
        width_div = laplace_quantile_div / np.quantile(
            plain_kernel_X_naive_div, q_data_div
        )
        intermediate_div_kernel = np.exp(-np.abs(width_div) * plain_kernel_X_naive_div)
        div_kernel = 1 - div_impact * intermediate_div_kernel
        nominator = np.exp(
            -np.abs(width) * plain_kernel_X - 1 / (2 * sigma**2) * plain_kernel_X_naive
        )
        div_kernel[
            ~(
                (nominator != 1)
                & (intermediate_div_kernel < (1 / div_impact) * (1 - nominator))
            )
        ] = nominator[
            ~(
                (nominator != 1)
                & (intermediate_div_kernel < (1 / div_impact) * (1 - nominator))
            )
        ]
        test_matrix_corr_naive = nominator / div_kernel

        forecast = reg_corr_naive.predict(test_matrix_corr_naive.reshape(1, -1))[0]
        if insample_mae:
            insample_fit = reg_corr_naive.predict(training_matrix_cor_naive_div)

    return (
        forecast,
        np.mean(np.abs(insample_fit - Y_train)),
        kernel_alignment,
        insample_fit,
    )
