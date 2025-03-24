import numpy as np


def generate_AR(alpha, beta, sample_len, seed, shock=True):
    Y = [0]

    rng = np.random.default_rng(
        seed=seed
    )  # important: works only by setting it before every generation
    random_vector = rng.normal(0, 1, sample_len)
    rng = np.random.default_rng(seed=seed)
    random_vector_uniform = rng.uniform(size=sample_len)
    # rng = np.random.default_rng(seed=seed)
    # coin_toss = np.random.uniform(0, 1, sample_len)

    # for i in range(1, sample_len):
    #     if coin_toss[i] > 0.5:
    #         Y.append(alpha*Y[i - 1] + alpha*np.sqrt(np.abs(Y[i - 1])) + beta + random_vector[i])
    #     else:
    #         Y.append(Y[i - 1])

    # for i in range(1, sample_len):
    #     Y.append(alpha*Y[i - 1] + np.sin(i/(sample_len/20)) + beta + random_vector[i])

    # for i in range(1, sample_len):
    #     Y.append(alpha*Y[i - 1] + np.sin(i/(sample_len/20)) + beta + random_vector[i])

    # put_spikes = np.random.randint(0, sample_len, sample_len//20)
    # spikes = []
    # for i in range(sample_len//20):
    #     if np.random.uniform() < 0.5:
    #         spikes.append(6)
    #     else:
    #         spikes.append(-10)
    # Y = np.exp((Y - np.mean(Y))/np.std(Y))
    # Y[put_spikes] = Y[put_spikes] + np.array(spikes)

    # return Y
    shock_risk = 0.05
    shock_val = 3
    shocks = [0]
    for i in range(1, sample_len):
        ar_shock = 0
        if random_vector_uniform[i] < shock_risk / 2:
            ar_shock = shock_val
        elif random_vector_uniform[i] >= 1 - shock_risk / 2:
            ar_shock = -shock_val
        Y.append(alpha * Y[i - 1] + beta + random_vector[i])
        shocks.append(ar_shock)

    if shock:
        return np.array(Y) + np.array(shocks)
    else:
        return np.array(Y)


def generate_ARX(alpha, beta, X, sample_len, seed, shock=True):
    Y = [0]

    rng = np.random.default_rng(
        seed=seed
    )  # important: works only by setting it before every generation
    random_vector = rng.normal(0, 1, sample_len)
    rng = np.random.default_rng(seed=seed)
    random_vector_uniform = rng.uniform(size=sample_len)

    # return Y
    shock_risk = 0.05
    shock_val = 3
    shocks = [0]
    for i in range(1, sample_len):
        ar_shock = 0
        if random_vector_uniform[i] < shock_risk / 2:
            ar_shock = shock_val
        elif random_vector_uniform[i] >= 1 - shock_risk / 2:
            ar_shock = -shock_val
        base_value = alpha * Y[i - 1] + beta + random_vector[i]
        for v in X:
            base_value += v[i]
        Y.append(base_value)
        shocks.append(ar_shock)

    if shock:
        return np.array(Y) + np.array(shocks)
    else:
        return np.array(Y)


def generate_ARX_multiexog(alpha, exog_coeff, beta, X, sample_len, seed, shock=False):
    Y = [0]  # Initialize the AR component
    rng = np.random.default_rng(seed=seed)
    random_vector = rng.normal(0, 1, sample_len)  # Random noise
    rng = np.random.default_rng(seed=seed)
    random_vector_uniform = rng.uniform(size=sample_len)
    shock_risk = 0.05
    shock_val = 3
    shocks = [0]

    for i in range(1, sample_len):
        ar_shock = 0
        if random_vector_uniform[i] < shock_risk / 2:
            ar_shock = shock_val
        elif random_vector_uniform[i] >= 1 - shock_risk / 2:
            ar_shock = -shock_val

        base_value = alpha[0] * Y[i - 1] + beta + random_vector[i]
        base_value += sum(exog_coeff[idx] * X[idx][i] for idx in range(len(X)))
        Y.append(base_value)
        shocks.append(ar_shock)

    if shock:
        return np.array(Y) + np.array(shocks)
    else:
        return np.array(Y)
