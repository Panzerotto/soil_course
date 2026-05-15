import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


EULER_GAMMA = 0.5772156649015329
RAD6 = math.sqrt(6) / math.pi


def load_rainfall_extremes(filename):
    data = pd.read_csv(filename)
    data = data.set_index("anno")
    data.columns = data.columns.str.strip().str.replace(" ora", "h").str.replace(" ore", "h")
    data = data.apply(pd.to_numeric, errors="coerce")
    return data


def calculate_rainfall_statistics(data):
    means = data.mean()
    variances = data.var()
    stds = data.std()
    cv = stds / means

    return means, variances, stds, cv


def estimate_gumbel_parameters(means, stds):
    return pd.DataFrame(
        [means - RAD6 * EULER_GAMMA * stds, RAD6 * stds],
        index=["a", "b"]
    )


def gumbel_probability(x, parameters, duration):
    a = parameters[duration]["a"]
    b = parameters[duration]["b"]
    return np.exp(-np.exp(-(x - a) / b))


def create_gumbel_curves(data, parameters):
    durations = list(data.columns)

    x_values = np.linspace(
        data.min().min(),
        data.max().max(),
        100
    )

    curves = pd.DataFrame(index=x_values)

    for duration in durations:
        curves[duration] = gumbel_probability(
            x_values,
            parameters,
            duration
        )

    return curves


def plot_gumbel_curves(data, curves):
    fig, ax = plt.subplots(figsize=(11, 7))

    curves.plot(ax=ax, linewidth=3)

    for duration in data.columns:
        values = data[duration].dropna()
        ecdf = ECDF(values)

        ax.plot(
            values,
            ecdf(values),
            "o",
            label=f"{duration} observations",
            markersize=6,
            alpha=0.8
        )

    ax.set_title("Confronto tra curve di Gumbel e osservazioni")
    ax.set_xlabel("Precipitazione (mm)")
    ax.set_ylabel("Probabilità cumulata P(H < h)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()#     return data
# 
# 
# def save_clean_data(data, output_name='dati_puliti.csv'):
# 
#     data.to_csv(output_name, index=False)
# 
#     print(f"Saved file: {output_name}")
