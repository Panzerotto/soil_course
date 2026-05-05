import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def theta_from_Se(Se, theta_r, theta_s):
    Se = np.clip(Se, 0.0, 1.0)
    return theta_r + (theta_s - theta_r) * Se

def Se_brooks_corey(psi_kpa, psi_b_kpa=10.0, lam=0.6):
    psi_kpa = np.asarray(psi_kpa, dtype=float)
    Se = np.ones_like(psi_kpa)
    mask = psi_kpa > psi_b_kpa
    Se[mask] = (psi_kpa[mask] / psi_b_kpa) ** (-lam)
    return np.clip(Se, 0.0, 1.0)

def plot_swrc_comparison():
    psi = np.logspace(-2, 4, 800)
    theta_r = 0.05
    theta_s = 0.45
    Se = Se_brooks_corey(psi)
    theta = theta_from_Se(Se, theta_r, theta_s)

    plt.figure(figsize=(10, 4))
    plt.plot(psi, theta)
    plt.xscale("log")
    plt.xlabel("Suction |ψ| (kPa)")
    plt.ylabel("Volumetric water content, θ")
    plt.title("Brooks–Corey SWRC")
    plt.grid(True)
    plt.show()
