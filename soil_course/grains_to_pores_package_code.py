import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv

# Physical constants (SI)
SIGMA = 0.0728
RHO_W = 1000.0
G = 9.81

# Grain-size boundaries in mm
CLAY_SILT = 0.002
SILT_SAND = 0.05
SAND_GRAVEL = 2.0


def percent_finer(d, d50=0.05, sigma=1.2):
    """Percentage of soil mass finer than diameter d (mm), using a lognormal curve."""
    d = np.asarray(d, dtype=float)
    return 100.0 * 0.5 * (
        1.0 + erf((np.log(d) - np.log(d50)) / (sigma * np.sqrt(2.0)))
    )


def diameter_at_percent(p, d50=0.05, sigma=1.2):
    """Diameter (mm) at which the percentage-finer curve reaches p percent."""
    return d50 * np.exp(sigma * np.sqrt(2.0) * erfinv(2.0 * p / 100.0 - 1.0))


def grain_mass_density(d, d50=0.05, sigma=1.2):
    """Unnormalised mass density per unit ln(d)."""
    d = np.asarray(d, dtype=float)
    z = (np.log(d) - np.log(d50)) / sigma
    return np.exp(-0.5 * z**2)


def print_grain_curve_check(d50=0.2, sigma=0.6):
    """Print D10, D50 and D90 for a grain-size curve."""
    print(f"grain curve, d50 = {d50} mm, sigma = {sigma}:")
    for p in [10, 50, 90]:
        print(f"  D{p:2d} = {diameter_at_percent(p, d50, sigma):.4f} mm")


def grains_to_pores(d50_mm, sigma_grain, beta=0.2, spread_factor=1.0):
    """Heuristic map from grain-size lognormal parameters to pore-size lognormal parameters."""
    d50_m = d50_mm * 1e-3
    r_pore_median_m = beta * d50_m
    r_median_um = r_pore_median_m * 1e6
    sigma_pore = sigma_grain * spread_factor
    return r_median_um, sigma_pore


def print_grains_to_pores_example(d50_mm=0.2, sigma_grain=0.6, beta=0.2, spread_factor=1.0):
    """Print the heuristic pore distribution derived from grain parameters."""
    r_med, sig_p = grains_to_pores(d50_mm, sigma_grain, beta, spread_factor)
    print(f"grain curve  d50 = {d50_mm:.2f} mm, sigma = {sigma_grain}")
    print("  -> heuristic pore distribution:")
    print(f"     median pore radius = {r_med:.2f} micrometres")
    print(f"     pore log-spread    = {sig_p:.2f}")


def young_laplace_head(r):
    """Suction head |psi| (m of water) that just empties a pore of radius r (m)."""
    return 2.0 * SIGMA / (RHO_W * G * np.asarray(r, float))


def pore_size_distribution(r_median=20.0, sigma_p=1.0, n_classes=400):
    """Lognormal pore-size distribution. r_median is in micrometres."""
    ln_med = np.log(r_median * 1e-6)
    ln_r = np.linspace(ln_med - 4 * sigma_p, ln_med + 4 * sigma_p, n_classes)
    r = np.exp(ln_r)
    dens = np.exp(-0.5 * ((ln_r - ln_med) / sigma_p) ** 2)
    return r, dens / dens.sum()


def retention_from_pores(r_median=20.0, sigma_p=1.0, theta_r=0.05, theta_s=0.45):
    """Construct the SWRC by the fill-smallest-first rule."""
    r, vol_frac = pore_size_distribution(r_median, sigma_p)
    order = np.argsort(r)
    r_asc = r[order]
    vf_asc = vol_frac[order]
    Se_asc = np.cumsum(vf_asc)
    psi_asc = young_laplace_head(r_asc)
    psi = psi_asc[::-1]
    Se = Se_asc[::-1]
    theta = theta_r + (theta_s - theta_r) * Se
    return psi, theta


def grains_to_retention(d50_mm=0.2, sigma_grain=0.6, beta=0.2,
                        spread_factor=1.0, theta_r=0.05, theta_s=0.45):
    """Full chain: grain-size curve -> pore distribution -> retention curve."""
    d = np.logspace(-4, 1, 300)
    Nfiner = percent_finer(d, d50_mm, sigma_grain)

    r_med_um, sigma_p = grains_to_pores(d50_mm, sigma_grain, beta, spread_factor)
    r_pore, vf = pore_size_distribution(r_med_um, sigma_p)
    psi, theta = retention_from_pores(r_med_um, sigma_p, theta_r, theta_s)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    bands = [
        (1e-4, CLAY_SILT, "#67c8e8", "clay"),
        (CLAY_SILT, SILT_SAND, "#7ad07a", "silt"),
        (SILT_SAND, SAND_GRAVEL, "#e89ad0", "sand"),
        (SAND_GRAVEL, 1e1, "#f2e85a", "gravel"),
    ]
    for x0, x1, col, lbl in bands:
        ax[0].axvspan(x0, x1, color=col, alpha=0.3)
        ax[0].text(np.sqrt(x0 * x1), 4, lbl, ha="center", fontsize=9)

    ax[0].semilogx(d, Nfiner, "k-", lw=2.4)
    ax[0].set_xlim(1e-4, 1e1)
    ax[0].set_ylim(0, 100)
    ax[0].set_xlabel("grain diameter  d  (mm)")
    ax[0].set_ylabel("percentage finer  (%)")
    ax[0].set_title(f"1. GRAIN-SIZE CURVE\n(d50={d50_mm} mm, sigma={sigma_grain})")
    ax[0].grid(True, which="both", color="0.92", lw=0.5)

    ax[1].semilogx(r_pore * 1e6, vf, "b-", lw=2.4)
    ax[1].fill_between(r_pore * 1e6, vf, color="tab:blue", alpha=0.25)
    ax[1].set_xlabel("pore radius  r  (micrometres)")
    ax[1].set_ylabel("pore-volume fraction")
    ax[1].set_title(
        f"2. PORE-SIZE DISTRIBUTION\n"
        f"(heuristic: r_med={r_med_um:.1f} um, sigma={sigma_p:.2f})"
    )
    ax[1].grid(True, which="both", color="0.92", lw=0.5)

    ax[2].semilogx(psi * 100.0, theta, "b-", lw=2.6)
    ax[2].axhline(theta_s, color="0.6", ls=":", lw=1)
    ax[2].axhline(theta_r, color="0.6", ls=":", lw=1)
    ax[2].set_xlabel("suction  |psi|  (cm of water)")
    ax[2].set_ylabel("water content  theta")
    ax[2].set_title("3. RETENTION CURVE\n(constructed, fill-smallest-first)")
    ax[2].grid(True, which="both", color="0.92", lw=0.5)

    fig.suptitle(
        f"From grains to pores to retention   "
        f"(pore ratio beta = {beta}, spread factor = {spread_factor})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()

    return r_med_um, sigma_p


def interactive_grains_to_retention():
    """Interactive sliders for the grain -> pore -> retention chain."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception:
        print("ipywidgets not installed - call grains_to_retention(...) directly.")
        return grains_to_retention(0.02, 1.4, beta=0.15)

    d50_sl = widgets.FloatLogSlider(
        value=0.2, base=10, min=-3, max=1, step=0.05,
        description="d50 (mm)", continuous_update=False
    )
    sig_sl = widgets.FloatSlider(
        value=0.6, min=0.2, max=2.2, step=0.05,
        description="sigma_grain", continuous_update=False
    )
    beta_sl = widgets.FloatSlider(
        value=0.2, min=0.02, max=0.6, step=0.01,
        description="beta", continuous_update=False
    )
    spr_sl = widgets.FloatSlider(
        value=1.0, min=0.4, max=2.0, step=0.05,
        description="spread", continuous_update=False
    )

    ui = widgets.interactive(
        lambda d50_mm, sigma_grain, beta, spread_factor:
            grains_to_retention(
                d50_mm=d50_mm,
                sigma_grain=sigma_grain,
                beta=beta,
                spread_factor=spread_factor,
            ),
        d50_mm=d50_sl,
        sigma_grain=sig_sl,
        beta=beta_sl,
        spread_factor=spr_sl,
    )
    display(ui)
    return ui


def van_genuchten(psi_cm, alpha, n, theta_r=0.05, theta_s=0.45):
    """van Genuchten retention curve. psi_cm = suction in cm, alpha in 1/cm."""
    psi_cm = np.abs(np.asarray(psi_cm, float))
    m = 1.0 - 1.0 / n
    Se = (1.0 + (alpha * psi_cm) ** n) ** (-m)
    return theta_r + (theta_s - theta_r) * Se


def bridge_vs_measured(d50_mm=0.2, sigma_grain=0.6, beta=0.2,
                       spread_factor=1.0, alpha_meas=0.04, n_meas=2.0,
                       theta_r=0.05, theta_s=0.45):
    """Overlay the grain-derived heuristic curve with a measured van Genuchten curve."""
    r_med_um, sigma_p = grains_to_pores(d50_mm, sigma_grain, beta, spread_factor)
    psi, theta = retention_from_pores(r_med_um, sigma_p, theta_r, theta_s)
    psi_cm = psi * 100.0

    vg = van_genuchten(psi_cm, alpha_meas, n_meas, theta_r, theta_s)
    mismatch = np.mean(np.abs(theta - vg))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(
        psi_cm, theta, "b-", lw=2.8,
        label="predicted from grain curve (heuristic)"
    )
    ax.semilogx(
        psi_cm, vg, "r--", lw=2.4,
        label=f"'measured' (van Genuchten, alpha={alpha_meas}, n={n_meas})"
    )
    ax.fill_between(psi_cm, theta, vg, color="0.6", alpha=0.3, label="error of the bridge")
    ax.set_xlabel("suction  |psi|  (cm of water)")
    ax.set_ylabel("water content  theta")
    ax.set_title(
        f"Heuristic prediction vs measured curve\n"
        f"mean |theta| mismatch = {mismatch:.3f}"
    )
    ax.grid(True, which="both", color="0.92", lw=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    return mismatch


def interactive_bridge_vs_measured():
    """Interactive sliders for comparing heuristic and measured retention curves."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception:
        print("ipywidgets not installed - call bridge_vs_measured(...) directly.")
        return bridge_vs_measured()

    d50b = widgets.FloatLogSlider(
        value=0.2, base=10, min=-3, max=1, step=0.05,
        description="d50 (mm)", continuous_update=False
    )
    sigb = widgets.FloatSlider(
        value=0.6, min=0.2, max=2.2, step=0.05,
        description="sigma_grain", continuous_update=False
    )
    betab = widgets.FloatSlider(
        value=0.2, min=0.02, max=0.6, step=0.01,
        description="beta", continuous_update=False
    )
    alm = widgets.FloatLogSlider(
        value=0.04, base=10, min=-3, max=0, step=0.05,
        description="alpha_meas", continuous_update=False
    )
    nm = widgets.FloatSlider(
        value=2.0, min=1.1, max=4.0, step=0.1,
        description="n_meas", continuous_update=False
    )

    ui = widgets.interactive(
        lambda d50_mm, sigma_grain, beta, alpha_meas, n_meas:
            bridge_vs_measured(
                d50_mm=d50_mm,
                sigma_grain=sigma_grain,
                beta=beta,
                spread_factor=1.0,
                alpha_meas=alpha_meas,
                n_meas=n_meas,
            ),
        d50_mm=d50b,
        sigma_grain=sigb,
        beta=betab,
        alpha_meas=alm,
        n_meas=nm,
    )
    display(ui)
    return ui
