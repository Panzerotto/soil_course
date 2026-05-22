import numpy as np
import matplotlib.pyplot as plt

try:
    from rosetta import rosetta
    HAS_ROSETTA = True
except Exception:
    HAS_ROSETTA = False

try:
    import ipywidgets as widgets
    from IPython.display import display
    HAS_WIDGETS = True
except Exception:
    HAS_WIDGETS = False

plt.rcParams.update({"figure.dpi": 110, "font.size": 12})


def check_pedotransfer_setup():
    """Print whether optional dependencies needed for this assignment are available."""
    print("rosetta-soil available :", HAS_ROSETTA)
    print("ipywidgets   available :", HAS_WIDGETS)
    if not HAS_ROSETTA:
        print("  -> install with: pip install rosetta-soil")
    return HAS_ROSETTA, HAS_WIDGETS


def rosetta_vg(sand, silt, clay):
    """Return van Genuchten parameters and bootstrap standard deviations for one texture.

    Parameters
    ----------
    sand, silt, clay : float
        Soil texture mass percentages. They should sum to approximately 100.

    Returns
    -------
    dict
        theta_r, theta_s, alpha [1/cm], n, Ks [cm/day],
        plus theta_r_sd, theta_s_sd, alpha_sd, n_sd, Ks_sd and model_code.
    """
    if not HAS_ROSETTA:
        raise RuntimeError(
            "rosetta-soil is not installed. In Colab run: !pip install rosetta-soil"
        )

    total = sand + silt + clay
    if abs(total - 100) > 1e-6:
        raise ValueError(f"sand + silt + clay must equal 100. Current total = {total}")

    mean, sd, code = rosetta(3, [[sand, silt, clay]])
    m, s = mean[0], sd[0]

    return {
        "theta_r": m[0],
        "theta_s": m[1],
        "alpha": m[2],
        "n": m[3],
        "Ks": m[4],
        "theta_r_sd": s[0],
        "theta_s_sd": s[1],
        "alpha_sd": s[2],
        "n_sd": s[3],
        "Ks_sd": s[4],
        "model_code": int(code[0]),
    }


def print_rosetta_examples():
    """Print Rosetta estimates for common textures used in the assignment."""
    examples = [
        ("Sand", (92, 5, 3)),
        ("Loam", (40, 40, 20)),
        ("Silt loam", (20, 60, 20)),
        ("Clay", (20, 20, 60)),
    ]

    if not HAS_ROSETTA:
        print("Install rosetta-soil to run this function.")
        return None

    results = {}
    for name, (sa, si, cl) in examples:
        p = rosetta_vg(sa, si, cl)
        results[name] = p
        print(
            f"{name:11s}  theta_r={p['theta_r']:.3f}  theta_s={p['theta_s']:.3f}"
            f"  alpha={p['alpha']:.4f} 1/cm  n={p['n']:.3f}"
            f"  Ks={p['Ks']:.1f} cm/day"
        )
    return results


def vg_theta(psi_cm, p):
    """Volumetric water content at suction |psi| in cm."""
    psi = np.abs(np.asarray(psi_cm, float))
    m = 1.0 - 1.0 / p["n"]
    Se = (1.0 + (p["alpha"] * psi) ** p["n"]) ** (-m)
    return p["theta_r"] + (p["theta_s"] - p["theta_r"]) * Se


def vg_K(psi_cm, p, ell=0.5):
    """Mualem-van Genuchten hydraulic conductivity in cm/day."""
    psi = np.abs(np.asarray(psi_cm, float))
    m = 1.0 - 1.0 / p["n"]
    Se = (1.0 + (p["alpha"] * psi) ** p["n"]) ** (-m)
    return p["Ks"] * Se**ell * (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2


def plot_pedotransfer_soil(sand, silt=None, clay=None, ax=None):
    """Draw retention and conductivity curves for one texture.

    You may call either:
    plot_pedotransfer_soil(sand=40, silt=40, clay=20)
    or:
    plot_pedotransfer_soil(sand=40, clay=20)
    in which case silt is calculated as 100 - sand - clay.
    """
    if clay is None:
        raise ValueError("Please provide clay percentage.")

    if silt is None:
        silt = 100 - sand - clay

    if silt < 0:
        raise ValueError("sand + clay cannot exceed 100.")

    p = rosetta_vg(sand, silt, clay)
    psi = np.logspace(-1, 6, 300)

    own = ax is None
    if own:
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax[0].semilogx(psi, vg_theta(psi, p), lw=2.4)
    ax[0].set_xlabel("suction  |psi|  (cm)")
    ax[0].set_ylabel("water content  theta")
    ax[0].set_title("Retention curve  theta(psi)")
    ax[0].grid(True, which="both", color="0.9", lw=0.5)

    ax[1].loglog(psi, np.maximum(vg_K(psi, p), 1e-12), lw=2.4)
    ax[1].set_xlabel("suction  |psi|  (cm)")
    ax[1].set_ylabel("K  (cm/day)")
    ax[1].set_title("Hydraulic conductivity  K(psi)")
    ax[1].grid(True, which="both", color="0.9", lw=0.5)

    if own:
        fig.suptitle(
            f"sand {sand:.0f} / silt {silt:.0f} / clay {clay:.0f}   "
            f"->   theta_r={p['theta_r']:.3f}, theta_s={p['theta_s']:.3f}, "
            f"alpha={p['alpha']:.4f}, n={p['n']:.2f}, Ks={p['Ks']:.1f} cm/day",
            fontsize=10,
        )
        plt.tight_layout()
        plt.show()

    return p


# Backward-compatible shorter alias
plot_soil = plot_pedotransfer_soil


def interactive_pedotransfer_soil():
    """Interactive sand/clay sliders for Rosetta retention and conductivity curves."""
    if not HAS_ROSETTA:
        print("Need rosetta-soil for sliders. In Colab run: !pip install rosetta-soil")
        return None

    if not HAS_WIDGETS:
        print("Need ipywidgets for sliders; otherwise call plot_pedotransfer_soil(sand, silt, clay).")
        return None

    sand_sl = widgets.FloatSlider(
        value=40,
        min=0,
        max=100,
        step=1,
        description="sand %",
        continuous_update=False,
    )
    clay_sl = widgets.FloatSlider(
        value=20,
        min=0,
        max=100,
        step=1,
        description="clay %",
        continuous_update=False,
    )

    def update(sand, clay):
        if sand + clay > 100:
            print(f"sand + clay = {sand + clay:.0f}% > 100% : impossible.")
            return
        plot_pedotransfer_soil(sand=sand, silt=100 - sand - clay, clay=clay)

    ui = widgets.interactive(update, sand=sand_sl, clay=clay_sl)
    display(ui)
    return ui


def compare_pedotransfer_soils(soils=None):
    """Overlay retention and conductivity curves for several soils.

    Parameters
    ----------
    soils : list of tuples, optional
        Each tuple is (label, sand, silt, clay).
    """
    if soils is None:
        soils = [
            ("Sand", 92, 5, 3),
            ("Loam", 40, 40, 20),
            ("Silt loam", 20, 60, 20),
            ("Clay", 20, 20, 60),
        ]

    psi = np.logspace(-1, 6, 300)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for label, sa, si, cl in soils:
        p = rosetta_vg(sa, si, cl)
        ax[0].semilogx(psi, vg_theta(psi, p), lw=2.2, label=label)
        ax[1].loglog(psi, np.maximum(vg_K(psi, p), 1e-12), lw=2.2, label=label)

    ax[0].set(xlabel="suction |psi| (cm)", ylabel="theta", title="Retention curves")
    ax[1].set(xlabel="suction |psi| (cm)", ylabel="K (cm/day)", title="Conductivity curves")

    for a in ax:
        a.grid(True, which="both", color="0.9", lw=0.5)
        a.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


# Backward-compatible alias
compare_soils = compare_pedotransfer_soils


def plot_pedotransfer_uncertainty(sand, silt=None, clay=None):
    """Retention curve with a +/- 1 sigma band from Rosetta bootstrap stdev."""
    if clay is None:
        raise ValueError("Please provide clay percentage.")

    if silt is None:
        silt = 100 - sand - clay

    if silt < 0:
        raise ValueError("sand + clay cannot exceed 100.")

    p = rosetta_vg(sand, silt, clay)
    psi = np.logspace(-1, 6, 300)

    base = vg_theta(psi, p)

    hi = dict(p)
    hi["n"] = p["n"] + p["n_sd"]
    hi["alpha"] = p["alpha"] + p["alpha_sd"]

    lo = dict(p)
    lo["n"] = max(1.05, p["n"] - p["n_sd"])
    lo["alpha"] = max(1e-4, p["alpha"] - p["alpha_sd"])

    band_hi = vg_theta(psi, hi)
    band_lo = vg_theta(psi, lo)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(
        psi,
        np.minimum(band_lo, band_hi),
        np.maximum(band_lo, band_hi),
        alpha=0.25,
        label="+/- 1 sigma (PTF bootstrap)",
    )
    ax.semilogx(psi, base, lw=2.4, label="Rosetta estimate")
    ax.set(
        xlabel="suction |psi| (cm)",
        ylabel="water content theta",
        title=(
            f"Retention curve and PTF uncertainty\n"
            f"sand {sand:.0f} / silt {silt:.0f} / clay {clay:.0f}"
        ),
    )
    ax.grid(True, which="both", color="0.9", lw=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


# Backward-compatible alias
plot_with_uncertainty = plot_pedotransfer_uncertainty


def theta_at_suction(sand, silt, clay, suction_cm=100):
    """Return water content theta for one texture at a chosen suction."""
    p = rosetta_vg(sand, silt, clay)
    return float(vg_theta(suction_cm, p))


def pedotransfer_parameter_table(soils=None):
    """Return a table of Rosetta parameters for several soils."""
    import pandas as pd

    if soils is None:
        soils = [
            ("Sand", 92, 5, 3),
            ("Loam", 40, 40, 20),
            ("Silt loam", 20, 60, 20),
            ("Clay", 20, 20, 60),
        ]

    rows = []
    for label, sand, silt, clay in soils:
        p = rosetta_vg(sand, silt, clay)
        rows.append(
            {
                "soil": label,
                "sand": sand,
                "silt": silt,
                "clay": clay,
                "theta_r": p["theta_r"],
                "theta_s": p["theta_s"],
                "alpha_1_per_cm": p["alpha"],
                "n": p["n"],
                "Ks_cm_per_day": p["Ks"],
            }
        )

    return pd.DataFrame(rows)
