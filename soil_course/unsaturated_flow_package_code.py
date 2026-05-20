
"""
Unsaturated flow teaching helpers.

This module hides the plotting and widget code for the notebook:
03x02_Unsaturated_Flow.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.stats import norm


VG_PARAMS = {
    "sand": dict(alpha=0.145, n=2.68),
    "silt": dict(alpha=0.020, n=1.41),
    "clay": dict(alpha=0.008, n=1.09),
}

BC_PARAMS = {
    "sand": dict(hb=7.26, lam=0.694),
    "silt": dict(hb=20.8, lam=0.211),
    "clay": dict(hb=40.5, lam=0.131),
}

KO_PARAMS = {
    "sand": dict(hm=10.0, sigma=0.9),
    "silt": dict(hm=70.0, sigma=1.8),
    "clay": dict(hm=350.0, sigma=2.5),
}

KS_SOILS = {
    "sand": 712.8,
    "silt": 43.7,
    "clay": 14.7,
}


def setup_unsaturated_flow(dpi=110, font_size=12):
    """Set plotting defaults for the unsaturated-flow notebook."""
    plt.rcParams.update({"figure.dpi": dpi, "font.size": font_size})
    print("Unsaturated-flow tools loaded.")


def swrc_van_genuchten(h, alpha, n):
    """van Genuchten effective saturation. h = suction (cm), alpha in 1/cm."""
    h = np.abs(np.asarray(h, float))
    m = 1.0 - 1.0 / n
    return (1.0 + (alpha * h) ** n) ** (-m)


def swrc_brooks_corey(h, hb, lam):
    """Brooks-Corey effective saturation. h, hb = suction and air-entry head (cm)."""
    h = np.abs(np.asarray(h, float))
    return np.where(h >= hb, (hb / h) ** lam, 1.0)


def swrc_kosugi(h, hm, sigma):
    """Kosugi effective saturation. h, hm = suction and median head (cm)."""
    h = np.abs(np.asarray(h, float))
    h = np.maximum(h, 1e-12)
    return 0.5 * erfc(np.log(h / hm) / (sigma * np.sqrt(2.0)))


def plot_swrc_model_comparison(texture="silt"):
    """Overlay van Genuchten, Brooks-Corey and Kosugi retention curves."""
    h = np.logspace(-1, 5, 400)

    vg = swrc_van_genuchten(h, **VG_PARAMS[texture])
    bc = swrc_brooks_corey(h, **BC_PARAMS[texture])
    ko = swrc_kosugi(h, **KO_PARAMS[texture])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(h, vg, "b-", lw=2.4, label="van Genuchten")
    ax.semilogx(h, bc, "r--", lw=2.2, label="Brooks-Corey")
    ax.semilogx(h, ko, "g-.", lw=2.2, label="Kosugi")
    ax.set_xlabel("suction  |psi|  (cm of water)")
    ax.set_ylabel("effective saturation  Se")
    ax.set_title(f"Soil water retention curve - {texture}")
    ax.grid(True, which="both", color="0.92", lw=0.5)
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.show()


# Backwards-compatible alias, in case older notebooks call this name.
plot_swrc_comparison_unsaturated = plot_swrc_model_comparison


def plot_single_swrc(model="van Genuchten", p1=0.02, p2=1.6):
    """Plot one retention curve with manually chosen parameters."""
    h = np.logspace(-1, 5, 400)

    if model == "van Genuchten":
        Se = swrc_van_genuchten(h, alpha=p1, n=p2)
        label = f"alpha={p1:.3f} 1/cm,  n={p2:.2f}"
    elif model in ["Brooks-Corey", "Brooks–Corey"]:
        Se = swrc_brooks_corey(h, hb=p1, lam=p2)
        label = f"hb={p1:.1f} cm,  lambda={p2:.2f}"
        model = "Brooks-Corey"
    elif model == "Kosugi":
        Se = swrc_kosugi(h, hm=p1, sigma=p2)
        label = f"hm={p1:.1f} cm,  sigma={p2:.2f}"
    else:
        raise ValueError("model must be 'van Genuchten', 'Brooks-Corey', or 'Kosugi'.")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.semilogx(h, Se, "b-", lw=2.6)
    ax.set_xlabel("suction  |psi|  (cm)")
    ax.set_ylabel("effective saturation  Se")
    ax.set_title(f"{model}   ({label})")
    ax.grid(True, which="both", color="0.92", lw=0.5)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.show()


def interactive_swrc_shape():
    """Interactive sliders for reshaping the retention curve."""
    import ipywidgets as widgets
    from IPython.display import display

    model_dd = widgets.Dropdown(
        options=["van Genuchten", "Brooks-Corey", "Kosugi"],
        value="van Genuchten",
        description="model",
    )

    p1_sl = widgets.FloatSlider(
        value=0.02,
        min=0.001,
        max=0.3,
        step=0.001,
        description="alpha (1/cm)",
        readout_format=".3f",
        continuous_update=False,
    )

    p2_sl = widgets.FloatSlider(
        value=1.6,
        min=1.05,
        max=4.0,
        step=0.05,
        description="n",
        continuous_update=False,
    )

    def retune(change=None):
        if model_dd.value == "van Genuchten":
            p1_sl.min, p1_sl.max, p1_sl.step = 0.001, 0.3, 0.001
            p1_sl.value = min(max(p1_sl.value, 0.001), 0.3)
            p1_sl.description = "alpha (1/cm)"
            p1_sl.readout_format = ".3f"
            p2_sl.min, p2_sl.max, p2_sl.step = 1.05, 4.0, 0.05
            p2_sl.value = min(max(p2_sl.value, 1.05), 4.0)
            p2_sl.description = "n"
        elif model_dd.value == "Brooks-Corey":
            p1_sl.min, p1_sl.max, p1_sl.step = 1.0, 100.0, 1.0
            p1_sl.value = min(max(p1_sl.value, 1.0), 100.0)
            p1_sl.description = "hb (cm)"
            p1_sl.readout_format = ".1f"
            p2_sl.min, p2_sl.max, p2_sl.step = 0.1, 1.5, 0.05
            p2_sl.value = min(max(p2_sl.value, 0.1), 1.5)
            p2_sl.description = "lambda"
        else:
            p1_sl.min, p1_sl.max, p1_sl.step = 1.0, 500.0, 1.0
            p1_sl.value = min(max(p1_sl.value, 1.0), 500.0)
            p1_sl.description = "hm (cm)"
            p1_sl.readout_format = ".1f"
            p2_sl.min, p2_sl.max, p2_sl.step = 0.3, 3.0, 0.05
            p2_sl.value = min(max(p2_sl.value, 0.3), 3.0)
            p2_sl.description = "sigma"

    model_dd.observe(retune, names="value")
    retune()

    out = widgets.interactive_output(
        lambda model, p1, p2: plot_single_swrc(model=model, p1=p1, p2=p2),
        {"model": model_dd, "p1": p1_sl, "p2": p2_sl},
    )

    display(widgets.VBox([model_dd, p1_sl, p2_sl, out]))


def Kr_van_genuchten(Se, n):
    """Mualem-van Genuchten relative conductivity."""
    Se = np.clip(np.asarray(Se, float), 1e-12, 1.0)
    m = 1.0 - 1.0 / n
    return Se ** 0.5 * (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2


def Kr_brooks_corey(Se, lam):
    """Mualem-Brooks-Corey relative conductivity."""
    Se = np.clip(np.asarray(Se, float), 1e-12, 1.0)
    return Se ** (2.0 / lam + 2.5)


def Kr_kosugi(Se, sigma):
    """Mualem-Kosugi relative conductivity, with sign-corrected expression."""
    Se = np.clip(np.asarray(Se, float), 1e-12, 1.0 - 1e-12)
    Qinv = norm.isf(Se)
    return Se ** 0.5 * norm.sf(Qinv + sigma) ** 2


def check_mualem_conductivity():
    """Print a quick numerical check of the three Mualem conductivity models."""
    for name, fn, par in [
        ("M-vG", Kr_van_genuchten, 2.0),
        ("M-BC", Kr_brooks_corey, 0.5),
        ("M-Ko", Kr_kosugi, 1.5),
    ]:
        print(
            f"{name}:  Kr(Se=1.0) = {fn(1.0, par):.4f}    "
            f"Kr(Se=0.2) = {fn(0.2, par):.3e}"
        )


def plot_Kr_comparison(texture="silt"):
    """Relative conductivity Kr(Se) for the three Mualem-based models."""
    Se = np.linspace(1e-3, 1.0, 400)

    kr_vg = Kr_van_genuchten(Se, VG_PARAMS[texture]["n"])
    kr_bc = Kr_brooks_corey(Se, BC_PARAMS[texture]["lam"])
    kr_ko = Kr_kosugi(Se, KO_PARAMS[texture]["sigma"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    for a in ax:
        a.plot(Se, kr_vg, "b-", lw=2.4, label="Mualem-van Genuchten")
        a.plot(Se, kr_bc, "r--", lw=2.2, label="Mualem-Brooks-Corey")
        a.plot(Se, kr_ko, "g-.", lw=2.2, label="Mualem-Kosugi")
        a.set_xlabel("effective saturation  Se")
        a.grid(True, which="both", color="0.92", lw=0.5)
        a.legend(fontsize=9)

    ax[0].set_ylabel("relative conductivity  Kr")
    ax[0].set_title(f"Kr(Se) - {texture}  (linear)")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-12, 2)
    ax[1].set_ylabel("relative conductivity  Kr  (log scale)")
    ax[1].set_title(f"Kr(Se) - {texture}  (log: note the collapse)")
    plt.tight_layout()
    plt.show()


def plot_K_of_psi(texture="silt", model="van Genuchten"):
    """K as a function of suction, chaining SWRC with Mualem conductivity."""
    h = np.logspace(-1, 5, 400)
    Ks = KS_SOILS[texture]

    if model == "van Genuchten":
        Se = swrc_van_genuchten(h, **VG_PARAMS[texture])
        Kr = Kr_van_genuchten(Se, VG_PARAMS[texture]["n"])
    elif model in ["Brooks-Corey", "Brooks–Corey"]:
        Se = swrc_brooks_corey(h, **BC_PARAMS[texture])
        Kr = Kr_brooks_corey(Se, BC_PARAMS[texture]["lam"])
        model = "Brooks-Corey"
    elif model == "Kosugi":
        Se = swrc_kosugi(h, **KO_PARAMS[texture])
        Kr = Kr_kosugi(Se, KO_PARAMS[texture]["sigma"])
    else:
        raise ValueError("model must be 'van Genuchten', 'Brooks-Corey', or 'Kosugi'.")

    K = Ks * Kr

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    ax[0].semilogx(h, Se, "b-", lw=2.4)
    ax[0].set_xlabel("suction  |psi|  (cm)")
    ax[0].set_ylabel("effective saturation  Se")
    ax[0].set_title(f"retention curve  Se(psi) - {texture}")
    ax[0].grid(True, which="both", color="0.92", lw=0.5)

    ax[1].loglog(h, K, "r-", lw=2.4)
    ax[1].set_xlabel("suction  |psi|  (cm)")
    ax[1].set_ylabel("hydraulic conductivity  K  (cm/day)")
    ax[1].set_title(f"Buckingham conductivity  K(psi) - {model}")
    ax[1].grid(True, which="both", color="0.92", lw=0.5)

    fig.suptitle(
        f"{texture}:  Ks = {Ks} cm/day  ->  K(psi) via {model} + Mualem",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


def interactive_K_of_psi():
    """Interactive selector for texture and model in the K(psi) plot."""
    import ipywidgets as widgets
    from IPython.display import display

    texture_dd = widgets.Dropdown(
        options=list(VG_PARAMS.keys()),
        value="silt",
        description="texture",
    )
    model_dd = widgets.Dropdown(
        options=["van Genuchten", "Brooks-Corey", "Kosugi"],
        value="van Genuchten",
        description="model",
    )

    out = widgets.interactive_output(
        lambda texture, model: plot_K_of_psi(texture=texture, model=model),
        {"texture": texture_dd, "model": model_dd},
    )

    display(widgets.VBox([texture_dd, model_dd, out]))
