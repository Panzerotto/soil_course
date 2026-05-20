import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "font.size": 12})

RHO_W = 1000.0
G = 9.81
MU_W = 1.0e-3

K_SOILS = {
    "gravel": 1e-2,
    "coarse sand": 1e-3,
    "fine sand": 1e-4,
    "silt": 1e-6,
    "clay": 1e-9,
}


def darcy_flow(delta_h, L, K, A=1.0):
    """Return specific discharge q and volumetric flow rate Q using Darcy's law."""
    gradient = delta_h / L
    q = K * gradient
    Q = q * A
    return q, Q


def darcy_example(delta_h=0.5, L=1.0, K=1e-4, A=0.01):
    """Print a simple Darcy-flow example."""
    q, Q = darcy_flow(delta_h=delta_h, L=L, K=K, A=A)
    print(f"head gradient        = {delta_h/L:.3f}  (dimensionless)")
    print(f"specific discharge q = {q:.3e} m/s   ( {q*1000*86400:.1f} mm/day )")
    print(f"flow rate Q          = {Q:.3e} m3/s")
    return q, Q


def run_darcy_experiment(K_true=1e-4, L=1.0, A=0.01, n_points=8, noise=0.03, seed=0):
    """Simulate a Darcy column experiment and recover K from the fitted slope."""
    rng = np.random.default_rng(seed)

    delta_h = np.linspace(0.05, 1.0, n_points)
    gradient = delta_h / L

    q_true = K_true * gradient
    q_meas = q_true * (1.0 + noise * rng.standard_normal(n_points))

    K_fit = np.sum(gradient * q_meas) / np.sum(gradient**2)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(
        gradient,
        q_meas * 1000 * 86400,
        "o",
        ms=9,
        mfc="tab:blue",
        mec="k",
        label="measurements",
    )

    g_line = np.linspace(0, gradient.max() * 1.05, 50)
    ax.plot(
        g_line,
        K_fit * g_line * 1000 * 86400,
        "r-",
        lw=2.2,
        label=f"Darcy's law, fitted K = {K_fit:.3e} m/s",
    )

    ax.set_xlabel("head gradient  delta_h / L  (dimensionless)")
    ax.set_ylabel("specific discharge  q  (mm/day)")
    ax.set_title("The Darcy experiment: discharge is proportional to gradient")
    ax.grid(True, color="0.92", lw=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"true  K = {K_true:.3e} m/s")
    print(f"fitted K = {K_fit:.3e} m/s   (error {100*(K_fit-K_true)/K_true:+.1f} %)")

    return K_fit


def interactive_darcy_experiment():
    """Interactive sliders/dropdown for the virtual Darcy experiment."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception:
        print("ipywidgets is not available. Use run_darcy_experiment(...) directly.")
        return run_darcy_experiment(K_true=K_SOILS["silt"])

    soil_dd = widgets.Dropdown(options=list(K_SOILS), value="fine sand", description="soil")
    noise_sl = widgets.FloatSlider(
        value=0.03,
        min=0.0,
        max=0.15,
        step=0.01,
        description="scatter",
        continuous_update=False,
    )
    npts_sl = widgets.IntSlider(
        value=8,
        min=4,
        max=20,
        step=1,
        description="# points",
        continuous_update=False,
    )

    def _run(soil, noise, n_points):
        run_darcy_experiment(K_true=K_SOILS[soil], noise=noise, n_points=n_points)

    ui = widgets.interactive(_run, soil=soil_dd, noise=noise_sl, n_points=npts_sl)
    display(ui)
    return ui


def K_to_permeability(K, rho=RHO_W, mu=MU_W):
    """Intrinsic permeability k [m2] from hydraulic conductivity K [m/s]."""
    return K * mu / (rho * G)


def permeability_to_K(k, rho=RHO_W, mu=MU_W):
    """Hydraulic conductivity K [m/s] from intrinsic permeability k [m2]."""
    return k * rho * G / mu


def print_permeability_table():
    """Print K, intrinsic permeability k, and k in darcy for representative materials."""
    print(f"{'material':14s}{'K (m/s)':>12s}{'k (m2)':>14s}{'k (darcy)':>14s}")
    for name, K in K_SOILS.items():
        k = K_to_permeability(K)
        k_darcy = k / 9.869e-13
        print(f"{name:14s}{K:12.1e}{k:14.2e}{k_darcy:14.2e}")


def conductivity_chart():
    """Plot the span of saturated hydraulic conductivity for earth materials."""
    materials = [
        ("clean gravel", 1e-1, 1e-2),
        ("coarse sand", 1e-2, 1e-4),
        ("fine sand", 1e-4, 1e-6),
        ("silt, loess", 1e-6, 1e-8),
        ("glacial till", 1e-8, 1e-10),
        ("unweath. clay", 1e-10, 1e-12),
        ("fresh granite", 1e-12, 1e-14),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.6))

    for i, (name, hi, lo) in enumerate(materials):
        ax.plot(
            [lo, hi],
            [i, i],
            lw=9,
            solid_capstyle="round",
            color=plt.cm.viridis(i / len(materials)),
        )
        ax.text(np.sqrt(lo * hi), i + 0.32, name, ha="center", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("saturated hydraulic conductivity  K  (m/s)")
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(materials) - 0.2)
    ax.set_title("K spans ~13 orders of magnitude across earth materials")
    ax.grid(True, axis="x", which="both", color="0.9", lw=0.5)
    plt.tight_layout()
    plt.show()


def compare_fluid_viscosity(k=1e-12, viscosity_factor=10):
    """Compare hydraulic conductivity for water and for a more viscous fluid."""
    K_water = permeability_to_K(k, mu=MU_W)
    K_more_viscous = permeability_to_K(k, mu=MU_W * viscosity_factor)

    print(f"intrinsic permeability k = {k:.2e} m2")
    print(f"K for water              = {K_water:.2e} m/s")
    print(f"K for {viscosity_factor:g}x more viscous fluid = {K_more_viscous:.2e} m/s")
    print("k does not change; K changes because the fluid viscosity changes.")

    return K_water, K_more_viscous
