import numpy as np
import matplotlib.pyplot as plt

SIGMA = 0.0728
RHO_W = 1000.0
G = 9.81

plt.rcParams.update({"figure.dpi": 110, "font.size": 12})


def young_laplace_head(r):
    """Suction head |psi| (m of water) that just empties a pore of radius r (m)."""
    return 2.0 * SIGMA / (RHO_W * G * np.asarray(r, float))


def head_to_radius(psi_head):
    """Inverse: pore radius (m) corresponding to suction head |psi_head| (m)."""
    return 2.0 * SIGMA / (RHO_W * G * np.asarray(psi_head, float))


def show_young_laplace_examples(radii_um=(1000, 100, 10, 1, 0.1)):
    """Print example pore radii and the suction head at which they empty."""
    for r_um in radii_um:
        r = r_um * 1e-6
        h = young_laplace_head(r)
        print(
            f"pore radius {r_um:7.1f} um  ->  empties at suction "
            f"{h:10.3f} m  ( {h*100:10.1f} cm )"
        )


def pore_size_distribution(r_median=20.0, sigma_p=1.0, n_classes=400):
    """Return a lognormal pore-size distribution.

    Parameters
    ----------
    r_median : float
        Median pore radius, in micrometres.
    sigma_p : float
        Standard deviation of ln(r), dimensionless.
    n_classes : int
        Number of discrete pore classes.

    Returns
    -------
    r : ndarray
        Pore radii in metres, ascending.
    vol_frac : ndarray
        Fraction of total pore volume in each class.
    """
    ln_med = np.log(r_median * 1e-6)
    ln_r = np.linspace(ln_med - 4 * sigma_p, ln_med + 4 * sigma_p, n_classes)
    r = np.exp(ln_r)
    dens = np.exp(-0.5 * ((ln_r - ln_med) / sigma_p) ** 2)
    vol_frac = dens / dens.sum()
    return r, vol_frac


def plot_pore_distribution(r_median=20.0, sigma_p=1.0):
    r, vf = pore_size_distribution(r_median, sigma_p)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.semilogx(r * 1e6, vf, "b-", lw=2)
    ax.fill_between(r * 1e6, vf, color="tab:blue", alpha=0.25)
    ax.set_xlabel("pore radius  r  (micrometres)")
    ax.set_ylabel("fraction of pore volume")
    ax.set_title(
        f"Pore-size distribution  (median {r_median} um, spread sigma = {sigma_p})"
    )
    ax.grid(True, which="both", color="0.92", lw=0.5)
    plt.tight_layout()
    plt.show()


def retention_from_pores(r_median=20.0, sigma_p=1.0, theta_r=0.05, theta_s=0.45):
    """Construct the SWRC by the 'fill smallest first' rule."""
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


def plot_constructed_swrc(r_median=20.0, sigma_p=1.0, theta_r=0.05, theta_s=0.45):
    """Show pore-size distribution and the SWRC built from it."""
    if theta_r >= theta_s:
        raise ValueError("theta_r must be smaller than theta_s.")

    psi, theta = retention_from_pores(r_median, sigma_p, theta_r, theta_s)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    r, vf = pore_size_distribution(r_median, sigma_p)
    ax[0].semilogx(r * 1e6, vf, "b-", lw=2)
    ax[0].fill_between(r * 1e6, vf, color="tab:blue", alpha=0.25)
    ax[0].set_xlabel("pore radius  r  (um)")
    ax[0].set_ylabel("pore-volume fraction")
    ax[0].set_title("Pore-size distribution")
    ax[0].grid(True, which="both", color="0.92", lw=0.5)

    ax[1].semilogx(psi * 100.0, theta, "b-", lw=2.6)
    ax[1].axhline(theta_s, color="0.6", ls=":", lw=1)
    ax[1].axhline(theta_r, color="0.6", ls=":", lw=1)
    ax[1].set_xlabel("suction  |psi|  (cm of water)")
    ax[1].set_ylabel("water content  theta")
    ax[1].set_title("Retention curve, CONSTRUCTED from the pores")
    ax[1].grid(True, which="both", color="0.92", lw=0.5)

    fig.suptitle(
        f"median pore {r_median} um, spread {sigma_p}, "
        f"theta_r={theta_r}, theta_s={theta_s}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


def interactive_constructed_swrc():
    """Interactive sliders for pore distribution -> constructed SWRC."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception:
        print("ipywidgets not available - call plot_constructed_swrc(...) directly.")
        plot_constructed_swrc(5.0, 1.6)
        return

    rmed_sl = widgets.FloatLogSlider(
        value=20, base=10, min=-1, max=3, step=0.1,
        description="median r (um)", continuous_update=False
    )
    sig_sl = widgets.FloatSlider(
        value=1.0, min=0.2, max=2.5, step=0.1,
        description="spread", continuous_update=False
    )
    tr_sl = widgets.FloatSlider(
        value=0.05, min=0.0, max=0.2, step=0.01,
        description="theta_r", continuous_update=False
    )
    ts_sl = widgets.FloatSlider(
        value=0.45, min=0.3, max=0.6, step=0.01,
        description="theta_s", continuous_update=False
    )

    def _update(rmed, sigma_p, theta_r, theta_s):
        if theta_r >= theta_s:
            print("theta_r must be smaller than theta_s.")
            return
        plot_constructed_swrc(rmed, sigma_p, theta_r, theta_s)

    display(
        widgets.interactive(
            _update, rmed=rmed_sl, sigma_p=sig_sl, theta_r=tr_sl, theta_s=ts_sl
        )
    )


def van_genuchten(psi_cm, alpha, n, theta_r=0.05, theta_s=0.45):
    """van Genuchten retention curve. psi_cm = suction in cm, alpha in 1/cm."""
    psi_cm = np.abs(np.asarray(psi_cm, float))
    m = 1.0 - 1.0 / n
    Se = (1.0 + (alpha * psi_cm) ** n) ** (-m)
    return theta_r + (theta_s - theta_r) * Se


def compare_curves(
    r_median=20.0, sigma_p=1.0, theta_r=0.05, theta_s=0.45, alpha=0.02, n=2.0
):
    """Compare the pore-integrated SWRC with the van Genuchten formula."""
    psi, theta = retention_from_pores(r_median, sigma_p, theta_r, theta_s)
    psi_cm = psi * 100.0

    vg = van_genuchten(psi_cm, alpha, n, theta_r, theta_s)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(psi_cm, theta, "b-", lw=2.8, label="constructed from pores")
    ax.semilogx(psi_cm, vg, "r--", lw=2.2, label=f"van Genuchten (alpha={alpha}, n={n})")
    ax.axhline(theta_s, color="0.6", ls=":", lw=1)
    ax.axhline(theta_r, color="0.6", ls=":", lw=1)
    ax.set_xlabel("suction  |psi|  (cm of water)")
    ax.set_ylabel("water content  theta")
    ax.set_title("Pore-integration curve vs the empirical van Genuchten formula")
    ax.grid(True, which="both", color="0.92", lw=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


def interactive_compare_curves():
    """Interactive sliders comparing constructed SWRC and van Genuchten."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception:
        print("ipywidgets not available - call compare_curves(...) directly.")
        compare_curves()
        return

    rmed2 = widgets.FloatLogSlider(
        value=20, base=10, min=-1, max=3, step=0.1,
        description="median r (um)", continuous_update=False
    )
    sig2 = widgets.FloatSlider(
        value=1.0, min=0.2, max=2.5, step=0.1,
        description="spread", continuous_update=False
    )
    al2 = widgets.FloatLogSlider(
        value=0.02, base=10, min=-3, max=0, step=0.1,
        description="alpha (1/cm)", continuous_update=False
    )
    n2 = widgets.FloatSlider(
        value=2.0, min=1.1, max=4.0, step=0.1,
        description="n", continuous_update=False
    )

    display(
        widgets.interactive(
            lambda rmed, sigma_p, alpha, n: compare_curves(
                r_median=rmed,
                sigma_p=sigma_p,
                theta_r=0.05,
                theta_s=0.45,
                alpha=alpha,
                n=n,
            ),
            rmed=rmed2,
            sigma_p=sig2,
            alpha=al2,
            n=n2,
        )
    )


def energy_rosetta(value, unit="head_m"):
    """Convert soil-water energy between head, potential, and chemical potential."""
    if unit == "head_m":
        head = value
        Psi = head * RHO_W * G
        mu = Psi / RHO_W
    elif unit == "potential_Pa":
        Psi = value
        head = Psi / (RHO_W * G)
        mu = Psi / RHO_W
    elif unit == "chempot_Jkg":
        mu = value
        Psi = mu * RHO_W
        head = Psi / (RHO_W * G)
    else:
        raise ValueError("unit must be head_m, potential_Pa or chempot_Jkg")

    return {
        "head_m": head,
        "head_cm": head * 100.0,
        "potential_Pa": Psi,
        "potential_kPa": Psi / 1e3,
        "chempot_Jkg": mu,
    }


def show_energy_rosetta(value=-150.0, unit="head_m"):
    """Print energy conversion in the three equivalent forms."""
    res = energy_rosetta(value, unit=unit)
    print("The same soil-water energy state expressed three ways:\n")
    print(f"  hydraulic head      h   = {res['head_m']:.4f} m  ( {res['head_cm']:.1f} cm )")
    print(f"  hydric potential    Psi = {res['potential_Pa']:.3e} Pa  ( {res['potential_kPa']:.2f} kPa )")
    print(f"  chemical potential  mu  = {res['chempot_Jkg']:.4f} J/kg")


def interactive_energy_rosetta():
    """Interactive converter between head, hydric potential, and chemical potential."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception:
        print("ipywidgets not available - call show_energy_rosetta(value, unit) directly.")
        show_energy_rosetta()
        return

    val_tx = widgets.FloatText(value=-150.0, description="value")
    unit_dd = widgets.Dropdown(
        options=[
            ("head (m of water)", "head_m"),
            ("hydric potential (Pa)", "potential_Pa"),
            ("chemical potential (J/kg)", "chempot_Jkg"),
        ],
        value="head_m",
        description="given as",
    )

    def _convert(value, unit):
        r = energy_rosetta(value, unit)
        print(f"  head            h   = {r['head_m']:12.4f} m   ( {r['head_cm']:.1f} cm )")
        print(f"  hydric potential Psi = {r['potential_Pa']:12.3e} Pa  ( {r['potential_kPa']:.2f} kPa )")
        print(f"  chemical pot.   mu  = {r['chempot_Jkg']:12.4f} J/kg")

    display(widgets.interactive(_convert, value=val_tx, unit=unit_dd))
