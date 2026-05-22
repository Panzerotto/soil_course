"""Teaching helpers for Chapter 5: Solving Richards' equation.

Add this file to your package as:
    soil_course/soil_course/solving_richards.py

Then add to soil_course/soil_course/__init__.py:
    from .solving_richards import *
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "font.size": 12})

# A few soils (van Genuchten parameters, lengths in cm, time in days).
SOILS_RICHARDS = {
    # name : theta_r, theta_s, alpha[1/cm], n, Ks[cm/day]
    "sand": dict(tr=0.045, ts=0.43, a=0.145, n=2.68, Ks=712.8),
    "loam": dict(tr=0.078, ts=0.43, a=0.036, n=1.56, Ks=24.96),
    "silty clay": dict(tr=0.070, ts=0.36, a=0.005, n=1.09, Ks=0.48),
}

# Backwards-compatible short alias.
SOILS = SOILS_RICHARDS


def vg_theta_richards(psi, soil):
    """Water content theta(psi). psi in cm; negative = unsaturated."""
    psi = np.asarray(psi, float)
    m = 1.0 - 1.0 / soil["n"]
    Se = np.where(
        psi < 0,
        (1.0 + (soil["a"] * np.abs(psi)) ** soil["n"]) ** (-m),
        1.0,
    )
    return soil["tr"] + (soil["ts"] - soil["tr"]) * Se


def vg_capacity_richards(psi, soil):
    """Hydraulic capacity C = d theta / d psi (1/cm)."""
    psi = np.asarray(psi, float)
    m = 1.0 - 1.0 / soil["n"]
    ap = soil["a"] * np.abs(psi)
    num = soil["a"] * m * soil["n"] * (soil["ts"] - soil["tr"]) * ap ** (soil["n"] - 1.0)
    den = (1.0 + ap ** soil["n"]) ** (m + 1.0)
    C = np.where(psi < 0, num / den, 0.0)
    return np.maximum(C, 1e-8)


def vg_K_richards(psi, soil):
    """Hydraulic conductivity K(psi) (cm/day)."""
    psi = np.asarray(psi, float)
    m = 1.0 - 1.0 / soil["n"]
    Se = np.where(
        psi < 0,
        (1.0 + (soil["a"] * np.abs(psi)) ** soil["n"]) ** (-m),
        1.0,
    )
    Se = np.clip(Se, 1e-12, 1.0)
    K = soil["Ks"] * Se ** 0.5 * (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2
    return np.where(psi < 0, K, soil["Ks"])


# Also expose original names for continuity with the original notebook.
vg_theta = vg_theta_richards
vg_capacity = vg_capacity_richards
vg_K = vg_K_richards


def plot_richards_constitutive_relations():
    """Plot theta(psi), C(psi), and K(psi) for the predefined soils."""
    psi = -np.logspace(-1, 4, 300)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))

    for name, soil in SOILS_RICHARDS.items():
        ax[0].semilogx(-psi, vg_theta_richards(psi, soil), label=name)
        ax[1].loglog(-psi, vg_capacity_richards(psi, soil), label=name)
        ax[2].loglog(-psi, vg_K_richards(psi, soil), label=name)

    for axis, title, ylabel in zip(
        ax,
        ["theta(psi)", "capacity C(psi)", "K(psi)"],
        ["theta", "C  (1/cm)", "K  (cm/day)"],
    ):
        axis.set_xlabel("suction  -psi  (cm)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", color="0.92", lw=0.5)
        axis.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def _thomas(a, b, c, d):
    """Solve a tridiagonal system: sub-diag a, diag b, super-diag c, rhs d."""
    n = len(b)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve_richards(
    soil,
    depth=100.0,
    ncell=100,
    t_end=1.0,
    dt=0.002,
    psi_top=0.0,
    psi_init=-1000.0,
    bottom="free",
    max_iter=50,
    tol=1e-4,
    store_every=10,
):
    """1-D vertical Richards solver using a mixed-form modified-Picard scheme."""
    dz = depth / ncell
    z = (np.arange(ncell) + 0.5) * dz
    psi = np.full(ncell, float(psi_init))

    nsteps = int(round(t_end / dt))
    saved_t = [0.0]
    saved_psi = [psi.copy()]
    saved_th = [vg_theta_richards(psi, soil)]

    theta0_total = vg_theta_richards(psi, soil).sum() * dz
    cum_inflow = 0.0
    cum_outflow = 0.0

    def face_K(pk):
        Kc = vg_K_richards(pk, soil)
        return 0.5 * (Kc[:-1] + Kc[1:])

    for step in range(1, nsteps + 1):
        psi_old = psi.copy()
        theta_old = vg_theta_richards(psi_old, soil)
        psi_it = psi.copy()

        for _ in range(max_iter):
            C = vg_capacity_richards(psi_it, soil)
            theta_it = vg_theta_richards(psi_it, soil)
            Kf = face_K(psi_it)
            Ktop = vg_K_richards(np.array([psi_top]), soil)[0]
            Kbot = vg_K_richards(psi_it[-1:], soil)[0]

            lo = np.zeros(ncell)
            di = np.zeros(ncell)
            up = np.zeros(ncell)
            rhs = np.zeros(ncell)

            for i in range(ncell):
                if i == 0:
                    g_up = 2.0 * Ktop / dz**2
                    kg_up = Ktop
                    psi_up_known = psi_top
                else:
                    g_up = Kf[i - 1] / dz**2
                    kg_up = Kf[i - 1]
                    psi_up_known = None

                if i == ncell - 1:
                    if bottom == "free":
                        g_dn = 0.0
                        kg_dn = Kbot
                    else:
                        g_dn = 0.0
                        kg_dn = 0.0
                else:
                    g_dn = Kf[i] / dz**2
                    kg_dn = Kf[i]

                lo[i] = -g_up if i > 0 else 0.0
                up[i] = -g_dn if i < ncell - 1 else 0.0
                di[i] = C[i] / dt + g_up + g_dn

                rhs[i] = (
                    (C[i] / dt) * psi_it[i]
                    - (theta_it[i] - theta_old[i]) / dt
                    + (kg_up - kg_dn) / dz
                )
                if psi_up_known is not None:
                    rhs[i] += g_up * psi_up_known

            psi_new = _thomas(lo, di, up, rhs)
            if np.max(np.abs(psi_new - psi_it)) < tol:
                psi_it = psi_new
                break
            psi_it = psi_new

        psi = psi_it
        theta_step = vg_theta_richards(psi, soil)
        d_storage = (theta_step - theta_old).sum() * dz
        q_bot = vg_K_richards(psi[-1:], soil)[0] if bottom == "free" else 0.0
        q_top = d_storage / dt + q_bot
        cum_inflow += q_top * dt
        cum_outflow += q_bot * dt

        if step % store_every == 0 or step == nsteps:
            saved_t.append(step * dt)
            saved_psi.append(psi.copy())
            saved_th.append(vg_theta_richards(psi, soil))

    theta_total = vg_theta_richards(psi, soil).sum() * dz
    stored = theta_total - theta0_total
    mb_error = stored - (cum_inflow - cum_outflow)

    return dict(
        z=z,
        t=np.array(saved_t),
        psi=np.array(saved_psi),
        theta=np.array(saved_th),
        cum_inflow=cum_inflow,
        cum_outflow=cum_outflow,
        stored=stored,
        mb_error=mb_error,
        dz=dz,
    )


def plot_infiltration(
    soil_name="loam",
    t_end=1.0,
    psi_top=0.0,
    psi_init=-1000.0,
    bottom="free",
    depth=100.0,
    ncell=100,
    dt=0.002,
):
    """Run the solver and plot wetting-front profiles plus mass balance."""
    soil = SOILS_RICHARDS[soil_name]
    out = solve_richards(
        soil,
        depth=depth,
        ncell=ncell,
        t_end=t_end,
        dt=dt,
        psi_top=psi_top,
        psi_init=psi_init,
        bottom=bottom,
        store_every=max(1, int(round(t_end / dt)) // 8),
    )

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    cmap = plt.cm.plasma(np.linspace(0, 0.9, len(out["t"])))

    for k, tt in enumerate(out["t"]):
        ax[0].plot(out["theta"][k], out["z"], color=cmap[k], lw=2, label=f"t = {tt:.3g} d")
    ax[0].invert_yaxis()
    ax[0].set_xlabel("water content  theta")
    ax[0].set_ylabel("depth  z  (cm)   [downward]")
    ax[0].set_title(f"Wetting front - {soil_name}")
    ax[0].grid(True, color="0.92", lw=0.5)
    ax[0].legend(fontsize=8)

    for k, tt in enumerate(out["t"]):
        ax[1].plot(out["psi"][k], out["z"], color=cmap[k], lw=2)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("pressure head  psi  (cm)")
    ax[1].set_ylabel("depth  z  (cm)")
    ax[1].set_title("Pressure-head profiles")
    ax[1].grid(True, color="0.92", lw=0.5)

    mb = out["mb_error"]
    rel = mb / max(abs(out["cum_inflow"]), 1e-12)
    fig.suptitle(
        f"{soil_name}  |  ponding psi_top = {psi_top} cm  |  bottom = {bottom}\n"
        f"cumulative infiltration = {out['cum_inflow']:.3f} cm   "
        f"storage gain = {out['stored']:.3f} cm   "
        f"mass-balance error = {mb:.2e} cm ({rel:.1e} rel.)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()
    return out


def interactive_infiltration():
    """Interactive slider version of the Richards infiltration experiment."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception:
        print("ipywidgets is not installed - call plot_infiltration(...) directly.")
        return plot_infiltration("sand", t_end=0.3)

    soil_dd = widgets.Dropdown(options=list(SOILS_RICHARDS), value="loam", description="soil")
    tend_sl = widgets.FloatSlider(
        value=1.0, min=0.1, max=5.0, step=0.1, description="t_end (d)", continuous_update=False
    )
    pond_sl = widgets.FloatSlider(
        value=0.0, min=0.0, max=10.0, step=0.5, description="ponding", continuous_update=False
    )
    init_sl = widgets.FloatSlider(
        value=-1000, min=-5000, max=-50, step=50, description="psi_init", continuous_update=False
    )
    bot_dd = widgets.Dropdown(options=["free", "imperv"], value="free", description="bottom")

    def _run(soil, t_end, ponding, psi_init, bottom):
        return plot_infiltration(soil, t_end=t_end, psi_top=ponding, psi_init=psi_init, bottom=bottom)

    display(
        widgets.interactive(
            _run,
            soil=soil_dd,
            t_end=tend_sl,
            ponding=pond_sl,
            psi_init=init_sl,
            bottom=bot_dd,
        )
    )


def time_step_study(soil_name="loam", t_end=1.0, dts=(0.02, 0.005, 0.001)):
    """Overlay final profiles for several time steps."""
    soil = SOILS_RICHARDS[soil_name]
    fig, ax = plt.subplots(figsize=(7.5, 5.4))

    for dt in dts:
        out = solve_richards(
            soil,
            t_end=t_end,
            dt=dt,
            psi_top=0.0,
            psi_init=-1000.0,
            bottom="free",
            store_every=10**9,
        )
        ax.plot(out["theta"][-1], out["z"], lw=2, label=f"dt = {dt} d   (mb err {out['mb_error']:.1e} cm)")

    ax.invert_yaxis()
    ax.set_xlabel("water content  theta")
    ax.set_ylabel("depth  z  (cm)")
    ax.set_title(f"Effect of time step - {soil_name}, t = {t_end} d")
    ax.grid(True, color="0.92", lw=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
