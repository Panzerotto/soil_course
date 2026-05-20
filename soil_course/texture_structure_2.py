import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv

try:
    import ipywidgets as widgets
    from IPython.display import display
    HAS_WIDGETS = True
except Exception:
    HAS_WIDGETS = False

CLAY_SILT = 0.002
SILT_SAND = 0.05
SAND_GRAVEL = 2.0


def usda_texture_class(sand, clay):
    silt = 100.0 - sand - clay
    if sand < 0 or clay < 0 or silt < 0:
        return "INVALID (percentages must be >= 0 and sum to 100)"
    if clay >= 40 and silt < 40 and sand <= 45:
        return "Clay"
    if clay >= 40 and silt >= 40:
        return "Silty clay"
    if clay >= 35 and sand >= 45:
        return "Sandy clay"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "Clay loam"
    if 27 <= clay < 40 and sand <= 20:
        return "Silty clay loam"
    if 20 <= clay < 35 and sand > 45 and silt < 28:
        return "Sandy clay loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "Loam"
    if clay < 27 and 50 <= silt < 80:
        return "Silt loam"
    if silt >= 80 and clay < 12:
        return "Silt"
    if clay >= 7 and clay < 20 and sand > 52 and silt + 2 * clay >= 30:
        return "Sandy loam"
    if clay < 7 and silt < 50 and silt + 2 * clay >= 30:
        return "Sandy loam"
    if silt + 1.5 * clay >= 15 and silt + 2 * clay < 30:
        return "Loamy sand"
    if silt + 1.5 * clay < 15:
        return "Sand"
    return "Loam"


def check_usda_texture_examples():
    examples = [(90, 5, "Sand"), (40, 40, "Clay"), (40, 20, "Loam"),
                (20, 10, "Silt loam"), (65, 10, "Sandy loam")]
    for sand, clay, expected in examples:
        print(f"sand={sand:3d}%  clay={clay:3d}%  ->  {usda_texture_class(sand, clay):16s}  (expected ~ {expected})")


def _ternary_xy(sand, silt, clay):
    total = sand + silt + clay
    x = 0.5 * (2 * silt / total + clay / total)
    y = (np.sqrt(3) / 2) * clay / total
    return x, y


def plot_texture_triangle(sand=40.0, clay=20.0, ax=None):
    silt = 100.0 - sand - clay
    name = usda_texture_class(sand, clay)
    own_axes = ax is None
    if own_axes:
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
    V_sand = _ternary_xy(100, 0, 0)
    V_silt = _ternary_xy(0, 100, 0)
    V_clay = _ternary_xy(0, 0, 100)
    tri = np.array([V_sand, V_silt, V_clay, V_sand])
    ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1.5)
    for f in range(10, 100, 10):
        p1 = _ternary_xy(100 - f, 0, f); p2 = _ternary_xy(0, 100 - f, f)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="0.85", lw=0.6)
        p1 = _ternary_xy(f, 100 - f, 0); p2 = _ternary_xy(f, 0, 100 - f)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="0.85", lw=0.6)
        p1 = _ternary_xy(100 - f, f, 0); p2 = _ternary_xy(0, f, 100 - f)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="0.85", lw=0.6)
    ax.text(V_sand[0]-0.04, V_sand[1]-0.05, "100%\nsand", ha="center", va="top", fontsize=11, weight="bold")
    ax.text(V_silt[0]+0.04, V_silt[1]-0.05, "100%\nsilt", ha="center", va="top", fontsize=11, weight="bold")
    ax.text(V_clay[0], V_clay[1]+0.05, "100% clay", ha="center", va="bottom", fontsize=11, weight="bold")
    for f in range(20, 100, 20):
        p = _ternary_xy(100 - f, 0, f); ax.text(p[0]-0.03, p[1], f"{f}", ha="right", va="center", fontsize=8, color="0.4")
        p = _ternary_xy(f, 100 - f, 0); ax.text(p[0], p[1]-0.03, f"{f}", ha="center", va="top", fontsize=8, color="0.4")
        p = _ternary_xy(0, 100 - f, f); ax.text(p[0]+0.03, p[1], f"{f}", ha="left", va="center", fontsize=8, color="0.4")
    ax.text(0.5*(V_sand[0]+V_silt[0]), V_sand[1]-0.11, "percent sand  -->", ha="center", fontsize=9, color="0.4")
    valid = sand >= 0 and clay >= 0 and silt >= 0
    if valid:
        x, y = _ternary_xy(sand, silt, clay)
        ax.plot(x, y, "o", ms=14, mfc="red", mec="k", mew=1.4, zorder=5)
        ax.annotate(f"  ({sand:.0f}, {silt:.0f}, {clay:.0f})", (x, y), fontsize=9, weight="bold")
    ax.set_aspect("equal"); ax.axis("off")
    title = f"sand {sand:.0f}%   silt {silt:.0f}%   clay {clay:.0f}%\nUSDA class:  {name}"
    if not valid:
        title = "INVALID: percentages must be >= 0 and sum to 100"
    ax.set_title(title, fontsize=12)
    if own_axes:
        plt.tight_layout(); plt.show()
    return name


def interactive_texture_triangle():
    import ipywidgets as widgets
    from IPython.display import display

    sand_slider = widgets.FloatSlider(
        value=40,
        min=0,
        max=100,
        step=1,
        description="Sand (%)"
    )

    clay_slider = widgets.FloatSlider(
        value=20,
        min=0,
        max=100,
        step=1,
        description="Clay (%)"
    )

    def update(sand, clay):
        if sand + clay > 100:
            print("Sand + clay cannot exceed 100%. Reduce one value.")
        else:
            plot_texture_triangle(sand=sand, clay=clay)

    ui = widgets.interactive(
        update,
        sand=sand_slider,
        clay=clay_slider
    )

    display(ui)

def percent_finer(d, d50=0.05, sigma=1.2):
    d = np.asarray(d, dtype=float)
    return 100.0 * 0.5 * (1.0 + erf((np.log(d) - np.log(d50)) / (sigma * np.sqrt(2.0))))


def diameter_at_percent(p, d50=0.05, sigma=1.2):
    return d50 * np.exp(sigma * np.sqrt(2.0) * erfinv(2.0 * p / 100.0 - 1.0))


def classify_grain_curve(d50=0.05, sigma=1.2):
    clay = percent_finer(CLAY_SILT, d50, sigma)
    silt = percent_finer(SILT_SAND, d50, sigma) - clay
    sand = percent_finer(SAND_GRAVEL, d50, sigma) - clay - silt
    gravel = 100.0 - clay - silt - sand
    fractions = {"clay": clay, "silt": silt, "sand": sand, "gravel": gravel}
    dominant = max(fractions, key=fractions.get)
    D10 = diameter_at_percent(10, d50, sigma)
    D30 = diameter_at_percent(30, d50, sigma)
    D60 = diameter_at_percent(60, d50, sigma)
    Cu = D60 / D10
    Cc = D30**2 / (D10 * D60)
    if dominant == "gravel":
        well_graded = (Cu > 4.0) and (1.0 < Cc < 3.0)
    else:
        well_graded = (Cu > 6.0) and (1.0 < Cc < 3.0)
    grading = "well-graded" if well_graded else "poorly-graded (uniform)"
    return {"fractions": fractions, "dominant": dominant, "D10": D10, "D30": D30, "D60": D60,
            "Cu": Cu, "Cc": Cc, "grading": grading, "name": f"{grading} {dominant}"}


def print_grain_curve_classification(d50=0.3, sigma=0.5):
    info = classify_grain_curve(d50=d50, sigma=sigma)
    for k, v in info.items():
        print(f"{k:12s}: {v}")
    return info


def plot_grain_curve(d50=0.05, sigma=1.2, ax=None):
    own_axes = ax is None
    if own_axes:
        fig, ax = plt.subplots(figsize=(8.4, 5.4))
    d = np.logspace(-4, 2, 400)
    N = percent_finer(d, d50, sigma)
    bands = [(1e-4, CLAY_SILT, "#67c8e8", "Clay"), (CLAY_SILT, SILT_SAND, "#7ad07a", "Silt"),
             (SILT_SAND, SAND_GRAVEL, "#e89ad0", "Sand"), (SAND_GRAVEL, 1e2, "#f2e85a", "Gravel")]
    for x0, x1, color, label in bands:
        ax.axvspan(x0, x1, color=color, alpha=0.35)
        ax.text(np.sqrt(x0*x1), 2, label, ha="center", va="bottom", fontsize=11, weight="bold")
    ax.semilogx(d, N, "k-", lw=2.6)
    info = classify_grain_curve(d50, sigma)
    for p, Dp, col in [(10, info["D10"], "tab:blue"), (30, info["D30"], "tab:orange"), (60, info["D60"], "tab:red")]:
        if 1e-4 < Dp < 1e2:
            ax.plot([Dp, Dp], [0, p], "--", color=col, lw=1.0)
            ax.plot([1e-4, Dp], [p, p], "--", color=col, lw=1.0)
            ax.plot(Dp, p, "o", color=col, ms=7)
            ax.text(Dp, p+3, f"D{p}", color=col, fontsize=9, ha="center")
    ax.set_xlim(1e-4, 1e2); ax.set_ylim(0, 100)
    ax.set_xlabel("grain diameter  d  (mm)  -- log scale")
    ax.set_ylabel("percentage finer  N  (%)")
    ax.grid(True, which="both", color="0.9", lw=0.5)
    f = info["fractions"]
    ax.set_title(f"d50 = {d50:.3g} mm    sigma = {sigma:.2f}\n"
                 f"clay {f['clay']:.0f}%  silt {f['silt']:.0f}%  sand {f['sand']:.0f}%  gravel {f['gravel']:.0f}%   |   "
                 f"Cu = {info['Cu']:.1f}, Cc = {info['Cc']:.2f}\nclassification:  {info['name'].upper()}", fontsize=11)
    if own_axes:
        plt.tight_layout(); plt.show()
    return info


def interactive_grain_curve():
    import ipywidgets as widgets
    from IPython.display import display

    d50_slider = widgets.FloatLogSlider(
        value=0.05,
        base=10,
        min=-4,
        max=2,
        step=0.1,
        description="d50"
    )

    sigma_slider = widgets.FloatSlider(
        value=1.2,
        min=0.2,
        max=3.0,
        step=0.1,
        description="sigma"
    )

    ui = widgets.interactive(
        plot_grain_curve,
        d50=d50_slider,
        sigma=sigma_slider
    )

    display(ui)

def curve_to_triangle(d50=0.05, sigma=1.2):
    info = classify_grain_curve(d50, sigma)
    f = info["fractions"]
    fines = f["clay"] + f["silt"] + f["sand"]
    if fines < 1e-6:
        print("This soil is essentially all gravel - the texture triangle does not apply.")
        return None
    sand = 100.0 * f["sand"] / fines
    clay = 100.0 * f["clay"] / fines
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    plot_grain_curve(d50=d50, sigma=sigma, ax=axes[0])
    texture_class = plot_texture_triangle(sand=sand, clay=clay, ax=axes[1])
    if f["gravel"] > 1:
        axes[1].text(0.5, -0.02, f"({f['gravel']:.0f}% gravel removed before renormalising)", transform=axes[1].transAxes, ha="center", fontsize=9, color="0.4")
    plt.tight_layout(); plt.show()
    return texture_class


def interactive_curve_to_triangle():
    import ipywidgets as widgets
    from IPython.display import display

    d50_slider = widgets.FloatLogSlider(
        value=0.05,
        base=10,
        min=-4,
        max=2,
        step=0.1,
        description="d50"
    )

    sigma_slider = widgets.FloatSlider(
        value=1.2,
        min=0.2,
        max=3.0,
        step=0.1,
        description="sigma"
    )

    ui = widgets.interactive(
        curve_to_triangle,
        d50=d50_slider,
        sigma=sigma_slider
    )

    display(ui)


PACKINGS = {"dense (rhombohedral)": 0.26, "random close": 0.36, "loose (cubic)": 0.48, "card-house clay": 0.70}


def bulk_density(porosity, particle_density=2.65):
    return (1.0 - porosity) * particle_density


def print_packing_table(particle_density=2.65):
    print(f"{'packing':24s}{'porosity n':>12s}{'bulk density':>15s}")
    print("-" * 51)
    for name, n in PACKINGS.items():
        rb = bulk_density(n, particle_density=particle_density)
        print(f"{name:24s}{n:12.2f}{rb:12.2f} g/cm3")
    print("\nSame grains throughout - only the packing differs.")


def bimodal_pore_distribution(r_textural=0.5, r_structural=50.0, sigma=0.5, f_struct=0.5, n_classes=400):
    r = np.logspace(np.log10(r_textural) - 4*sigma, np.log10(r_structural) + 4*sigma, n_classes)
    lr = np.log(r)
    textural = np.exp(-0.5 * ((lr - np.log(r_textural)) / sigma)**2)
    structural = np.exp(-0.5 * ((lr - np.log(r_structural)) / sigma)**2)
    textural /= textural.sum(); structural /= structural.sum()
    vol_frac = (1.0 - f_struct) * textural + f_struct * structural
    vol_frac /= vol_frac.sum()
    return r, vol_frac


def plot_bimodal_pores(r_textural=0.5, r_structural=50.0, sigma=0.5, f_struct=0.5):
    r, vf = bimodal_pore_distribution(r_textural, r_structural, sigma, f_struct)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.semilogx(r, vf, "b-", lw=2.4)
    ax.fill_between(r, vf, color="tab:blue", alpha=0.25)
    ax.axvline(r_textural, color="0.5", ls=":", lw=1)
    ax.axvline(r_structural, color="0.5", ls=":", lw=1)
    ax.text(r_textural, ax.get_ylim()[1]*0.92, " textural mode\n (intra-aggregate,\n  small pores)", fontsize=9, ha="center", va="top")
    ax.text(r_structural, ax.get_ylim()[1]*0.92, " structural mode\n (inter-aggregate,\n  large pores)", fontsize=9, ha="center", va="top")
    ax.set_xlabel("pore radius  r  (micrometres)")
    ax.set_ylabel("pore-volume fraction")
    ax.set_title(f"Bimodal (dual-porosity) pore structure  (structural fraction f_struct = {f_struct:.2f})")
    ax.grid(True, which="both", color="0.92", lw=0.5)
    plt.tight_layout(); plt.show()


def interactive_bimodal_pores():
    if not HAS_WIDGETS:
        print("ipywidgets not installed - call plot_bimodal_pores(...) directly.")
        return plot_bimodal_pores(f_struct=0.7)
    fstruct_sl = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description="f_struct", continuous_update=False)
    rt_sl = widgets.FloatLogSlider(value=0.5, base=10, min=-1, max=1, step=0.1, description="r_textural (um)", continuous_update=False)
    rs_sl = widgets.FloatLogSlider(value=50.0, base=10, min=1, max=3, step=0.1, description="r_structural (um)", continuous_update=False)
    sg_sl = widgets.FloatSlider(value=0.5, min=0.2, max=1.2, step=0.05, description="spread", continuous_update=False)
    def _show(f_struct, r_textural, r_structural, sigma):
        if r_textural >= r_structural:
            print("r_textural must be smaller than r_structural.")
            return
        plot_bimodal_pores(r_textural, r_structural, sigma, f_struct)
    display(widgets.interactive(_show, f_struct=fstruct_sl, r_textural=rt_sl, r_structural=rs_sl, sigma=sg_sl))


def drainage_split(r_textural=0.5, r_structural=50.0, sigma=0.5, f_struct=0.5, cutoff_um=25.0):
    r, vf = bimodal_pore_distribution(r_textural, r_structural, sigma, f_struct)
    draining = vf[r >= cutoff_um].sum()
    retentive = vf[r < cutoff_um].sum()
    return draining, retentive


def print_drainage_table(cutoff_um=25.0):
    print(f"freely-draining pore fraction (pores > {cutoff_um:g} um):\n")
    print(f"{'f_struct':>12s}{'draining':>12s}{'retentive':>12s}")
    for fs in [0.0, 0.25, 0.5, 0.75, 1.0]:
        d, rkeep = drainage_split(f_struct=fs, cutoff_um=cutoff_um)
        print(f"{fs:12.2f}{d:12.2f}{rkeep:12.2f}")
    print("\nSame texture every row -- only the structural fraction changes.")
    print("More structural porosity -> more large inter-aggregate pores -> faster drainage.")
