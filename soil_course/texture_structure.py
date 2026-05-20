import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv

CLAY_SILT = 0.002
SILT_SAND = 0.05
SAND_GRAVEL = 2.0


def usda_texture_class(sand, clay):
    silt = 100.0 - sand - clay

    if sand < 0 or clay < 0 or silt < 0:
        return "INVALID"

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


def _ternary_xy(sand, silt, clay):
    total = sand + silt + clay
    x = 0.5 * (2 * silt / total + clay / total)
    y = (np.sqrt(3) / 2) * clay / total
    return x, y


def plot_texture_triangle(sand=40, clay=20, ax=None):
    silt = 100 - sand - clay
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
        p1 = _ternary_xy(100 - f, 0, f)
        p2 = _ternary_xy(0, 100 - f, f)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="0.85", lw=0.6)

        p1 = _ternary_xy(f, 100 - f, 0)
        p2 = _ternary_xy(f, 0, 100 - f)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="0.85", lw=0.6)

        p1 = _ternary_xy(100 - f, f, 0)
        p2 = _ternary_xy(0, f, 100 - f)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="0.85", lw=0.6)

    if sand >= 0 and clay >= 0 and silt >= 0:
        x, y = _ternary_xy(sand, silt, clay)
        ax.plot(x, y, "o", ms=14, mfc="red", mec="k", mew=1.4)
        ax.annotate(f"({sand:.0f}, {silt:.0f}, {clay:.0f})", (x, y))

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"sand {sand:.0f}%   silt {silt:.0f}%   clay {clay:.0f}%\nUSDA class: {name}"
    )

    if own_axes:
        plt.tight_layout()
        plt.show()

    return name


def percent_finer(d, d50=0.05, sigma=1.2):
    d = np.asarray(d, dtype=float)
    return 100 * 0.5 * (
        1 + erf((np.log(d) - np.log(d50)) / (sigma * np.sqrt(2)))
    )


def diameter_at_percent(p, d50=0.05, sigma=1.2):
    return d50 * np.exp(
        sigma * np.sqrt(2) * erfinv(2 * p / 100 - 1)
    )


def classify_grain_curve(d50=0.05, sigma=1.2):
    clay = percent_finer(CLAY_SILT, d50, sigma)
    silt = percent_finer(SILT_SAND, d50, sigma) - clay
    sand = percent_finer(SAND_GRAVEL, d50, sigma) - clay - silt
    gravel = 100 - clay - silt - sand

    fractions = {
        "clay": clay,
        "silt": silt,
        "sand": sand,
        "gravel": gravel,
    }

    dominant = max(fractions, key=fractions.get)

    D10 = diameter_at_percent(10, d50, sigma)
    D30 = diameter_at_percent(30, d50, sigma)
    D60 = diameter_at_percent(60, d50, sigma)

    Cu = D60 / D10
    Cc = D30**2 / (D10 * D60)

    if dominant == "gravel":
        well_graded = Cu > 4 and 1 < Cc < 3
    else:
        well_graded = Cu > 6 and 1 < Cc < 3

    grading = "well-graded" if well_graded else "poorly-graded (uniform)"

    return {
        "fractions": fractions,
        "dominant": dominant,
        "D10": D10,
        "D30": D30,
        "D60": D60,
        "Cu": Cu,
        "Cc": Cc,
        "grading": grading,
        "name": f"{grading} {dominant}",
    }


def plot_grain_curve(d50=0.05, sigma=1.2, ax=None):
    own_axes = ax is None
    if own_axes:
        fig, ax = plt.subplots(figsize=(8.4, 5.4))

    d = np.logspace(-4, 2, 400)
    N = percent_finer(d, d50, sigma)

    bands = [
        (1e-4, CLAY_SILT, "#67c8e8", "Clay"),
        (CLAY_SILT, SILT_SAND, "#7ad07a", "Silt"),
        (SILT_SAND, SAND_GRAVEL, "#e89ad0", "Sand"),
        (SAND_GRAVEL, 1e2, "#f2e85a", "Gravel"),
    ]

    for x0, x1, colour, label in bands:
        ax.axvspan(x0, x1, color=colour, alpha=0.35)
        ax.text(np.sqrt(x0 * x1), 2, label, ha="center", weight="bold")

    ax.semilogx(d, N, "k-", lw=2.6)

    info = classify_grain_curve(d50, sigma)

    for p, Dp in [(10, info["D10"]), (30, info["D30"]), (60, info["D60"])]:
        if 1e-4 < Dp < 1e2:
            ax.plot([Dp, Dp], [0, p], "--", lw=1)
            ax.plot([1e-4, Dp], [p, p], "--", lw=1)
            ax.plot(Dp, p, "o")
            ax.text(Dp, p + 3, f"D{p}", fontsize=9, ha="center")

    f = info["fractions"]

    ax.set_xlim(1e-4, 1e2)
    ax.set_ylim(0, 100)
    ax.set_xlabel("grain diameter d (mm)")
    ax.set_ylabel("percentage finer N (%)")
    ax.grid(True, which="both", alpha=0.4)
    ax.set_title(
        f"d50 = {d50:.3g} mm    sigma = {sigma:.2f}\n"
        f"clay {f['clay']:.0f}%  silt {f['silt']:.0f}%  "
        f"sand {f['sand']:.0f}%  gravel {f['gravel']:.0f}%\n"
        f"Cu = {info['Cu']:.1f}, Cc = {info['Cc']:.2f} | {info['name']}"
    )

    if own_axes:
        plt.tight_layout()
        plt.show()

    return info


def curve_to_triangle(d50=0.05, sigma=1.2):
    info = classify_grain_curve(d50, sigma)
    f = info["fractions"]

    fines = f["clay"] + f["silt"] + f["sand"]

    if fines < 1e-6:
        print("This soil is essentially all gravel; the texture triangle does not apply.")
        return None

    sand = 100 * f["sand"] / fines
    clay = 100 * f["clay"] / fines

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    plot_grain_curve(d50=d50, sigma=sigma, ax=axes[0])
    texture_class = plot_texture_triangle(sand=sand, clay=clay, ax=axes[1])

    plt.tight_layout()
    plt.show()

    return texture_class

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

