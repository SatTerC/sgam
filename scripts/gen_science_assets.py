"""Generate the PFT parameter table and figures for docs/science.md.

Writes docs/_static/pft_table.md and docs/_static/images/*.png.
Run directly or via `just docs` (which calls this before the zensical build).
science.md references these outputs statically and is not modified by this script.
"""

import tomllib
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend — no display available in CI / doc generation
import matplotlib.pyplot as plt

from sgam.sgam import Sgam, PlantFunctionalType

REPO = Path(__file__).parent.parent
TOML_PATH = REPO / "src/sgam/config/pft_defaults.toml"
IMAGES_DIR = REPO / "docs/_static/images"
TABLE_PATH = REPO / "docs/_static/pft_table.md"

PFTS = ["tree", "grass", "shrub", "crop"]
PFT_COLORS = {
    "tree": "#2d6a4f",
    "grass": "#f77f00",
    "shrub": "#7209b7",
    "crop": "#d62828",
}

ROWS: list[tuple[str, str]] = [
    ("leaf_base_allocation", "Leaf base allocation"),
    ("stem_base_allocation", "Stem base allocation"),
    ("root_base_allocation", "Root base allocation"),
    ("leaf_turnover_rate", "Leaf turnover (wk⁻¹)"),
    ("stem_turnover_rate", "Stem turnover (wk⁻¹)"),
    ("root_turnover_rate", "Root turnover (wk⁻¹)"),
    ("lue_max", "LUE_max (gC MJ⁻¹)"),
    ("iwue_max", "iWUE_max (μmol mol⁻¹)"),
    ("vpd_threshold", "VPD threshold (Pa)"),
    ("vpd_sensitivity", "VPD sensitivity (Pa⁻¹)"),
    ("wilting_point", "Wilting point (m³ m⁻³)"),
    ("field_capacity", "Field capacity (m³ m⁻³)"),
]


def _fmt(value: float) -> str:
    if float(int(value)) == value:
        return str(int(value))
    return f"{value:g}"


def build_table(data: dict) -> str:
    """Build the Markdown PFT parameter table."""
    header = "| Parameter | " + " | ".join(p.capitalize() for p in PFTS) + " |"
    sep = "|---|" + "---|" * len(PFTS)
    rows = []
    for key, label in ROWS:
        cells = " | ".join(_fmt(data[pft][key]) for pft in PFTS)
        rows.append(f"| {label} | {cells} |")
    return "\n".join([header, sep, *rows])


def fig_cue(out_path: Path) -> None:
    """CUE response surface and 1D curves."""
    s_lue = np.linspace(0, 1, 101)
    s_iwue = np.linspace(0, 1, 101)
    S_LUE, S_IWUE = np.meshgrid(s_lue, s_iwue)
    cue = 0.2 + 0.5 * (S_LUE + S_IWUE) / 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    im = axes[0].contourf(S_LUE, S_IWUE, cue, levels=50, cmap="viridis")
    axes[0].set_xlabel("s_LUE")
    axes[0].set_ylabel("s_iWUE")
    axes[0].set_title("CUE response surface")
    plt.colorbar(im, ax=axes[0], label="CUE")

    axes[1].plot(
        s_lue,
        0.2 + 0.5 * s_lue / 2,
        label="s_iWUE = 0",
        color=PFT_COLORS["grass"],
    )
    axes[1].plot(
        s_lue,
        0.2 + 0.5 * (1 + s_lue) / 2,
        label="s_iWUE = 1",
        color=PFT_COLORS["tree"],
    )
    axes[1].plot(
        s_iwue,
        0.2 + 0.5 * s_iwue / 2,
        "--",
        label="s_LUE = 0",
        color=PFT_COLORS["shrub"],
    )
    axes[1].plot(
        s_iwue,
        0.2 + 0.5 * (1 + s_iwue) / 2,
        "--",
        label="s_LUE = 1",
        color=PFT_COLORS["crop"],
    )
    axes[1].set_xlabel("Score (0 → 1)")
    axes[1].set_ylabel("CUE")
    axes[1].set_title("CUE vs individual scores")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_drought(data: dict, out_path: Path) -> None:
    """Drought modifier curves per PFT."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sm = np.linspace(0, 0.5, 201)
    for pft in PFTS:
        wp = data[pft]["wilting_point"]
        fc = data[pft]["field_capacity"]
        f_sm = np.clip((sm - wp) / (fc - wp), 0, 1)
        axes[0].plot(sm, f_sm, label=pft.capitalize(), color=PFT_COLORS[pft])
        axes[0].axvspan(wp, fc, alpha=0.1, color=PFT_COLORS[pft])

    axes[0].set_xlabel("Soil moisture (m³ m⁻³)")
    axes[0].set_ylabel("f_sm")
    axes[0].set_title("Soil moisture stress (f_sm)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    vpd = np.linspace(0, 2500, 251)
    for pft in PFTS:
        vthr = data[pft]["vpd_threshold"]
        vgam = data[pft]["vpd_sensitivity"]
        f_vpd = np.exp(-vgam * np.maximum(vpd - vthr, 0))
        axes[1].plot(vpd, f_vpd, label=pft.capitalize(), color=PFT_COLORS[pft])
        axes[1].axvline(vthr, alpha=0.3, color=PFT_COLORS[pft], linestyle=":")

    axes[1].set_xlabel("VPD (Pa)")
    axes[1].set_ylabel("f_vpd")
    axes[1].set_title("VPD stress (f_vpd)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_allocation(data: dict, out_path: Path) -> None:
    """Dynamic allocation fractions over a synthetic year per PFT."""
    weeks = np.arange(1, 53)
    temp = 15 + 10 * np.sin(2 * np.pi * (weeks - 12) / 52)
    soil_moisture = 0.25
    vpd = 800.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for i, pft_name in enumerate(PFTS):
        pft_enum = PlantFunctionalType(pft_name)
        model = Sgam(pft_enum, use_dynamic_allocation=True)
        alloc_leaf, alloc_stem, alloc_root = model.compute_allocation_fractions(
            temp,
            np.full(52, soil_moisture),
            np.full(52, vpd),
            weeks.astype(float),
        )

        ax = axes[i]
        ax.stackplot(
            weeks,
            alloc_leaf,
            alloc_stem,
            alloc_root,
            labels=["Leaf", "Stem", "Root"],
            colors=["#80ed99", "#52b788", "#40916c"],
            alpha=0.8,
        )
        ax.set_title(pft_name.capitalize())
        ax.set_xlabel("Week")
        ax.set_ylabel("Allocation fraction")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_radar(data: dict, out_path: Path) -> None:
    """Radar chart comparing PFTs across normalised parameters."""
    params = [
        ("leaf_base_allocation", "Leaf alloc"),
        ("stem_base_allocation", "Stem alloc"),
        ("root_base_allocation", "Root alloc"),
        ("leaf_turnover_rate", "Leaf turnover"),
        ("lue_max", "LUE_max"),
        ("iwue_max", "iWUE_max"),
        ("vpd_threshold", "VPD threshold\n(inverted)"),
    ]

    raw = {}
    for pft in PFTS:
        raw[pft] = [data[pft][k] for k, _ in params]

    mins = np.min([raw[pft] for pft in PFTS], axis=0)
    maxs = np.max([raw[pft] for pft in PFTS], axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    norm = {}
    for pft in PFTS:
        vals = [(v - mi) / r for v, mi, r in zip(raw[pft], mins, ranges)]
        vals[6] = 1 - vals[6]
        norm[pft] = vals

    labels = [label for _, label in params]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    for pft in PFTS:
        norm[pft].append(norm[pft][0])
    angles.append(angles[0])
    labels.append(labels[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for pft in PFTS:
        ax.plot(
            angles,
            norm[pft],
            label=pft.capitalize(),
            color=PFT_COLORS[pft],
            linewidth=2,
        )
        ax.fill(angles, norm[pft], alpha=0.15, color=PFT_COLORS[pft])

    ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
    ax.set_ylim(0, 1)
    ax.set_title("PFT comparison (normalised)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Generate PFT table and figures for docs/science.md."""
    with TOML_PATH.open("rb") as f:
        data = tomllib.load(f)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    TABLE_PATH.write_text(build_table(data))
    fig_cue(IMAGES_DIR / "cue.png")
    fig_drought(data, IMAGES_DIR / "drought.png")
    fig_allocation(data, IMAGES_DIR / "allocation.png")
    fig_radar(data, IMAGES_DIR / "radar.png")

    print(f"Written table: {TABLE_PATH.relative_to(REPO)}")
    print(f"Written figures: {IMAGES_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
