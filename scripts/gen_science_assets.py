"""Generate the PFT parameter table in docs/science.md from pft_defaults.toml.

Run directly or via `just docs` (which calls this before zensical build).
"""

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).parent.parent
TOML_PATH = REPO / "src/sgam/config/pft_defaults.toml"
DOCS_PATH = REPO / "docs/science.md"

PFTS = ["tree", "grass", "shrub", "crop"]

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

START = "<!-- PFT_TABLE_START -->"
END = "<!-- PFT_TABLE_END -->"


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def build_table(data: dict) -> str:
    header = "| Parameter | " + " | ".join(p.capitalize() for p in PFTS) + " |"
    sep = "|---|" + "---|" * len(PFTS)
    rows = []
    for key, label in ROWS:
        cells = " | ".join(_fmt(data[pft][key]) for pft in PFTS)
        rows.append(f"| {label} | {cells} |")
    return "\n".join([header, sep, *rows])


def main() -> None:
    with TOML_PATH.open("rb") as f:
        data = tomllib.load(f)

    table = build_table(data)
    replacement = f"{START}\n{table}\n{END}"

    text = DOCS_PATH.read_text()
    new_text, n = re.subn(
        rf"{re.escape(START)}.*?{re.escape(END)}",
        replacement,
        text,
        flags=re.DOTALL,
    )
    if n == 0:
        raise RuntimeError(f"Markers {START!r} / {END!r} not found in {DOCS_PATH}")

    DOCS_PATH.write_text(new_text)
    print(f"Updated PFT table in {DOCS_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
