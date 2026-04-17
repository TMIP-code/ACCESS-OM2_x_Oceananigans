"""
This script is for writing the locations of ACCESS-OM2 config files into
ACCESS-OM2_configs.yaml. To run it, simply use
    scripts/write_ACCESS-OM2_configs.sh
"""

import intake
import yaml
from pathlib import Path

# NOTE: `intake.cat.access_nri` returns an `AliasedDataframeCatalog`
# (access_nri_intake ≥ 1.6.0) whose `_normalise_value()` mangles list
# queries that contain regex strings by iterating over the string's
# characters. Bypass it with the underlying `DfFileCatalog` until the
# upstream fix lands in the conda env.
# See: https://github.com/ACCESS-NRI/access-nri-intake-catalog/issues/603
# catalogs = intake.cat.access_nri  # original; re-enable once #603 ships
catalogs = intake.cat.access_nri._cat

variable=[
    "tx_trans", "ty_trans",
    "mld",
    "eta_t",
    "^d[hz]t$",
]

all_om2 = set(catalogs.search(model="ACCESS-OM2.*").unique()["name"])
included = set(catalogs.search(
    model="ACCESS-OM2.*",
    variable=variable,
    require_all=True,
).unique()["name"])

excluded = sorted(all_om2 - included)
if excluded:
    base_variables = ["tx_trans", "ty_trans", "mld", "eta_t"]
    # Each subcatalog ("name") spans multiple rows (one per realm/file);
    # union their `variable` arrays to get the full set.
    name_to_vars = {
        name: {v for arr in group["variable"] for v in arr}
        for name, group in catalogs.df.groupby("name")
    }
    print(f"Excluded {len(excluded)} ACCESS-OM2 subcatalog(s) — missing variables:")
    for subcatalog in excluded:
        have = name_to_vars.get(subcatalog, set())
        missing = [v for v in base_variables if v not in have]
        if "dht" not in have and "dzt" not in have:
            missing.append("dht|dzt")
        print(f"  {subcatalog}: missing {missing}")
    print()

configs = {}
for subcatalog in sorted(included):
    cat = catalogs[subcatalog]
    p = Path(cat.search(variable="u").df.path.iloc[0])
    while p != p.parent:
        if (p / "config.yaml").exists():
            configs[subcatalog] = str(p / "config.yaml")
            break
        p = p.parent
    else:
        print(f"{subcatalog}: config.yaml not found")

output_file = Path(__file__).resolve().parent.parent / "ACCESS-OM2_configs.yaml"
with open(output_file, "w") as f:
    yaml.dump(configs, f, default_flow_style=False, sort_keys=True)

print(f"Wrote {len(configs)} config paths to {output_file}")
