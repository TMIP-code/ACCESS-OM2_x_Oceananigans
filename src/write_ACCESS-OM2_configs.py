"""
This script is for writing the locations of ACCESS-OM2 config files into
ACCESS-OM2_configs.yaml. To run it, simply use
    scripts/write_ACCESS-OM2_configs.sh
"""

import intake
import yaml
from pathlib import Path

catalogs = intake.cat.access_nri

variable=[
    "tx_trans", "ty_trans",
    "mld",
    "eta_t",
    "^d[hz]t$",
]

subcatalogs = catalogs.search(
    model="ACCESS-OM2.*",
    variable=variable,
    require_all=True
).unique()["name"]

configs = {}
for subcatalog in subcatalogs:
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
