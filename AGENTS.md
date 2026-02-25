# Agent Context

## NetCDF inspection before coding
Use this quick check before changing data-loading, grid, or velocity scripts.

```bash
module load netcdf
ncdump -h xxx.nc
```

Verify variable names, dimension ordering, units, and missing-value conventions from the header output.
For ACCESS-OM2 periodic inputs, this helps avoid index-order mistakes (for example `month, z, y, x` vs `x, y, z`) before implementation.

## Code formatting
Use the Runic formatter for Julia code:
Run it after you have finished editing files with `runic --inplace .`
