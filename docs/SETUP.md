## FLARE environment setup

This project expects certain local directories for datasets and MRI processing. To keep paths portable between teammates, configuration is driven by environment variables instead of hard‑coded absolute paths.

### Required environment variables

- **`FLARE_DATA_ROOT`**: Base directory where raw datasets live for FLARE (CT, MRI, and other modalities).
  - Example layout: `${FLARE_DATA_ROOT}/<project>/data/...`

- **`FLARE_MRI_ROOT`**: Project root for the brain MRI pipeline.
  - This is the directory that contains the MRI code and its `data/` subfolders, matching what `ml/brain/mri/pre-processing/config.py` expects.
  - In `config.py`, `PROJECT_ROOT` is resolved as:
    - `FLARE_MRI_ROOT` if it is set in the environment.
    - The original author’s absolute path if `FLARE_MRI_ROOT` is not set (to keep existing behavior working).

You only need to set these variables to point to your own local dataset locations; the rest of the pipeline logic uses these roots.

### Windows PowerShell examples (temporary for current session)

From a PowerShell window opened at the FLARE repo root:

```powershell
# Point to your shared data directory
$env:FLARE_DATA_ROOT = "C:\data\flare"

# Point to your MRI project root (where the BraTS data and MRI scripts live)
$env:FLARE_MRI_ROOT = "C:\data\flare\brain_mri"

# Run your scripts in the same session so they see these variables
python .\ml\brain\mri\pre-processing\config.py
```

These settings last only for the current PowerShell session. To make them permanent, add them via Windows “Environment Variables” settings or your shell profile if desired.

