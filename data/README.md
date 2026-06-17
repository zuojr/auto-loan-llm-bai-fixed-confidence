# Data Directory

Raw data is not tracked in git.

To reproduce the full pipeline, place the raw auto-loan CSV at:

```text
data/raw/CPRM_AutoLOan_OnlineAutoLoanData.csv
```

Then run:

```bash
python scripts/01_prepare_replay_environment.py
```

