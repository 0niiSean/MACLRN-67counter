# Shared tools

Small scripts every role uses. They need only `numpy` (the tests also need the server requirements).

| File | What it does | Who uses it |
| --- | --- | --- |
| `rule_counter.py` | Reference rule based rep counter, the definition the Kotlin port must match | A (port and tune), D (rep timestamps for labels) |
| `make_fake_runs.py` | Synthetic runs in the exact upload and export formats | B and C (fill the server), D (build the loader) |
| `test_tools.py` | Checks that the tools still match the contract and the server accepts fake runs | CI |

Run these from the repo root.

```bash
python tools/make_fake_runs.py --n 5  --out data/fake/payloads              # payload JSON files
python tools/make_fake_runs.py --n 20 --post http://localhost:8000          # upload to the server
python tools/make_fake_runs.py --n 60 --export data/fake/export             # training export format
python tools/rule_counter.py data/fake/payloads/run_00001.json              # count one run
python -m pytest tools                                                      # run the checks
```

Fake runs are for plumbing and smoke tests only. Keep them under `data/fake/` and out of the real dataset and the report numbers.
