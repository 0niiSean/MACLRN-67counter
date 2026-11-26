# Contributing

## Rules from the course

Every member must commit at least once. Commit messages must be meaningful. Work done by more than one person must say so in the commit message. A member who does not contribute can get a lower grade, so commit your own work under your own name.

## Ownership

| Role | Area | Folder | Name |
| --- | --- | --- | --- |
| A | Pose and counting in the app | `app/` (see the file split in `app/README.md`) | Angie Olonan |
| B | App flow and networking | `app/` (see the file split in `app/README.md`) | April Denise Besa |
| C | Server, storage, export, MLflow server | `server/` | Jamil So |
| D | Model training, evaluation, TFLite export | `training/` | Sean Andrei Olores |

Shared files (`docs/data_format.md`, `tools/`) change only through a pull request that every affected role approves.

## First time setup

```bash
git clone https://github.com/0niiSean/MACLRN-67counter.git
cd MACLRN-67counter
git config user.name "Your Name"          # the name that should appear on your commits
git config user.email "you@example.com"   # the email linked to your GitHub account
```

Python work (server, tools, training) uses a virtual environment per folder. Python 3.11 or 3.12 is the safest choice for PyTorch and TFLite.

```bash
cd server
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest                            # 3 tests should pass
```

## Start here, by role

You do not need to wait for anyone. Synthetic runs stand in for the app until real recordings exist (see `tools/README.md`).

| Role | First three things |
| --- | --- |
| A | Run the MediaPipe Pose Landmarker Android sample on the demo phone. Read the landmark order in `docs/data_format.md`. Port `tools/rule_counter.py` to Kotlin. |
| B | Create the Android Studio project with A and agree on the file split in `app/README.md`. Build start, countdown, and result screens with placeholder numbers. Run the server and open `http://<laptop-ip>:8000/docs` from the phone. |
| C | Run the server and tests. Fill it with `python tools/make_fake_runs.py --n 20 --post http://localhost:8000`. Write `server/export_runs.py` that produces the training export format in `docs/data_format.md`. |
| D | Prove the export path first: a dummy PyTorch model to `.tflite` (see `training/README.md`). Generate `python tools/make_fake_runs.py --n 60 --export data/fake/export` and write the data loader against it. |

## Branches

`main` always runs. Work on a branch named `feature/<short-name>` (for example `feature/rep-counter`), then open a pull request and have one teammate review it before merging.

```bash
git switch main && git pull
git switch -c feature/rep-counter
# ... work, commit ...
git push -u origin feature/rep-counter
# open the pull request on GitHub and request a reviewer
```

Keep pull requests small and merge often. Pull `main` into your branch every day so A and B do not drift apart in `app/`.

## Commit messages

Start with a short imperative summary, for example `Add leaderboard endpoint` or `Fix wrist landmark ordering`. Add a line of detail when the reason is not obvious.

For shared work, end the message with one line per co-author, using the email linked to their GitHub account so GitHub credits them.

```
Add landmark upload to app

Co-authored-by: Angie Olonan <email@example.com>
Co-authored-by: Sean Andrei Olores <email@example.com>
```

## What not to commit

Datasets, videos, trained models, and database files stay out of Git (see `.gitignore`). Share them through the team drive (videos go in the 67 Recordings folder) and describe how to obtain them in `data/README.md`. Run `git status` before every commit and check that nothing under `data/` or `models/` is listed.

## Blockers

Post in the group chat within one day of getting stuck. If a contract in `docs/data_format.md` needs to change, say so there first.
