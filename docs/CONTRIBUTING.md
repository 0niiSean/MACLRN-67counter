# Contributing

## Rules from the course

Every member must commit at least once. Commit messages must be meaningful. Work done by more than one person must say so in the commit message.

## Branches

`main` always runs. Work on a branch named `feature/<short-name>` (for example `feature/rep-counter`), then open a pull request and have one teammate review it before merging.

## Commit messages

Start with a short imperative summary, for example `Add leaderboard endpoint` or `Fix wrist landmark ordering`. Add a line of detail when the reason is not obvious.

For shared work, end the message with one line per person.

```
Add landmark upload to app

Co-authored-by: Angie Olonan <email@example.com>
Co-authored-by: Sean Andrei Olores <email@example.com>
```

## What not to commit

Datasets, videos, trained models, and database files stay out of Git (see `.gitignore`). Share them through a team drive and describe how to obtain them in `data/README.md`.

## Ownership

| Area | Folder | Owner |
| --- | --- | --- |
| Android app | `app/` | TBD |
| Server and database | `server/` | TBD |
| Training and MLflow | `training/` | TBD |
| Data collection, report, slides | `docs/`, `data/` | TBD |
