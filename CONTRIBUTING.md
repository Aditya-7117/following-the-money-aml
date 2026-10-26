# How we work on this repository

## Branches

Nobody commits to `main` directly, Aditya included. `main` only changes through a merged pull request.

Make one branch for each piece of work, named `<your-name>/<what>`, for example `yashasvi/eda-profile`. Start it from the latest `main`:

```
git checkout main
git pull
git checkout -b yashasvi/eda-profile
```

## Pull requests

1. Push your branch: `git push -u origin yashasvi/eda-profile`
2. On GitHub, open a pull request into `main`. Say what changed and how you checked it.
3. Your reviewer reads it and runs it:

   | Your work | Reviewed by |
   |---|---|
   | Aditya (data pipeline) | Sofia |
   | Yashasvi (statistics and EDA) | Priyanka |
   | Sofia (models) | Aditya |
   | Priyanka (evaluation) | Yashasvi |

4. Aditya does the final check and merges with **Squash and merge**.
5. After the merge, delete your branch, then `git checkout main` and `git pull` before starting the next piece.

## Before you open a pull request

- Change only your own module's files. Ask the owner before touching theirs.
- No data or generated tables. Never force-add a file that `.gitignore` excludes.
- No personal editor or tool config files.
- Normal commit messages. If a coding assistant adds a co-author line, remove it.
- Your code runs from a fresh pull of your branch.
