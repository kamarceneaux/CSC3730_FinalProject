# Bike Rental Demand

CSC 3730 final project. We're using the UCI Bike Sharing dataset to model casual and registered riders separately and compare what drives demand for each group.

## Data

`data/` has the two CSVs from UCI:

- `day.csv` - daily totals (731 rows)
- `hour.csv` - hourly totals (17,379 rows)

`data/Readme.txt` explains the columns. `cnt` is always `casual + registered`.

Source: Fanaee-T, H. & Gama, J. (2013), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

## Setup

Use Python 3.12 if you can, since that's what CI runs.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

If you add a library, put it in `requirements.txt` with a pinned version so everyone's environment stays the same.

## Notebooks

Every push and pull request to `main` runs all the notebooks from top to bottom on GitHub Actions. If a cell errors, the check fails. Before you push:

- Restart the kernel and run all cells to make sure it works from a clean start.
- Load data with relative paths (`../data/hour.csv`), not paths on your own computer.
- Don't put spaces in notebook file names.

## Layout

```
data/        raw dataset
results/     saved figures and model outputs
.github/     CI workflow
```
