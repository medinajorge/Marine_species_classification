# Marine Species Classification

Classify marine animal trajectories by species or breeding stage, using the InceptionTime neural network.

## Getting Started

Replace the `dataset.csv` and `metadata.csv` files in the `data` directory with your specific datasets.

### dataset.csv

The `dataset.csv` file contains time series data for each animal. The expected columns in this file are:

- `LATITUDE`: The latitude coordinate of the animal's position.
- `LONGITUDE`: The longitude coordinate of the animal's position.
- `DATE_TIME`: The date and time of the recorded position.
- `ID`: A unique identifier for the time series corresponding to each animal.

For computing the breeding stage, the file must also contain:

- `Stage`: The breeding stage of the animal.

#### Format of `dataset.csv`

```plaintext
LATITUDE, LONGITUDE, DATE_TIME, ID, Stage
-34.9285, 138.6007, 2020-01-01 12:00:00, A123, Chick-rearing
-33.8688, 151.2093, 2020-01-01 13:00:00, A123, Chick-rearing
...
```

### metadata.csv

The `metadata.csv` file contains static information about each animal. The required columns in this file are:

- `ID`: The unique identifier corresponding to each animal (matching `ID` in `dataset.csv`).
- `Species`: The species of the animal.
- `Taxa`: The broader taxonomic classification of the species.

#### Format of `metadata.csv`

```plaintext
ID, Species, Taxa
A123, Chinstrap penguin, Penguin
B456, Great White Shark, Fish
...
```

## Usage

Run the scripts under the `scripts` directory for time-demanding computations. Use the `species_clf.ipynb` and `breeding_stage_clf.ipynb` notebooks for interactive computation and plotting.

## Important Notes

- Ensure that the `ID` column is consistent across both `dataset.csv` and `metadata.csv` files.
- The date and time format in `DATE_TIME` should be consistent and in a standard format (e.g., YYYY-MM-DD HH:MM:SS).
- The breeding stage information is required for classifying animals based on their breeding cycle.

## Installation

Clone the conda environment

```bash
conda env create -f environment.yml
