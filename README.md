# Spotify dataset at Kaggle

In this project we build a classification model to predict song tracks' genre from audio features.

Link to the corresponding kaggle dataset: [https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks)

## Requirements

The code in the `classification_model.py` script requires python >= 3.6 with the following packages:
- tensorflow
- scikit-learn
- numpy
- pandas
- matplotlib

In addition to the packages, you will also need these three files in the root directory of this folder:
- tracks.csv
- artists.csv
- data_by_genres_o.csv

These files can be downloaded from the [dataset kaggle webpage](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks)

## Usage

You can run this script with this command:

```
$ python3 classification_model.py
```

This will take several minutes to complete.
