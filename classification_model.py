import os
import re
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.model_selection import KFold

def normalize(x: pd.Series, offset: float=0, scale: float=0):
    offset = x.min() if not offset else offset
    x = x - offset
    scale = 1. / x.max() if not scale else scale
    x = x * scale

    return x, offset, scale


def get_data():
    """
    Prepares the labeled data for the classification process

    Returns:
        tuple: (X, y)
            X is a numpy array with X.shape = (n_samples, n_features)
            y is a numpy array with y.shape = (n_samples, n_outcomes)
    """

    if os.path.exists('data.p'):
        # save some time when this has been previously computed
        with open('data.p', 'rb') as file:
            X, y = pickle.load(file)
        return X, y

    # read data from disk
    tracks_df = pd.read_csv('tracks.csv').set_index('id')
    artists_df = pd.read_csv('artists.csv')
    genres_data = pd.read_csv('data_by_genres_o.csv').set_index('genres')

    # these are the features available in the dataset
    feature_names = genres_data.columns.tolist()

    # extract feature data for tracks
    tracks_data = tracks_df.loc[:, feature_names]

    # parse columns with lists written in string format
    string2list = lambda x: [s.replace("'", '') for s in re.findall('\'.*?\'', x)]
    artists_df['genres'] = artists_df.genres.apply(string2list)
    tracks_df['id_artists'] = tracks_df.id_artists.apply(string2list)

    # we will limit the genres to the more popular ones in the USA
    # this list is taken from: https://www.statista.com/statistics/442354/music-genres-preferred-consumers-usa/
    genres_data = genres_data.loc[['rock', 'pop', 'country', 'hip hop', 'easy listening', 'jazz', 'blues', 'reggae', 'folk']]

    # process features to make sure they are properly normalized
    for feature in feature_names:
        if feature in ['duration_ms']:
            tracks_data[feature] = np.log(1 + tracks_data[feature])
            genres_data[feature] = np.log(1 + genres_data[feature])
        values = pd.concat([tracks_data[feature], genres_data[feature]])
        _, offset, scale = normalize(values)
        tracks_data[feature], _, _ = normalize(tracks_data[feature], offset=offset, scale=scale)
        genres_data[feature], _, _ = normalize(genres_data[feature], offset=offset, scale=scale)

    # for each genre, find artists associated with it. Then match songs with its artists genre
    labels = pd.DataFrame()
    for genre in genres_data.index:
        find_artists = lambda x: genre in x
        artists_with_genre = artists_df.loc[artists_df.genres.apply(find_artists), 'id'].tolist()
        match_genre = lambda x: any([art in artists_with_genre for art in x])
        labels[genre] = tracks_df.id_artists.apply(match_genre)
        print(f"{genre} has {labels[genre].sum()} tracks.")

    # we want to train on those rows for which we have at least one label
    mask = labels.any(axis=1)
    X, y = tracks_data.loc[mask].values, labels.loc[mask].values

    if not os.path.exists('data.p'):
        # save the computed data for future use
        with open('data.p', 'bw') as file:
            pickle.dump((X, y), file)

    return X, y


def get_model(n_categories, net_spec, learning_rate, steps_per_epoch, lr_decay_period=5, staircase=True):
    """
    Generates a sequential model with the provided parameters

    Args:
        n_categories: number of possible outcomes for the model to predict
        net_spec: a tuple with the numbers of nodes in each layer. For single layer must be like (N,)
        learning_rate: initial learning rate for the optimization process
        steps_per_epoch: total steps taken in a single epoch
        lr_decay_period: number of epochs before the learning rate becomes halved
        staircase: boolean. Whether to use a staircase pattern for the learning rate decay

    Returns:
        A corresponding keras.Sequetial model
    """
    

    model = keras.Sequential(
        [
            layers.Dense(net_spec[0], input_dim=13, activation='relu', name='hidden_layer_0'),
        ] + [
            layers.Dense(layer_size, activation='relu', name=f'hidden_layer_{i+1}') for i, layer_size in enumerate(net_spec[1:])
        ] + [
            layers.Dense(n_categories, activation='softmax', name='output_layer')
        ]
    )
    
    decay_steps = steps_per_epoch * lr_decay_period
    lr_schedule = ExponentialDecay(learning_rate, decay_steps=decay_steps, decay_rate=0.5, staircase=staircase)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

    return model


def run_cross_validation(learning_rate, net_spec, batch_size, n_splits=10, lr_decay_period=5,
                         epochs=10, verbose=0, random_state=None, staircase=True):
    """
    This is the main method in the pipeline. Here we retrieve the labeled training data, split it
    in k folds for cross-validation, and run k models' training process

    Args:
        learning_rate: float. initial learning rate for the optimization process
        net_spec: tuple. Contains numbers of nodes in each layer. For single layer must be like (N,)
        batch_size: int.
        n_splits: int. number of splits for the kfold cross-validation process
        lr_decay_period: int. Number of epochs before the learning rate becomes halved
        epochs: int. Number of epochs for which to train the model
        verbose: int. 0 for no verbosity, 1 prints progress for training process
        random_state: None or int. If None, a random seed will be used for kfold process
        staircase: boolean. Whether to use a staircase pattern for the learning rate decay

    Returns:
        A list of History objects. One for each kfold split
    """
    
    X, y = get_data()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # will store the resulting histories of the training at each split
    results = []
    fold_n = 0
    for train_index, test_index in kf.split(X, y):
        # for each split, run training and validation

        fold_n += 1
        if verbose:
            print(f"\nDoing fold {fold_n}.")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        training_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        validation_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

        n_train_samples, n_categories = y_train.shape
        steps_per_epoch = round(n_train_samples / batch_size)

        model = get_model(n_categories, net_spec, learning_rate, steps_per_epoch, lr_decay_period=lr_decay_period, staircase=staircase)
        history = model.fit(training_data, validation_data=validation_data, epochs=epochs, verbose=verbose)
        results.append(history)

    return results


def plot_figure(results, filename=None, show=False):
    """
    Processes the results list to plot figures for loss and accuracy metric, for each of the
    splits in the kfold
    Args:
        results: list. Contains History objects
        filename: string. If provided, the computed figure will be saved to this file
        show: boolean. Specify whether the figure should be displayed

    Returns:
        None
    """

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.ravel()
    for i, history in enumerate(results):
        fold_n = i + 1
        df = pd.DataFrame(history.history)

        for j, ax in enumerate(axes):
            ax.plot(df.index, df.iloc[:, j], label=f'fold {fold_n: 3d}')

    for j, ax in enumerate(axes):
        ax.set_title(f'{df.columns[j]}')
        if j > 1:
            ax.set_xlabel('epoch')
        ax.legend()

    if filename:
        plt.savefig(filename)

    if show:
        plt.show()


if __name__ == "__main__":

    learning_rate = 0.0001
    batch_size = 5
    net_spec = (10,)

    results = run_cross_validation(learning_rate, net_spec, batch_size, n_splits=5,
                                    lr_decay_period=5, epochs=60, verbose=1, random_state=None,
                                    staircase=False)
    plot_figure(results, show=True)
