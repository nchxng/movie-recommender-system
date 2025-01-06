# Movie Recommender System

This is a Python-based Movie Recommender System built using the `lightfm` library. It trains a hybrid content-based and collaborative filtering model to recommend movies to users based on their ratings using the WARP (Weighted Approximate-Rank Pairwise) loss function.

## Overview

The recommender system is trained on the MovieLens dataset, which contains ratings of movies from over 1700 users. The system predicts movies that a user is likely to enjoy based on their past ratings.

Once trained, the system can recommend movies for a set of users in the dataset.

## Features

- Hybrid model using collaborative filtering and content-based features
- Uses WARP loss function to optimize the recommendations
- Recommends movies with a rating of 4.0 or higher
- Prints out recommended movies for selected users

## Requirements

- Python 3.x
- `lightfm` library
- `numpy` library

You can install the required libraries with:

```bash
pip install lightfm numpy
