#!/bin/bash
# Download MovieLens 25M dataset and extract movies.csv and ratings.csv to data/

set -e

DATA_DIR="$(dirname "$0")/data"
URL="https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ZIP_FILE="$DATA_DIR/ml-25m.zip"

mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_DIR/movies.csv" ] || [ ! -f "$DATA_DIR/ratings.csv" ]; then
  echo "Downloading MovieLens 25M dataset..."
  curl -L "$URL" -o "$ZIP_FILE"
  echo "Extracting movies.csv and ratings.csv..."
  unzip -j "$ZIP_FILE" "ml-25m/movies.csv" -d "$DATA_DIR"
  unzip -j "$ZIP_FILE" "ml-25m/ratings.csv" -d "$DATA_DIR"
  rm "$ZIP_FILE"
  echo "Done. Files are in $DATA_DIR."
else
  echo "movies.csv and ratings.csv already exist in $DATA_DIR."
fi
