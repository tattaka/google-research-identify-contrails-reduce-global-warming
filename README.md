# 6th place solution (2.5D models part)

## Environment
Use [Kaggle Docker v134](https://console.cloud.google.com/gcr/images/kaggle-gpu-images/GLOBAL/python).

## Usage
0. Place competition data in the `input` directory
1. Construct ash color dataset
    ```bash
    $ cd input
    $ python make_ash_dataset.py
    ```
2. Training
    ```bash
    $ cd src/exp043 && sh train.sh
    $ cd src/exp055 && sh train.sh
    ```
    `exp043` is a hard label and `exp055` is a soft label experiment.

## License
MIT