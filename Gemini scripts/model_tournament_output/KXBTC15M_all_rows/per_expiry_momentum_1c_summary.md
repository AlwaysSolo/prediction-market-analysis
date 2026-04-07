# KXBTC15M Tournament By Expiry Minute And Price Momentum

- Metric: `log_loss` on the held-out test split
- TTE buckets: rounded `tau_minutes`, from `14` down to `2`
- Momentum buckets: signed `price_momentum` in `1`-cent bins
- Models: same tournament lineup as `smarterpred.py`

## Winners By Joint Bucket

| TTE (min) | Price Momentum | Test Rows | Best Model | Best Log-Loss |
| --- | --- | ---: | --- | ---: |
| 14 | -92c | 1 | PySR (Symbolic Regression) | 0.01005 |
| 14 | -67c | 1 | XGBoost | 0.56552 |
| 14 | -59c | 1 | XGBoost | 0.37814 |
| 14 | -58c | 1 | Momentum-Adjusted Gaussian | 0.49430 |
| 14 | -56c | 1 | XGBoost | 0.25589 |
| 14 | -55c | 1 | CatBoost | 0.51661 |
| 14 | -53c | 1 | CatBoost | 0.47226 |
| 14 | -52c | 2 | LightGBM | 0.49910 |
| 14 | -50c | 2 | XGBoost | 0.30873 |
| 14 | -47c | 2 | Momentum-Adjusted Gaussian | 0.22346 |
| 14 | -46c | 1 | XGBoost | 0.41531 |
| 14 | -45c | 2 | Machine Learning (Logistic Regression) | 0.54973 |
| 14 | -44c | 3 | Standard Gaussian | 0.37851 |
| 14 | -43c | 1 | CatBoost | 0.52758 |
| 14 | -41c | 2 | Machine Learning (Logistic Regression) | 0.19537 |
| 14 | -40c | 5 | Machine Learning (Logistic Regression) | 0.61985 |
| 14 | -39c | 3 | CatBoost | 0.71125 |
| 14 | -38c | 6 | XGBoost | 0.71719 |
| 14 | -37c | 4 | XGBoost | 0.71586 |
| 14 | -36c | 1 | Momentum-Adjusted Gaussian | 0.35667 |
| 14 | -35c | 1 | Momentum-Adjusted Gaussian | 0.31471 |
| 14 | -34c | 7 | CatBoost | 0.66610 |
| 14 | -33c | 8 | PySR (Symbolic Regression) | 0.46092 |
| 14 | -32c | 4 | CatBoost | 0.69551 |
| 14 | -31c | 7 | Momentum-Adjusted Gaussian | 0.47747 |
| 14 | -30c | 5 | CatBoost | 0.65283 |
| 14 | -29c | 1 | CatBoost | 1.33471 |
| 14 | -28c | 8 | LightGBM | 0.73488 |
| 14 | -27c | 10 | XGBoost | 0.62085 |
| 14 | -26c | 9 | Momentum-Adjusted Gaussian | 0.56854 |
| 14 | -25c | 15 | Momentum-Adjusted Gaussian | 0.61900 |
| 14 | -24c | 10 | Machine Learning (Logistic Regression) | 0.45677 |
| 14 | -23c | 13 | Machine Learning (Logistic Regression) | 0.64348 |
| 14 | -22c | 16 | LightGBM | 0.51772 |
| 14 | -21c | 12 | Momentum-Adjusted Gaussian | 0.54852 |
| 14 | -20c | 23 | LightGBM | 0.65868 |
| 14 | -19c | 23 | LightGBM | 0.70131 |
| 14 | -18c | 25 | Machine Learning (Logistic Regression) | 0.58486 |
| 14 | -17c | 33 | LightGBM | 0.67356 |
| 14 | -16c | 38 | Momentum-Adjusted Gaussian | 0.59401 |
| 14 | -15c | 47 | LightGBM | 0.69179 |
| 14 | -14c | 51 | LightGBM | 0.67045 |
| 14 | -13c | 69 | Student-T df=3 | 0.65926 |
| 14 | -12c | 64 | Momentum-Adjusted Gaussian | 0.63355 |
| 14 | -11c | 95 | LightGBM | 0.65814 |
| 14 | -10c | 136 | Standard Gaussian | 0.63892 |
| 14 | -9c | 172 | LightGBM | 0.68101 |
| 14 | -8c | 205 | LightGBM | 0.64860 |
| 14 | -7c | 350 | LightGBM | 0.66023 |
| 14 | -6c | 495 | LightGBM | 0.67535 |
| 14 | -5c | 871 | LightGBM | 0.65239 |
| 14 | -4c | 1,956 | LightGBM | 0.66584 |
| 14 | -3c | 4,872 | LightGBM | 0.65609 |
| 14 | -2c | 11,578 | LightGBM | 0.66134 |
| 14 | -1c | 32,322 | LightGBM | 0.65572 |
| 14 | +0c | 93,381 | LightGBM | 0.65423 |
| 14 | +1c | 31,891 | LightGBM | 0.65720 |
| 14 | +2c | 11,573 | LightGBM | 0.66315 |
| 14 | +3c | 4,885 | LightGBM | 0.65718 |
| 14 | +4c | 1,973 | LightGBM | 0.65184 |
| 14 | +5c | 898 | LightGBM | 0.66679 |
| 14 | +6c | 472 | LightGBM | 0.66420 |
| 14 | +7c | 308 | LightGBM | 0.65572 |
| 14 | +8c | 220 | Machine Learning (Logistic Regression) | 0.60957 |
| 14 | +9c | 151 | LightGBM | 0.65983 |
| 14 | +10c | 131 | LightGBM | 0.64586 |
| 14 | +11c | 106 | LightGBM | 0.64069 |
| 14 | +12c | 65 | Momentum-Adjusted Gaussian | 0.63864 |
| 14 | +13c | 55 | CatBoost | 0.64329 |
| 14 | +14c | 52 | LightGBM | 0.61341 |
| 14 | +15c | 45 | XGBoost | 0.65872 |
| 14 | +16c | 26 | PySR (Symbolic Regression) | 0.65583 |
| 14 | +17c | 29 | LightGBM | 0.61193 |
| 14 | +18c | 23 | Machine Learning (Logistic Regression) | 0.60834 |
| 14 | +19c | 22 | CatBoost | 0.64732 |
| 14 | +20c | 24 | Machine Learning (Logistic Regression) | 0.61362 |
| 14 | +21c | 7 | LightGBM | 0.63224 |
| 14 | +22c | 15 | LightGBM | 0.60745 |
| 14 | +23c | 12 | LightGBM | 0.70545 |
| 14 | +24c | 6 | Machine Learning (Logistic Regression) | 0.52657 |
| 14 | +25c | 13 | Machine Learning (Logistic Regression) | 0.59480 |
| 14 | +26c | 14 | XGBoost | 0.67264 |
| 14 | +27c | 10 | Machine Learning (Logistic Regression) | 0.47788 |
| 14 | +28c | 7 | Machine Learning (Logistic Regression) | 0.43357 |
| 14 | +29c | 6 | Momentum-Adjusted Gaussian | 0.49574 |
| 14 | +30c | 7 | CatBoost | 0.74375 |
| 14 | +31c | 4 | Machine Learning (Logistic Regression) | 0.56087 |
| 14 | +32c | 2 | CatBoost | 0.67772 |
| 14 | +33c | 3 | XGBoost | 0.52835 |
| 14 | +34c | 5 | Momentum-Adjusted Gaussian | 0.52844 |
| 14 | +35c | 8 | LightGBM | 0.53624 |
| 14 | +36c | 1 | Momentum-Adjusted Gaussian | 0.18633 |
| 14 | +37c | 4 | PySR (Symbolic Regression) | 0.58733 |
| 14 | +38c | 4 | CatBoost | 0.55728 |
| 14 | +39c | 4 | LightGBM | 0.44117 |
| 14 | +40c | 1 | XGBoost | 0.41393 |
| 14 | +42c | 1 | LightGBM | 0.69650 |
| 14 | +43c | 3 | CatBoost | 0.53470 |
| 14 | +45c | 1 | XGBoost | 0.58809 |
| 14 | +47c | 1 | XGBoost | 0.56347 |
| 14 | +49c | 1 | Machine Learning (Logistic Regression) | 0.60084 |
| 14 | +55c | 2 | Machine Learning (Logistic Regression) | 0.54216 |
| 14 | +56c | 1 | Standard Gaussian | 0.01005 |
| 14 | +57c | 1 | Momentum-Adjusted Gaussian | 0.23572 |
| 14 | +59c | 2 | Momentum-Adjusted Gaussian | 0.24404 |
| 14 | +94c | 1 | CatBoost | 0.20741 |
| 13 | -53c | 1 | CatBoost | 0.37411 |
| 13 | -50c | 1 | Machine Learning (Logistic Regression) | 0.37643 |
| 13 | -49c | 1 | CatBoost | 0.44495 |
| 13 | -46c | 1 | CatBoost | 0.38744 |
| 13 | -45c | 2 | XGBoost | 0.51522 |
| 13 | -44c | 1 | XGBoost | 0.53031 |
| 13 | -43c | 1 | Machine Learning (Logistic Regression) | 0.27944 |
| 13 | -40c | 2 | Machine Learning (Logistic Regression) | 0.28300 |
| 13 | -35c | 1 | Machine Learning (Logistic Regression) | 0.36859 |
| 13 | -34c | 2 | Momentum-Adjusted Gaussian | 0.24797 |
| 13 | -33c | 3 | CatBoost | 0.48462 |
| 13 | -32c | 4 | CatBoost | 0.61544 |
| 13 | -31c | 1 | XGBoost | 0.56616 |
| 13 | -30c | 4 | Momentum-Adjusted Gaussian | 0.35915 |
| 13 | -29c | 2 | Momentum-Adjusted Gaussian | 0.25658 |
| 13 | -28c | 2 | Machine Learning (Logistic Regression) | 0.27317 |
| 13 | -27c | 2 | LightGBM | 0.52302 |
| 13 | -26c | 3 | Machine Learning (Logistic Regression) | 0.30367 |
| 13 | -25c | 6 | LightGBM | 0.70890 |
| 13 | -24c | 5 | XGBoost | 0.76273 |
| 13 | -23c | 4 | LightGBM | 0.84014 |
| 13 | -22c | 6 | CatBoost | 0.78451 |
| 13 | -21c | 6 | Machine Learning (Logistic Regression) | 0.56131 |
| 13 | -20c | 8 | LightGBM | 0.69082 |
| 13 | -19c | 10 | Momentum-Adjusted Gaussian | 0.54130 |
| 13 | -18c | 8 | Momentum-Adjusted Gaussian | 0.39293 |
| 13 | -17c | 16 | Machine Learning (Logistic Regression) | 0.48385 |
| 13 | -16c | 18 | LightGBM | 0.66433 |
| 13 | -15c | 22 | CatBoost | 0.66340 |
| 13 | -14c | 26 | Momentum-Adjusted Gaussian | 0.61841 |
| 13 | -13c | 35 | PySR (Symbolic Regression) | 0.67844 |
| 13 | -12c | 50 | Machine Learning (Logistic Regression) | 0.56369 |
| 13 | -11c | 43 | LightGBM | 0.61134 |
| 13 | -10c | 59 | LightGBM | 0.61098 |
| 13 | -9c | 103 | CatBoost | 0.68082 |
| 13 | -8c | 138 | LightGBM | 0.63027 |
| 13 | -7c | 215 | LightGBM | 0.63836 |
| 13 | -6c | 302 | Momentum-Adjusted Gaussian | 0.62854 |
| 13 | -5c | 604 | LightGBM | 0.62156 |
| 13 | -4c | 1,512 | LightGBM | 0.64877 |
| 13 | -3c | 4,319 | LightGBM | 0.64067 |
| 13 | -2c | 10,770 | LightGBM | 0.63693 |
| 13 | -1c | 30,000 | LightGBM | 0.63216 |
| 13 | +0c | 89,299 | LightGBM | 0.62451 |
| 13 | +1c | 30,067 | LightGBM | 0.62770 |
| 13 | +2c | 11,127 | LightGBM | 0.63432 |
| 13 | +3c | 4,336 | LightGBM | 0.62229 |
| 13 | +4c | 1,522 | LightGBM | 0.64240 |
| 13 | +5c | 591 | LightGBM | 0.62969 |
| 13 | +6c | 361 | Momentum-Adjusted Gaussian | 0.64148 |
| 13 | +7c | 199 | LightGBM | 0.65474 |
| 13 | +8c | 119 | Machine Learning (Logistic Regression) | 0.62261 |
| 13 | +9c | 93 | LightGBM | 0.64178 |
| 13 | +10c | 79 | PySR (Symbolic Regression) | 0.67447 |
| 13 | +11c | 52 | LightGBM | 0.73172 |
| 13 | +12c | 34 | Machine Learning (Logistic Regression) | 0.58756 |
| 13 | +13c | 33 | XGBoost | 0.78552 |
| 13 | +14c | 23 | CatBoost | 0.69162 |
| 13 | +15c | 24 | XGBoost | 0.56822 |
| 13 | +16c | 24 | PySR (Symbolic Regression) | 0.71249 |
| 13 | +17c | 17 | LightGBM | 0.65797 |
| 13 | +18c | 8 | LightGBM | 0.54313 |
| 13 | +19c | 6 | Momentum-Adjusted Gaussian | 0.54982 |
| 13 | +20c | 11 | Momentum-Adjusted Gaussian | 0.51207 |
| 13 | +21c | 5 | XGBoost | 0.58281 |
| 13 | +22c | 12 | Machine Learning (Logistic Regression) | 0.46551 |
| 13 | +23c | 11 | Machine Learning (Logistic Regression) | 0.41120 |
| 13 | +24c | 6 | Machine Learning (Logistic Regression) | 0.46960 |
| 13 | +25c | 1 | Momentum-Adjusted Gaussian | 0.40048 |
| 13 | +26c | 2 | Momentum-Adjusted Gaussian | 0.37836 |
| 13 | +27c | 5 | Momentum-Adjusted Gaussian | 0.52906 |
| 13 | +28c | 3 | Momentum-Adjusted Gaussian | 0.27155 |
| 13 | +29c | 5 | XGBoost | 0.56777 |
| 13 | +31c | 1 | XGBoost | 0.71084 |
| 13 | +32c | 2 | Machine Learning (Logistic Regression) | 0.31811 |
| 13 | +33c | 1 | Momentum-Adjusted Gaussian | 0.34249 |
| 13 | +34c | 1 | CatBoost | 0.42941 |
| 13 | +35c | 2 | Machine Learning (Logistic Regression) | 0.35056 |
| 13 | +36c | 2 | PySR (Symbolic Regression) | 0.42509 |
| 13 | +37c | 3 | CatBoost | 0.62509 |
| 13 | +41c | 1 | Momentum-Adjusted Gaussian | 0.38566 |
| 13 | +42c | 1 | CatBoost | 0.37542 |
| 13 | +44c | 2 | CatBoost | 0.41371 |
| 13 | +45c | 1 | CatBoost | 0.49333 |
| 13 | +49c | 1 | Machine Learning (Logistic Regression) | 0.40959 |
| 13 | +53c | 1 | Machine Learning (Logistic Regression) | 0.43322 |
| 13 | +56c | 1 | Momentum-Adjusted Gaussian | 0.07257 |
| 12 | -64c | 1 | LightGBM | 0.39325 |
| 12 | -63c | 1 | Momentum-Adjusted Gaussian | 0.09431 |
| 12 | -56c | 1 | CatBoost | 0.21417 |
| 12 | -53c | 1 | Momentum-Adjusted Gaussian | 0.16252 |
| 12 | -52c | 1 | LightGBM | 0.26285 |
| 12 | -49c | 1 | Momentum-Adjusted Gaussian | 0.15082 |
| 12 | -47c | 1 | CatBoost | 0.52982 |
| 12 | -42c | 1 | CatBoost | 0.41217 |
| 12 | -41c | 1 | LightGBM | 0.30207 |
| 12 | -40c | 1 | Machine Learning (Logistic Regression) | 0.53368 |
| 12 | -39c | 1 | LightGBM | 0.40833 |
| 12 | -38c | 2 | Machine Learning (Logistic Regression) | 0.53746 |
| 12 | -37c | 1 | Momentum-Adjusted Gaussian | 0.19845 |
| 12 | -36c | 3 | LightGBM | 0.55454 |
| 12 | -34c | 1 | CatBoost | 0.49128 |
| 12 | -33c | 1 | Momentum-Adjusted Gaussian | 0.28768 |
| 12 | -32c | 3 | LightGBM | 0.79098 |
| 12 | -30c | 5 | CatBoost | 0.67709 |
| 12 | -29c | 1 | Momentum-Adjusted Gaussian | 0.06188 |
| 12 | -28c | 5 | Machine Learning (Logistic Regression) | 0.48107 |
| 12 | -27c | 1 | Momentum-Adjusted Gaussian | 0.13926 |
| 12 | -26c | 2 | Machine Learning (Logistic Regression) | 0.34443 |
| 12 | -25c | 7 | Machine Learning (Logistic Regression) | 0.29115 |
| 12 | -24c | 7 | LightGBM | 0.72537 |
| 12 | -23c | 3 | Momentum-Adjusted Gaussian | 0.20092 |
| 12 | -22c | 8 | LightGBM | 0.70822 |
| 12 | -21c | 7 | PySR (Symbolic Regression) | 0.63169 |
| 12 | -20c | 7 | XGBoost | 0.70592 |
| 12 | -19c | 12 | Standard Gaussian | 0.52321 |
| 12 | -18c | 8 | LightGBM | 0.77574 |
| 12 | -17c | 14 | XGBoost | 0.69335 |
| 12 | -16c | 16 | LightGBM | 0.64467 |
| 12 | -15c | 16 | Machine Learning (Logistic Regression) | 0.63096 |
| 12 | -14c | 28 | Machine Learning (Logistic Regression) | 0.60363 |
| 12 | -13c | 29 | XGBoost | 0.61942 |
| 12 | -12c | 37 | Machine Learning (Logistic Regression) | 0.58739 |
| 12 | -11c | 53 | Momentum-Adjusted Gaussian | 0.51930 |
| 12 | -10c | 73 | XGBoost | 0.67002 |
| 12 | -9c | 93 | PySR (Symbolic Regression) | 0.64956 |
| 12 | -8c | 144 | LightGBM | 0.68210 |
| 12 | -7c | 196 | LightGBM | 0.59696 |
| 12 | -6c | 347 | LightGBM | 0.60689 |
| 12 | -5c | 578 | LightGBM | 0.63893 |
| 12 | -4c | 1,531 | LightGBM | 0.60830 |
| 12 | -3c | 4,548 | LightGBM | 0.62401 |
| 12 | -2c | 10,993 | LightGBM | 0.61360 |
| 12 | -1c | 29,155 | LightGBM | 0.61037 |
| 12 | +0c | 84,964 | LightGBM | 0.60448 |
| 12 | +1c | 29,282 | LightGBM | 0.60692 |
| 12 | +2c | 10,675 | LightGBM | 0.62042 |
| 12 | +3c | 4,507 | LightGBM | 0.62427 |
| 12 | +4c | 1,548 | LightGBM | 0.61944 |
| 12 | +5c | 607 | LightGBM | 0.61829 |
| 12 | +6c | 337 | LightGBM | 0.59565 |
| 12 | +7c | 182 | LightGBM | 0.63853 |
| 12 | +8c | 106 | LightGBM | 0.64469 |
| 12 | +9c | 76 | LightGBM | 0.65634 |
| 12 | +10c | 59 | PySR (Symbolic Regression) | 0.63465 |
| 12 | +11c | 51 | LightGBM | 0.65303 |
| 12 | +12c | 36 | LightGBM | 0.68785 |
| 12 | +13c | 23 | PySR (Symbolic Regression) | 0.62896 |
| 12 | +14c | 26 | CatBoost | 0.67530 |
| 12 | +15c | 20 | CatBoost | 0.69487 |
| 12 | +16c | 12 | Machine Learning (Logistic Regression) | 0.55871 |
| 12 | +17c | 15 | LightGBM | 0.56684 |
| 12 | +18c | 10 | Machine Learning (Logistic Regression) | 0.42880 |
| 12 | +19c | 10 | LightGBM | 0.55061 |
| 12 | +20c | 10 | PySR (Symbolic Regression) | 0.71940 |
| 12 | +21c | 5 | CatBoost | 0.93906 |
| 12 | +22c | 6 | PySR (Symbolic Regression) | 0.73578 |
| 12 | +23c | 5 | XGBoost | 0.91651 |
| 12 | +24c | 1 | Machine Learning (Logistic Regression) | 0.95502 |
| 12 | +25c | 4 | Momentum-Adjusted Gaussian | 0.40597 |
| 12 | +26c | 4 | LightGBM | 0.40031 |
| 12 | +27c | 6 | Machine Learning (Logistic Regression) | 0.33039 |
| 12 | +28c | 2 | CatBoost | 0.83811 |
| 12 | +29c | 5 | LightGBM | 0.57784 |
| 12 | +30c | 3 | LightGBM | 0.68560 |
| 12 | +31c | 2 | Machine Learning (Logistic Regression) | 0.54604 |
| 12 | +32c | 1 | Machine Learning (Logistic Regression) | 0.28437 |
| 12 | +33c | 2 | Momentum-Adjusted Gaussian | 0.12577 |
| 12 | +34c | 1 | Machine Learning (Logistic Regression) | 0.91237 |
| 12 | +36c | 1 | CatBoost | 0.36589 |
| 12 | +37c | 2 | Machine Learning (Logistic Regression) | 0.55366 |
| 12 | +38c | 1 | Machine Learning (Logistic Regression) | 0.49195 |
| 12 | +39c | 1 | CatBoost | 0.54328 |
| 12 | +40c | 1 | Machine Learning (Logistic Regression) | 0.21301 |
| 12 | +41c | 3 | CatBoost | 0.42510 |
| 12 | +42c | 1 | LightGBM | 0.38940 |
| 12 | +43c | 2 | CatBoost | 0.51251 |
| 12 | +44c | 2 | Momentum-Adjusted Gaussian | 0.32275 |
| 12 | +45c | 2 | CatBoost | 0.56259 |
| 12 | +46c | 1 | CatBoost | 0.32748 |
| 12 | +47c | 1 | CatBoost | 0.27168 |
| 12 | +57c | 1 | XGBoost | 0.15224 |
| 12 | +58c | 1 | Machine Learning (Logistic Regression) | 0.46954 |
| 12 | +62c | 1 | Machine Learning (Logistic Regression) | 0.42800 |
| 12 | +63c | 1 | XGBoost | 0.15538 |
| 11 | -55c | 3 | CatBoost | 0.38943 |
| 11 | -48c | 1 | Machine Learning (Logistic Regression) | 0.53116 |
| 11 | -41c | 1 | CatBoost | 0.57976 |
| 11 | -39c | 2 | CatBoost | 0.56063 |
| 11 | -38c | 1 | CatBoost | 0.49271 |
| 11 | -37c | 2 | Momentum-Adjusted Gaussian | 0.25960 |
| 11 | -36c | 1 | Machine Learning (Logistic Regression) | 0.36351 |
| 11 | -35c | 3 | XGBoost | 0.50012 |
| 11 | -34c | 3 | Machine Learning (Logistic Regression) | 0.29670 |
| 11 | -33c | 2 | Machine Learning (Logistic Regression) | 0.41926 |
| 11 | -32c | 2 | XGBoost | 0.61013 |
| 11 | -31c | 3 | CatBoost | 0.78285 |
| 11 | -30c | 2 | LightGBM | 0.60212 |
| 11 | -29c | 1 | Momentum-Adjusted Gaussian | 0.08338 |
| 11 | -28c | 3 | XGBoost | 0.63593 |
| 11 | -27c | 5 | LightGBM | 0.52838 |
| 11 | -26c | 3 | XGBoost | 0.56805 |
| 11 | -25c | 7 | XGBoost | 0.44685 |
| 11 | -24c | 2 | Machine Learning (Logistic Regression) | 0.22210 |
| 11 | -23c | 4 | Momentum-Adjusted Gaussian | 0.46479 |
| 11 | -22c | 7 | Machine Learning (Logistic Regression) | 0.51349 |
| 11 | -21c | 2 | Machine Learning (Logistic Regression) | 0.36571 |
| 11 | -20c | 5 | XGBoost | 0.76940 |
| 11 | -19c | 9 | LightGBM | 0.74764 |
| 11 | -18c | 10 | Machine Learning (Logistic Regression) | 0.37961 |
| 11 | -17c | 10 | LightGBM | 0.50789 |
| 11 | -16c | 13 | Machine Learning (Logistic Regression) | 0.49792 |
| 11 | -15c | 14 | Momentum-Adjusted Gaussian | 0.60279 |
| 11 | -14c | 18 | Machine Learning (Logistic Regression) | 0.51579 |
| 11 | -13c | 24 | Momentum-Adjusted Gaussian | 0.54318 |
| 11 | -12c | 33 | XGBoost | 0.73013 |
| 11 | -11c | 50 | CatBoost | 0.62067 |
| 11 | -10c | 57 | LightGBM | 0.62037 |
| 11 | -9c | 86 | LightGBM | 0.61372 |
| 11 | -8c | 123 | CatBoost | 0.61664 |
| 11 | -7c | 148 | CatBoost | 0.57987 |
| 11 | -6c | 349 | CatBoost | 0.58632 |
| 11 | -5c | 540 | LightGBM | 0.60616 |
| 11 | -4c | 1,369 | LightGBM | 0.60977 |
| 11 | -3c | 3,998 | LightGBM | 0.60962 |
| 11 | -2c | 9,594 | LightGBM | 0.59847 |
| 11 | -1c | 26,402 | LightGBM | 0.59365 |
| 11 | +0c | 77,590 | LightGBM | 0.58109 |
| 11 | +1c | 26,530 | LightGBM | 0.58874 |
| 11 | +2c | 9,817 | LightGBM | 0.60135 |
| 11 | +3c | 3,960 | LightGBM | 0.61284 |
| 11 | +4c | 1,306 | LightGBM | 0.60733 |
| 11 | +5c | 558 | LightGBM | 0.61010 |
| 11 | +6c | 282 | LightGBM | 0.62505 |
| 11 | +7c | 172 | Machine Learning (Logistic Regression) | 0.59153 |
| 11 | +8c | 124 | LightGBM | 0.62114 |
| 11 | +9c | 70 | Momentum-Adjusted Gaussian | 0.55764 |
| 11 | +10c | 67 | Momentum-Adjusted Gaussian | 0.57532 |
| 11 | +11c | 37 | PySR (Symbolic Regression) | 0.63528 |
| 11 | +12c | 34 | PySR (Symbolic Regression) | 0.69810 |
| 11 | +13c | 22 | LightGBM | 0.59255 |
| 11 | +14c | 21 | Machine Learning (Logistic Regression) | 0.50997 |
| 11 | +15c | 14 | Momentum-Adjusted Gaussian | 0.43463 |
| 11 | +16c | 14 | LightGBM | 0.44906 |
| 11 | +17c | 11 | Momentum-Adjusted Gaussian | 0.44069 |
| 11 | +18c | 10 | Momentum-Adjusted Gaussian | 0.48437 |
| 11 | +19c | 6 | Machine Learning (Logistic Regression) | 0.54049 |
| 11 | +20c | 5 | LightGBM | 0.45704 |
| 11 | +21c | 7 | LightGBM | 0.45808 |
| 11 | +22c | 6 | LightGBM | 0.83585 |
| 11 | +23c | 4 | CatBoost | 0.76837 |
| 11 | +24c | 4 | LightGBM | 0.60217 |
| 11 | +25c | 4 | LightGBM | 0.71703 |
| 11 | +26c | 5 | Machine Learning (Logistic Regression) | 0.48820 |
| 11 | +27c | 2 | Machine Learning (Logistic Regression) | 0.38850 |
| 11 | +28c | 4 | Momentum-Adjusted Gaussian | 0.14119 |
| 11 | +29c | 1 | CatBoost | 0.61039 |
| 11 | +30c | 3 | Momentum-Adjusted Gaussian | 0.03413 |
| 11 | +31c | 1 | CatBoost | 0.49385 |
| 11 | +32c | 3 | XGBoost | 0.47046 |
| 11 | +33c | 2 | LightGBM | 0.43644 |
| 11 | +34c | 3 | XGBoost | 0.45008 |
| 11 | +35c | 3 | XGBoost | 0.59556 |
| 11 | +36c | 3 | Machine Learning (Logistic Regression) | 0.53368 |
| 11 | +39c | 2 | XGBoost | 0.36454 |
| 11 | +40c | 1 | CatBoost | 0.24315 |
| 11 | +42c | 4 | CatBoost | 0.45709 |
| 11 | +45c | 1 | Momentum-Adjusted Gaussian | 0.19845 |
| 11 | +48c | 1 | LightGBM | 0.51115 |
| 11 | +52c | 1 | Standard Gaussian | 0.01005 |
| 11 | +55c | 1 | Momentum-Adjusted Gaussian | 0.07257 |
| 11 | +56c | 1 | CatBoost | 0.46353 |
| 10 | -62c | 1 | Machine Learning (Logistic Regression) | 0.42843 |
| 10 | -42c | 1 | Momentum-Adjusted Gaussian | 0.32850 |
| 10 | -40c | 1 | CatBoost | 0.86518 |
| 10 | -39c | 1 | CatBoost | 0.40172 |
| 10 | -38c | 1 | Momentum-Adjusted Gaussian | 0.09431 |
| 10 | -36c | 4 | CatBoost | 0.65098 |
| 10 | -34c | 2 | CatBoost | 0.51547 |
| 10 | -33c | 1 | CatBoost | 0.62057 |
| 10 | -32c | 1 | CatBoost | 0.53801 |
| 10 | -31c | 1 | XGBoost | 0.55002 |
| 10 | -30c | 1 | Momentum-Adjusted Gaussian | 0.23572 |
| 10 | -28c | 2 | CatBoost | 0.71819 |
| 10 | -27c | 3 | Momentum-Adjusted Gaussian | 0.37776 |
| 10 | -26c | 1 | Momentum-Adjusted Gaussian | 0.54473 |
| 10 | -25c | 6 | Machine Learning (Logistic Regression) | 0.67610 |
| 10 | -24c | 4 | LightGBM | 0.57092 |
| 10 | -23c | 4 | Momentum-Adjusted Gaussian | 0.52780 |
| 10 | -22c | 7 | Machine Learning (Logistic Regression) | 0.43018 |
| 10 | -21c | 6 | LightGBM | 0.50894 |
| 10 | -20c | 10 | CatBoost | 0.73326 |
| 10 | -19c | 10 | CatBoost | 0.50817 |
| 10 | -18c | 10 | LightGBM | 0.53110 |
| 10 | -17c | 13 | LightGBM | 0.59957 |
| 10 | -16c | 16 | LightGBM | 0.48268 |
| 10 | -15c | 18 | LightGBM | 0.49221 |
| 10 | -14c | 21 | Machine Learning (Logistic Regression) | 0.52231 |
| 10 | -13c | 23 | LightGBM | 0.55395 |
| 10 | -12c | 38 | Momentum-Adjusted Gaussian | 0.42768 |
| 10 | -11c | 45 | LightGBM | 0.69960 |
| 10 | -10c | 63 | LightGBM | 0.63596 |
| 10 | -9c | 84 | Machine Learning (Logistic Regression) | 0.58587 |
| 10 | -8c | 136 | LightGBM | 0.53511 |
| 10 | -7c | 176 | LightGBM | 0.61029 |
| 10 | -6c | 296 | LightGBM | 0.57195 |
| 10 | -5c | 557 | LightGBM | 0.61112 |
| 10 | -4c | 1,339 | LightGBM | 0.59082 |
| 10 | -3c | 4,011 | LightGBM | 0.59710 |
| 10 | -2c | 10,144 | LightGBM | 0.58003 |
| 10 | -1c | 27,429 | LightGBM | 0.57044 |
| 10 | +0c | 81,884 | LightGBM | 0.56203 |
| 10 | +1c | 27,595 | LightGBM | 0.56582 |
| 10 | +2c | 10,295 | LightGBM | 0.58673 |
| 10 | +3c | 4,065 | LightGBM | 0.60163 |
| 10 | +4c | 1,327 | LightGBM | 0.60788 |
| 10 | +5c | 537 | LightGBM | 0.61004 |
| 10 | +6c | 317 | LightGBM | 0.58030 |
| 10 | +7c | 200 | LightGBM | 0.56468 |
| 10 | +8c | 132 | Machine Learning (Logistic Regression) | 0.58200 |
| 10 | +9c | 89 | Machine Learning (Logistic Regression) | 0.54790 |
| 10 | +10c | 56 | Momentum-Adjusted Gaussian | 0.44817 |
| 10 | +11c | 48 | XGBoost | 0.63261 |
| 10 | +12c | 33 | Momentum-Adjusted Gaussian | 0.59844 |
| 10 | +13c | 20 | Momentum-Adjusted Gaussian | 0.54545 |
| 10 | +14c | 22 | Standard Gaussian | 0.49735 |
| 10 | +15c | 13 | LightGBM | 0.68231 |
| 10 | +16c | 16 | Momentum-Adjusted Gaussian | 0.53258 |
| 10 | +17c | 14 | Machine Learning (Logistic Regression) | 0.50534 |
| 10 | +18c | 15 | PySR (Symbolic Regression) | 0.74975 |
| 10 | +19c | 12 | Machine Learning (Logistic Regression) | 0.57176 |
| 10 | +20c | 10 | Momentum-Adjusted Gaussian | 0.58629 |
| 10 | +21c | 5 | Machine Learning (Logistic Regression) | 0.44740 |
| 10 | +22c | 3 | XGBoost | 0.72011 |
| 10 | +23c | 7 | Momentum-Adjusted Gaussian | 0.44748 |
| 10 | +24c | 4 | LightGBM | 0.70498 |
| 10 | +25c | 4 | Standard Gaussian | 0.52133 |
| 10 | +26c | 2 | Machine Learning (Logistic Regression) | 0.48887 |
| 10 | +27c | 4 | CatBoost | 0.79141 |
| 10 | +28c | 2 | Momentum-Adjusted Gaussian | 0.05179 |
| 10 | +29c | 2 | Machine Learning (Logistic Regression) | 0.37332 |
| 10 | +30c | 3 | CatBoost | 0.68209 |
| 10 | +32c | 4 | CatBoost | 0.64992 |
| 10 | +33c | 1 | Momentum-Adjusted Gaussian | 0.24846 |
| 10 | +34c | 4 | CatBoost | 0.66214 |
| 10 | +35c | 1 | CatBoost | 0.63234 |
| 10 | +36c | 1 | Momentum-Adjusted Gaussian | 0.31471 |
| 10 | +39c | 2 | CatBoost | 0.51112 |
| 10 | +43c | 1 | CatBoost | 0.32470 |
| 10 | +45c | 1 | Momentum-Adjusted Gaussian | 0.12783 |
| 10 | +59c | 1 | Standard Gaussian | 0.01005 |
| 9 | -68c | 1 | Momentum-Adjusted Gaussian | 0.17435 |
| 9 | -67c | 1 | Momentum-Adjusted Gaussian | 0.19845 |
| 9 | -61c | 1 | CatBoost | 0.33462 |
| 9 | -58c | 1 | CatBoost | 0.29971 |
| 9 | -54c | 3 | XGBoost | 0.24268 |
| 9 | -52c | 1 | CatBoost | 0.15641 |
| 9 | -51c | 1 | CatBoost | 0.23897 |
| 9 | -49c | 2 | CatBoost | 0.21078 |
| 9 | -48c | 1 | CatBoost | 0.30180 |
| 9 | -47c | 1 | XGBoost | 0.74312 |
| 9 | -46c | 2 | CatBoost | 0.37395 |
| 9 | -45c | 1 | CatBoost | 0.74923 |
| 9 | -42c | 1 | Momentum-Adjusted Gaussian | 0.35667 |
| 9 | -41c | 3 | CatBoost | 0.33064 |
| 9 | -40c | 2 | CatBoost | 0.61291 |
| 9 | -38c | 2 | CatBoost | 0.40337 |
| 9 | -36c | 1 | CatBoost | 0.55732 |
| 9 | -35c | 2 | Momentum-Adjusted Gaussian | 0.24797 |
| 9 | -34c | 1 | XGBoost | 0.32306 |
| 9 | -33c | 1 | CatBoost | 0.59258 |
| 9 | -32c | 1 | Momentum-Adjusted Gaussian | 0.17435 |
| 9 | -31c | 3 | Machine Learning (Logistic Regression) | 0.22824 |
| 9 | -30c | 2 | LightGBM | 0.70557 |
| 9 | -28c | 2 | Machine Learning (Logistic Regression) | 0.43667 |
| 9 | -27c | 3 | LightGBM | 0.59461 |
| 9 | -26c | 6 | LightGBM | 0.67642 |
| 9 | -25c | 3 | Machine Learning (Logistic Regression) | 0.50199 |
| 9 | -24c | 6 | CatBoost | 0.63663 |
| 9 | -22c | 5 | PySR (Symbolic Regression) | 0.58941 |
| 9 | -21c | 6 | LightGBM | 0.67407 |
| 9 | -20c | 5 | Momentum-Adjusted Gaussian | 0.50414 |
| 9 | -19c | 5 | PySR (Symbolic Regression) | 0.70103 |
| 9 | -18c | 12 | XGBoost | 0.70012 |
| 9 | -17c | 4 | Machine Learning (Logistic Regression) | 0.46411 |
| 9 | -16c | 12 | LightGBM | 0.61794 |
| 9 | -15c | 23 | LightGBM | 0.64695 |
| 9 | -14c | 22 | LightGBM | 0.57780 |
| 9 | -13c | 22 | Momentum-Adjusted Gaussian | 0.64540 |
| 9 | -12c | 24 | LightGBM | 0.53637 |
| 9 | -11c | 39 | Machine Learning (Logistic Regression) | 0.49279 |
| 9 | -10c | 72 | PySR (Symbolic Regression) | 0.62722 |
| 9 | -9c | 85 | LightGBM | 0.58651 |
| 9 | -8c | 110 | LightGBM | 0.52862 |
| 9 | -7c | 194 | Momentum-Adjusted Gaussian | 0.54423 |
| 9 | -6c | 231 | LightGBM | 0.58132 |
| 9 | -5c | 528 | LightGBM | 0.59544 |
| 9 | -4c | 1,289 | Momentum-Adjusted Gaussian | 0.57296 |
| 9 | -3c | 3,710 | LightGBM | 0.58939 |
| 9 | -2c | 9,173 | LightGBM | 0.55903 |
| 9 | -1c | 25,011 | LightGBM | 0.53973 |
| 9 | +0c | 74,596 | LightGBM | 0.52703 |
| 9 | +1c | 24,887 | LightGBM | 0.54096 |
| 9 | +2c | 9,465 | LightGBM | 0.55863 |
| 9 | +3c | 3,696 | LightGBM | 0.57574 |
| 9 | +4c | 1,184 | LightGBM | 0.56645 |
| 9 | +5c | 488 | LightGBM | 0.58005 |
| 9 | +6c | 255 | LightGBM | 0.57277 |
| 9 | +7c | 148 | Momentum-Adjusted Gaussian | 0.55943 |
| 9 | +8c | 124 | Machine Learning (Logistic Regression) | 0.52363 |
| 9 | +9c | 78 | LightGBM | 0.64672 |
| 9 | +10c | 58 | LightGBM | 0.59907 |
| 9 | +11c | 41 | LightGBM | 0.55643 |
| 9 | +12c | 36 | Momentum-Adjusted Gaussian | 0.56558 |
| 9 | +13c | 25 | Machine Learning (Logistic Regression) | 0.44074 |
| 9 | +14c | 16 | Machine Learning (Logistic Regression) | 0.55514 |
| 9 | +15c | 16 | LightGBM | 0.62482 |
| 9 | +16c | 12 | PySR (Symbolic Regression) | 0.69916 |
| 9 | +17c | 10 | CatBoost | 0.74724 |
| 9 | +18c | 7 | Momentum-Adjusted Gaussian | 0.48121 |
| 9 | +19c | 7 | Machine Learning (Logistic Regression) | 0.51839 |
| 9 | +20c | 8 | Machine Learning (Logistic Regression) | 0.79628 |
| 9 | +21c | 4 | Momentum-Adjusted Gaussian | 0.47449 |
| 9 | +22c | 4 | LightGBM | 0.56698 |
| 9 | +23c | 7 | LightGBM | 0.71530 |
| 9 | +24c | 3 | CatBoost | 0.58348 |
| 9 | +25c | 2 | Machine Learning (Logistic Regression) | 0.26772 |
| 9 | +26c | 2 | Momentum-Adjusted Gaussian | 0.29620 |
| 9 | +27c | 4 | LightGBM | 0.59613 |
| 9 | +28c | 5 | Momentum-Adjusted Gaussian | 0.39003 |
| 9 | +30c | 2 | CatBoost | 0.68533 |
| 9 | +31c | 2 | Machine Learning (Logistic Regression) | 0.41679 |
| 9 | +34c | 4 | XGBoost | 0.54236 |
| 9 | +35c | 3 | Momentum-Adjusted Gaussian | 0.45154 |
| 9 | +37c | 1 | Machine Learning (Logistic Regression) | 0.31846 |
| 9 | +38c | 1 | PySR (Symbolic Regression) | 0.85729 |
| 9 | +39c | 3 | XGBoost | 0.48202 |
| 9 | +41c | 3 | LightGBM | 0.39959 |
| 9 | +46c | 1 | CatBoost | 0.22740 |
| 9 | +48c | 1 | CatBoost | 0.35399 |
| 9 | +50c | 1 | CatBoost | 0.40751 |
| 9 | +52c | 1 | Momentum-Adjusted Gaussian | 0.18633 |
| 9 | +53c | 1 | CatBoost | 0.42496 |
| 9 | +54c | 1 | CatBoost | 0.33253 |
| 9 | +55c | 1 | Momentum-Adjusted Gaussian | 0.30111 |
| 9 | +56c | 1 | CatBoost | 0.44600 |
| 9 | +61c | 1 | CatBoost | 0.10531 |
| 9 | +90c | 1 | CatBoost | 0.11293 |
| 8 | -89c | 1 | CatBoost | 0.05515 |
| 8 | -80c | 1 | CatBoost | 0.10805 |
| 8 | -66c | 1 | Momentum-Adjusted Gaussian | 0.31471 |
| 8 | -62c | 1 | XGBoost | 0.15876 |
| 8 | -59c | 1 | XGBoost | 0.11547 |
| 8 | -47c | 1 | XGBoost | 0.22869 |
| 8 | -44c | 1 | XGBoost | 0.48267 |
| 8 | -41c | 2 | LightGBM | 0.71687 |
| 8 | -38c | 1 | Momentum-Adjusted Gaussian | 0.41552 |
| 8 | -34c | 1 | XGBoost | 0.46433 |
| 8 | -33c | 2 | XGBoost | 0.53905 |
| 8 | -32c | 1 | Momentum-Adjusted Gaussian | 0.35667 |
| 8 | -31c | 1 | XGBoost | 0.87074 |
| 8 | -29c | 2 | Machine Learning (Logistic Regression) | 0.34478 |
| 8 | -28c | 2 | CatBoost | 0.79042 |
| 8 | -27c | 1 | Momentum-Adjusted Gaussian | 0.63488 |
| 8 | -26c | 3 | LightGBM | 0.87214 |
| 8 | -25c | 4 | XGBoost | 0.56950 |
| 8 | -24c | 6 | XGBoost | 0.52297 |
| 8 | -23c | 2 | Momentum-Adjusted Gaussian | 0.71774 |
| 8 | -22c | 12 | XGBoost | 0.55710 |
| 8 | -21c | 6 | LightGBM | 0.46680 |
| 8 | -20c | 5 | Machine Learning (Logistic Regression) | 0.52912 |
| 8 | -19c | 6 | LightGBM | 0.52967 |
| 8 | -18c | 8 | Momentum-Adjusted Gaussian | 0.52258 |
| 8 | -17c | 12 | Momentum-Adjusted Gaussian | 0.34797 |
| 8 | -16c | 9 | LightGBM | 0.40865 |
| 8 | -15c | 8 | Machine Learning (Logistic Regression) | 0.50838 |
| 8 | -14c | 18 | PySR (Symbolic Regression) | 0.52175 |
| 8 | -13c | 17 | Momentum-Adjusted Gaussian | 0.55464 |
| 8 | -12c | 28 | Momentum-Adjusted Gaussian | 0.44018 |
| 8 | -11c | 42 | LightGBM | 0.46447 |
| 8 | -10c | 53 | LightGBM | 0.48949 |
| 8 | -9c | 91 | LightGBM | 0.59579 |
| 8 | -8c | 127 | LightGBM | 0.57853 |
| 8 | -7c | 174 | Student-T df=3 | 0.65293 |
| 8 | -6c | 266 | LightGBM | 0.57006 |
| 8 | -5c | 490 | LightGBM | 0.58338 |
| 8 | -4c | 1,186 | LightGBM | 0.54222 |
| 8 | -3c | 3,515 | LightGBM | 0.56164 |
| 8 | -2c | 9,306 | LightGBM | 0.53052 |
| 8 | -1c | 25,702 | LightGBM | 0.50914 |
| 8 | +0c | 76,509 | LightGBM | 0.49992 |
| 8 | +1c | 25,485 | LightGBM | 0.50771 |
| 8 | +2c | 9,292 | LightGBM | 0.52954 |
| 8 | +3c | 3,547 | LightGBM | 0.56505 |
| 8 | +4c | 1,142 | LightGBM | 0.55837 |
| 8 | +5c | 474 | XGBoost | 0.57801 |
| 8 | +6c | 290 | LightGBM | 0.54331 |
| 8 | +7c | 154 | LightGBM | 0.51976 |
| 8 | +8c | 107 | Machine Learning (Logistic Regression) | 0.47921 |
| 8 | +9c | 96 | LightGBM | 0.60167 |
| 8 | +10c | 63 | LightGBM | 0.56715 |
| 8 | +11c | 43 | LightGBM | 0.55729 |
| 8 | +12c | 32 | PySR (Symbolic Regression) | 0.66626 |
| 8 | +13c | 35 | Momentum-Adjusted Gaussian | 0.51166 |
| 8 | +14c | 23 | LightGBM | 0.63269 |
| 8 | +15c | 19 | CatBoost | 0.63526 |
| 8 | +16c | 14 | LightGBM | 0.76036 |
| 8 | +17c | 13 | CatBoost | 0.59762 |
| 8 | +18c | 16 | Machine Learning (Logistic Regression) | 0.53727 |
| 8 | +19c | 10 | PySR (Symbolic Regression) | 0.76462 |
| 8 | +20c | 2 | CatBoost | 0.91922 |
| 8 | +21c | 3 | Machine Learning (Logistic Regression) | 0.36114 |
| 8 | +22c | 6 | XGBoost | 0.87879 |
| 8 | +23c | 2 | XGBoost | 0.72467 |
| 8 | +24c | 1 | PySR (Symbolic Regression) | 0.77523 |
| 8 | +25c | 7 | Machine Learning (Logistic Regression) | 0.50059 |
| 8 | +26c | 3 | Momentum-Adjusted Gaussian | 0.51172 |
| 8 | +27c | 5 | PySR (Symbolic Regression) | 0.95078 |
| 8 | +28c | 1 | Momentum-Adjusted Gaussian | 0.10536 |
| 8 | +29c | 2 | Machine Learning (Logistic Regression) | 0.47506 |
| 8 | +30c | 1 | Machine Learning (Logistic Regression) | 0.30879 |
| 8 | +31c | 1 | Momentum-Adjusted Gaussian | 0.31471 |
| 8 | +33c | 1 | Momentum-Adjusted Gaussian | 0.71335 |
| 8 | +34c | 1 | CatBoost | 1.02687 |
| 8 | +35c | 1 | Momentum-Adjusted Gaussian | 0.07257 |
| 8 | +36c | 2 | XGBoost | 0.49872 |
| 8 | +37c | 1 | Momentum-Adjusted Gaussian | 0.09431 |
| 8 | +40c | 2 | CatBoost | 0.62489 |
| 8 | +41c | 1 | XGBoost | 0.39888 |
| 8 | +42c | 2 | CatBoost | 0.77148 |
| 8 | +43c | 1 | Machine Learning (Logistic Regression) | 0.85848 |
| 8 | +44c | 1 | CatBoost | 0.54876 |
| 8 | +49c | 1 | CatBoost | 0.28574 |
| 8 | +53c | 1 | CatBoost | 0.35556 |
| 8 | +59c | 1 | CatBoost | 0.26448 |
| 8 | +66c | 1 | XGBoost | 0.11215 |
| 8 | +67c | 1 | CatBoost | 0.15337 |
| 8 | +72c | 1 | XGBoost | 0.07194 |
| 8 | +73c | 1 | Momentum-Adjusted Gaussian | 0.05129 |
| 7 | -55c | 1 | XGBoost | 0.30555 |
| 7 | -47c | 1 | LightGBM | 0.57108 |
| 7 | -45c | 1 | CatBoost | 0.54145 |
| 7 | -42c | 1 | XGBoost | 0.41233 |
| 7 | -41c | 1 | XGBoost | 0.49084 |
| 7 | -39c | 1 | Machine Learning (Logistic Regression) | 0.61116 |
| 7 | -38c | 1 | XGBoost | 0.86519 |
| 7 | -37c | 1 | CatBoost | 0.70133 |
| 7 | -36c | 2 | CatBoost | 0.53785 |
| 7 | -31c | 3 | Momentum-Adjusted Gaussian | 0.22171 |
| 7 | -29c | 5 | Machine Learning (Logistic Regression) | 0.24708 |
| 7 | -28c | 1 | Momentum-Adjusted Gaussian | 0.37106 |
| 7 | -27c | 4 | Momentum-Adjusted Gaussian | 0.49569 |
| 7 | -26c | 2 | XGBoost | 0.64254 |
| 7 | -25c | 4 | Momentum-Adjusted Gaussian | 0.41774 |
| 7 | -24c | 2 | Machine Learning (Logistic Regression) | 0.53620 |
| 7 | -23c | 3 | Momentum-Adjusted Gaussian | 0.19243 |
| 7 | -22c | 1 | Momentum-Adjusted Gaussian | 0.06188 |
| 7 | -21c | 7 | LightGBM | 0.59847 |
| 7 | -20c | 7 | Machine Learning (Logistic Regression) | 0.22575 |
| 7 | -19c | 7 | PySR (Symbolic Regression) | 0.64564 |
| 7 | -18c | 12 | PySR (Symbolic Regression) | 0.67566 |
| 7 | -17c | 12 | Momentum-Adjusted Gaussian | 0.47135 |
| 7 | -16c | 13 | Machine Learning (Logistic Regression) | 0.46092 |
| 7 | -15c | 21 | LightGBM | 0.55618 |
| 7 | -14c | 15 | Standard Gaussian | 0.49192 |
| 7 | -13c | 20 | Momentum-Adjusted Gaussian | 0.58746 |
| 7 | -12c | 30 | LightGBM | 0.50442 |
| 7 | -11c | 33 | Machine Learning (Logistic Regression) | 0.53087 |
| 7 | -10c | 54 | PySR (Symbolic Regression) | 0.72301 |
| 7 | -9c | 72 | Machine Learning (Logistic Regression) | 0.48440 |
| 7 | -8c | 117 | Momentum-Adjusted Gaussian | 0.57399 |
| 7 | -7c | 180 | PySR (Symbolic Regression) | 0.57052 |
| 7 | -6c | 270 | Machine Learning (Logistic Regression) | 0.50890 |
| 7 | -5c | 413 | XGBoost | 0.55628 |
| 7 | -4c | 1,093 | LightGBM | 0.54779 |
| 7 | -3c | 3,230 | LightGBM | 0.54637 |
| 7 | -2c | 8,752 | LightGBM | 0.51031 |
| 7 | -1c | 23,901 | LightGBM | 0.49092 |
| 7 | +0c | 73,727 | LightGBM | 0.47603 |
| 7 | +1c | 24,388 | LightGBM | 0.48658 |
| 7 | +2c | 8,799 | LightGBM | 0.51640 |
| 7 | +3c | 3,352 | LightGBM | 0.55909 |
| 7 | +4c | 1,116 | LightGBM | 0.57214 |
| 7 | +5c | 447 | LightGBM | 0.54461 |
| 7 | +6c | 251 | Standard Gaussian | 0.54567 |
| 7 | +7c | 181 | Machine Learning (Logistic Regression) | 0.58261 |
| 7 | +8c | 120 | LightGBM | 0.52514 |
| 7 | +9c | 67 | LightGBM | 0.51815 |
| 7 | +10c | 61 | CatBoost | 0.62934 |
| 7 | +11c | 48 | Machine Learning (Logistic Regression) | 0.53703 |
| 7 | +12c | 36 | LightGBM | 0.55453 |
| 7 | +13c | 31 | CatBoost | 0.52706 |
| 7 | +14c | 14 | LightGBM | 0.68783 |
| 7 | +15c | 16 | Momentum-Adjusted Gaussian | 0.47019 |
| 7 | +16c | 11 | Momentum-Adjusted Gaussian | 0.28653 |
| 7 | +17c | 7 | CatBoost | 0.51406 |
| 7 | +18c | 12 | LightGBM | 0.72819 |
| 7 | +19c | 8 | Machine Learning (Logistic Regression) | 0.53915 |
| 7 | +20c | 2 | Momentum-Adjusted Gaussian | 0.18336 |
| 7 | +21c | 8 | LightGBM | 0.46745 |
| 7 | +22c | 4 | LightGBM | 0.67746 |
| 7 | +23c | 4 | Momentum-Adjusted Gaussian | 0.66095 |
| 7 | +24c | 9 | CatBoost | 0.54560 |
| 7 | +25c | 1 | Machine Learning (Logistic Regression) | 0.27278 |
| 7 | +26c | 1 | Standard Gaussian | 0.01005 |
| 7 | +27c | 3 | Machine Learning (Logistic Regression) | 0.28601 |
| 7 | +28c | 7 | PySR (Symbolic Regression) | 0.66541 |
| 7 | +29c | 3 | Machine Learning (Logistic Regression) | 0.17725 |
| 7 | +30c | 1 | Momentum-Adjusted Gaussian | 0.69315 |
| 7 | +31c | 1 | Machine Learning (Logistic Regression) | 0.23410 |
| 7 | +34c | 1 | Momentum-Adjusted Gaussian | 0.30111 |
| 7 | +37c | 1 | Machine Learning (Logistic Regression) | 0.44985 |
| 7 | +39c | 1 | LightGBM | 0.72501 |
| 7 | +42c | 1 | Machine Learning (Logistic Regression) | 0.35922 |
| 7 | +46c | 1 | CatBoost | 0.45404 |
| 7 | +47c | 1 | Machine Learning (Logistic Regression) | 0.34209 |
| 7 | +58c | 1 | CatBoost | 0.17875 |
| 7 | +61c | 1 | XGBoost | 0.14322 |
| 7 | +62c | 1 | XGBoost | 0.16499 |
| 7 | +64c | 1 | XGBoost | 0.13093 |
| 6 | -79c | 1 | CatBoost | 0.40885 |
| 6 | -56c | 1 | Momentum-Adjusted Gaussian | 0.30111 |
| 6 | -53c | 2 | Momentum-Adjusted Gaussian | 0.39373 |
| 6 | -47c | 1 | CatBoost | 0.31426 |
| 6 | -42c | 3 | Momentum-Adjusted Gaussian | 0.29781 |
| 6 | -40c | 1 | XGBoost | 0.30631 |
| 6 | -39c | 1 | Machine Learning (Logistic Regression) | 0.43716 |
| 6 | -38c | 2 | Machine Learning (Logistic Regression) | 0.35585 |
| 6 | -37c | 1 | Machine Learning (Logistic Regression) | 0.46877 |
| 6 | -32c | 2 | Machine Learning (Logistic Regression) | 0.28488 |
| 6 | -31c | 1 | Momentum-Adjusted Gaussian | 0.09431 |
| 6 | -30c | 3 | Machine Learning (Logistic Regression) | 0.23876 |
| 6 | -29c | 3 | Momentum-Adjusted Gaussian | 0.28170 |
| 6 | -28c | 1 | Standard Gaussian | 0.01005 |
| 6 | -27c | 4 | Momentum-Adjusted Gaussian | 0.23251 |
| 6 | -26c | 4 | XGBoost | 0.63222 |
| 6 | -25c | 8 | Momentum-Adjusted Gaussian | 0.39130 |
| 6 | -24c | 3 | Machine Learning (Logistic Regression) | 0.12692 |
| 6 | -23c | 7 | Machine Learning (Logistic Regression) | 0.28788 |
| 6 | -22c | 6 | LightGBM | 0.58455 |
| 6 | -21c | 7 | Momentum-Adjusted Gaussian | 0.51772 |
| 6 | -20c | 12 | Momentum-Adjusted Gaussian | 0.46438 |
| 6 | -19c | 8 | PySR (Symbolic Regression) | 0.57097 |
| 6 | -18c | 7 | LightGBM | 0.95340 |
| 6 | -17c | 12 | LightGBM | 0.52034 |
| 6 | -16c | 13 | XGBoost | 0.61053 |
| 6 | -15c | 28 | Momentum-Adjusted Gaussian | 0.46895 |
| 6 | -14c | 23 | Machine Learning (Logistic Regression) | 0.42559 |
| 6 | -13c | 22 | LightGBM | 0.67502 |
| 6 | -12c | 30 | LightGBM | 0.57987 |
| 6 | -11c | 34 | PySR (Symbolic Regression) | 0.57456 |
| 6 | -10c | 43 | Machine Learning (Logistic Regression) | 0.52034 |
| 6 | -9c | 73 | LightGBM | 0.50411 |
| 6 | -8c | 111 | Machine Learning (Logistic Regression) | 0.47888 |
| 6 | -7c | 153 | LightGBM | 0.55184 |
| 6 | -6c | 256 | LightGBM | 0.58369 |
| 6 | -5c | 422 | LightGBM | 0.54245 |
| 6 | -4c | 1,067 | LightGBM | 0.52892 |
| 6 | -3c | 3,274 | LightGBM | 0.55066 |
| 6 | -2c | 8,691 | LightGBM | 0.51130 |
| 6 | -1c | 24,564 | LightGBM | 0.46114 |
| 6 | +0c | 78,387 | LightGBM | 0.43773 |
| 6 | +1c | 24,658 | LightGBM | 0.46312 |
| 6 | +2c | 8,823 | LightGBM | 0.49854 |
| 6 | +3c | 3,310 | LightGBM | 0.53959 |
| 6 | +4c | 1,067 | LightGBM | 0.56063 |
| 6 | +5c | 456 | LightGBM | 0.55520 |
| 6 | +6c | 273 | LightGBM | 0.54811 |
| 6 | +7c | 142 | LightGBM | 0.55773 |
| 6 | +8c | 127 | LightGBM | 0.51306 |
| 6 | +9c | 82 | LightGBM | 0.51194 |
| 6 | +10c | 63 | LightGBM | 0.51301 |
| 6 | +11c | 41 | PySR (Symbolic Regression) | 0.58815 |
| 6 | +12c | 26 | Momentum-Adjusted Gaussian | 0.51983 |
| 6 | +13c | 18 | Machine Learning (Logistic Regression) | 0.42252 |
| 6 | +14c | 17 | Machine Learning (Logistic Regression) | 0.47546 |
| 6 | +15c | 16 | LightGBM | 0.59444 |
| 6 | +16c | 7 | LightGBM | 0.71593 |
| 6 | +17c | 7 | XGBoost | 0.61733 |
| 6 | +18c | 8 | XGBoost | 0.67300 |
| 6 | +19c | 8 | LightGBM | 0.61699 |
| 6 | +20c | 10 | Momentum-Adjusted Gaussian | 0.43395 |
| 6 | +21c | 9 | Machine Learning (Logistic Regression) | 0.52179 |
| 6 | +22c | 2 | LightGBM | 0.33599 |
| 6 | +23c | 5 | Machine Learning (Logistic Regression) | 0.36258 |
| 6 | +24c | 2 | PySR (Symbolic Regression) | 0.90393 |
| 6 | +25c | 4 | Machine Learning (Logistic Regression) | 0.37157 |
| 6 | +26c | 1 | Machine Learning (Logistic Regression) | 1.16773 |
| 6 | +27c | 3 | LightGBM | 0.22884 |
| 6 | +29c | 4 | Momentum-Adjusted Gaussian | 0.40252 |
| 6 | +30c | 2 | Momentum-Adjusted Gaussian | 0.66364 |
| 6 | +32c | 1 | CatBoost | 0.15129 |
| 6 | +33c | 4 | Machine Learning (Logistic Regression) | 0.62723 |
| 6 | +34c | 1 | Momentum-Adjusted Gaussian | 0.13926 |
| 6 | +35c | 1 | XGBoost | 1.10251 |
| 6 | +36c | 3 | Momentum-Adjusted Gaussian | 0.29317 |
| 6 | +38c | 2 | Machine Learning (Logistic Regression) | 0.49070 |
| 6 | +48c | 1 | Machine Learning (Logistic Regression) | 0.56038 |
| 5 | -65c | 1 | Momentum-Adjusted Gaussian | 0.24846 |
| 5 | -63c | 1 | Momentum-Adjusted Gaussian | 0.28768 |
| 5 | -49c | 1 | LightGBM | 0.47064 |
| 5 | -48c | 1 | Momentum-Adjusted Gaussian | 0.08338 |
| 5 | -46c | 1 | LightGBM | 0.49224 |
| 5 | -44c | 1 | LightGBM | 0.50623 |
| 5 | -43c | 1 | Momentum-Adjusted Gaussian | 0.28768 |
| 5 | -40c | 2 | LightGBM | 0.65084 |
| 5 | -38c | 2 | CatBoost | 0.69689 |
| 5 | -37c | 1 | LightGBM | 0.70900 |
| 5 | -36c | 2 | Machine Learning (Logistic Regression) | 0.28859 |
| 5 | -35c | 2 | LightGBM | 0.53604 |
| 5 | -34c | 1 | LightGBM | 0.64424 |
| 5 | -31c | 1 | Momentum-Adjusted Gaussian | 0.10536 |
| 5 | -30c | 2 | LightGBM | 0.69714 |
| 5 | -29c | 3 | LightGBM | 0.54031 |
| 5 | -28c | 1 | Momentum-Adjusted Gaussian | 0.06188 |
| 5 | -27c | 4 | CatBoost | 0.68545 |
| 5 | -26c | 2 | Machine Learning (Logistic Regression) | 0.25087 |
| 5 | -25c | 4 | PySR (Symbolic Regression) | 0.70140 |
| 5 | -24c | 7 | Momentum-Adjusted Gaussian | 0.35812 |
| 5 | -23c | 5 | Machine Learning (Logistic Regression) | 0.25199 |
| 5 | -22c | 5 | Machine Learning (Logistic Regression) | 0.33120 |
| 5 | -21c | 6 | LightGBM | 0.78099 |
| 5 | -20c | 6 | PySR (Symbolic Regression) | 0.87994 |
| 5 | -19c | 15 | XGBoost | 0.67021 |
| 5 | -18c | 8 | LightGBM | 0.78704 |
| 5 | -17c | 18 | LightGBM | 0.61641 |
| 5 | -16c | 17 | XGBoost | 0.63979 |
| 5 | -15c | 23 | Standard Gaussian | 0.50839 |
| 5 | -14c | 28 | Machine Learning (Logistic Regression) | 0.59214 |
| 5 | -13c | 18 | LightGBM | 0.58354 |
| 5 | -12c | 42 | LightGBM | 0.52547 |
| 5 | -11c | 46 | PySR (Symbolic Regression) | 0.62300 |
| 5 | -10c | 58 | PySR (Symbolic Regression) | 0.56749 |
| 5 | -9c | 104 | Student-T df=3 | 0.58647 |
| 5 | -8c | 133 | Standard Gaussian | 0.51182 |
| 5 | -7c | 182 | PySR (Symbolic Regression) | 0.55224 |
| 5 | -6c | 287 | LightGBM | 0.55219 |
| 5 | -5c | 504 | LightGBM | 0.50786 |
| 5 | -4c | 1,121 | LightGBM | 0.52302 |
| 5 | -3c | 3,285 | LightGBM | 0.52611 |
| 5 | -2c | 8,908 | LightGBM | 0.47498 |
| 5 | -1c | 25,925 | LightGBM | 0.42901 |
| 5 | +0c | 86,211 | LightGBM | 0.39912 |
| 5 | +1c | 26,484 | LightGBM | 0.42796 |
| 5 | +2c | 8,927 | LightGBM | 0.47197 |
| 5 | +3c | 3,235 | LightGBM | 0.53074 |
| 5 | +4c | 1,088 | LightGBM | 0.52214 |
| 5 | +5c | 479 | LightGBM | 0.54657 |
| 5 | +6c | 323 | LightGBM | 0.49991 |
| 5 | +7c | 172 | LightGBM | 0.49955 |
| 5 | +8c | 118 | LightGBM | 0.53632 |
| 5 | +9c | 86 | LightGBM | 0.53311 |
| 5 | +10c | 70 | PySR (Symbolic Regression) | 0.63822 |
| 5 | +11c | 52 | Standard Gaussian | 0.53001 |
| 5 | +12c | 37 | LightGBM | 0.56273 |
| 5 | +13c | 32 | Machine Learning (Logistic Regression) | 0.51250 |
| 5 | +14c | 24 | Momentum-Adjusted Gaussian | 0.48103 |
| 5 | +15c | 21 | Standard Gaussian | 0.47253 |
| 5 | +16c | 9 | CatBoost | 0.47892 |
| 5 | +17c | 14 | LightGBM | 0.38057 |
| 5 | +18c | 9 | Momentum-Adjusted Gaussian | 0.44194 |
| 5 | +19c | 7 | Machine Learning (Logistic Regression) | 0.60871 |
| 5 | +20c | 8 | Momentum-Adjusted Gaussian | 0.55859 |
| 5 | +21c | 6 | PySR (Symbolic Regression) | 0.50535 |
| 5 | +22c | 8 | XGBoost | 0.68107 |
| 5 | +23c | 3 | Momentum-Adjusted Gaussian | 0.62682 |
| 5 | +24c | 6 | Machine Learning (Logistic Regression) | 0.74571 |
| 5 | +25c | 7 | Machine Learning (Logistic Regression) | 0.23989 |
| 5 | +27c | 3 | Momentum-Adjusted Gaussian | 0.51926 |
| 5 | +28c | 1 | Machine Learning (Logistic Regression) | 0.26265 |
| 5 | +29c | 1 | PySR (Symbolic Regression) | 1.01311 |
| 5 | +30c | 1 | Momentum-Adjusted Gaussian | 0.32850 |
| 5 | +31c | 2 | Momentum-Adjusted Gaussian | 0.41839 |
| 5 | +32c | 1 | Machine Learning (Logistic Regression) | 0.84089 |
| 5 | +33c | 2 | LightGBM | 0.59547 |
| 5 | +35c | 2 | Machine Learning (Logistic Regression) | 0.45189 |
| 5 | +36c | 4 | Machine Learning (Logistic Regression) | 0.47418 |
| 5 | +37c | 1 | Machine Learning (Logistic Regression) | 0.43272 |
| 5 | +38c | 1 | Momentum-Adjusted Gaussian | 0.49430 |
| 5 | +39c | 1 | Momentum-Adjusted Gaussian | 0.13926 |
| 5 | +43c | 1 | Momentum-Adjusted Gaussian | 0.23572 |
| 5 | +45c | 1 | Machine Learning (Logistic Regression) | 0.57516 |
| 5 | +47c | 1 | Momentum-Adjusted Gaussian | 0.23572 |
| 5 | +49c | 2 | Machine Learning (Logistic Regression) | 0.64805 |
| 5 | +58c | 1 | XGBoost | 0.16592 |
| 5 | +59c | 1 | Machine Learning (Logistic Regression) | 0.78912 |
| 5 | +78c | 1 | LightGBM | 0.43765 |
| 4 | -84c | 1 | Momentum-Adjusted Gaussian | 0.07257 |
| 4 | -78c | 1 | Momentum-Adjusted Gaussian | 0.19845 |
| 4 | -73c | 2 | Momentum-Adjusted Gaussian | 0.14504 |
| 4 | -71c | 1 | Momentum-Adjusted Gaussian | 0.10536 |
| 4 | -68c | 1 | Momentum-Adjusted Gaussian | 0.12783 |
| 4 | -67c | 1 | Momentum-Adjusted Gaussian | 0.30111 |
| 4 | -64c | 1 | CatBoost | 0.20344 |
| 4 | -60c | 1 | XGBoost | 0.23833 |
| 4 | -55c | 1 | Momentum-Adjusted Gaussian | 0.01005 |
| 4 | -53c | 1 | Momentum-Adjusted Gaussian | 0.40048 |
| 4 | -52c | 2 | CatBoost | 0.38927 |
| 4 | -51c | 1 | Machine Learning (Logistic Regression) | 0.39166 |
| 4 | -46c | 1 | Machine Learning (Logistic Regression) | 0.31262 |
| 4 | -44c | 2 | LightGBM | 0.45109 |
| 4 | -39c | 1 | CatBoost | 0.69391 |
| 4 | -38c | 1 | Momentum-Adjusted Gaussian | 0.34249 |
| 4 | -34c | 2 | Machine Learning (Logistic Regression) | 0.27970 |
| 4 | -33c | 1 | CatBoost | 1.05309 |
| 4 | -32c | 1 | Momentum-Adjusted Gaussian | 0.07257 |
| 4 | -31c | 1 | LightGBM | 0.55037 |
| 4 | -29c | 2 | Momentum-Adjusted Gaussian | 0.18077 |
| 4 | -28c | 1 | Momentum-Adjusted Gaussian | 0.08338 |
| 4 | -27c | 5 | PySR (Symbolic Regression) | 0.71062 |
| 4 | -26c | 2 | Machine Learning (Logistic Regression) | 0.32775 |
| 4 | -25c | 3 | CatBoost | 0.94070 |
| 4 | -24c | 5 | CatBoost | 0.66543 |
| 4 | -23c | 6 | LightGBM | 0.69891 |
| 4 | -22c | 3 | LightGBM | 0.22619 |
| 4 | -21c | 5 | Momentum-Adjusted Gaussian | 0.43610 |
| 4 | -20c | 3 | XGBoost | 0.91057 |
| 4 | -19c | 16 | CatBoost | 0.70260 |
| 4 | -18c | 14 | XGBoost | 0.77250 |
| 4 | -17c | 12 | Machine Learning (Logistic Regression) | 0.40738 |
| 4 | -16c | 10 | Machine Learning (Logistic Regression) | 0.40277 |
| 4 | -15c | 17 | Momentum-Adjusted Gaussian | 0.55071 |
| 4 | -14c | 20 | LightGBM | 0.77404 |
| 4 | -13c | 24 | LightGBM | 0.57804 |
| 4 | -12c | 47 | LightGBM | 0.57526 |
| 4 | -11c | 43 | Momentum-Adjusted Gaussian | 0.54245 |
| 4 | -10c | 67 | LightGBM | 0.54584 |
| 4 | -9c | 65 | PySR (Symbolic Regression) | 0.60980 |
| 4 | -8c | 107 | PySR (Symbolic Regression) | 0.54118 |
| 4 | -7c | 194 | Machine Learning (Logistic Regression) | 0.48507 |
| 4 | -6c | 277 | LightGBM | 0.49876 |
| 4 | -5c | 490 | LightGBM | 0.53773 |
| 4 | -4c | 1,134 | LightGBM | 0.54167 |
| 4 | -3c | 3,090 | LightGBM | 0.52403 |
| 4 | -2c | 8,528 | LightGBM | 0.47263 |
| 4 | -1c | 25,585 | LightGBM | 0.40930 |
| 4 | +0c | 86,785 | LightGBM | 0.36505 |
| 4 | +1c | 25,698 | LightGBM | 0.41029 |
| 4 | +2c | 8,583 | LightGBM | 0.45719 |
| 4 | +3c | 3,130 | LightGBM | 0.53087 |
| 4 | +4c | 1,085 | LightGBM | 0.53958 |
| 4 | +5c | 494 | LightGBM | 0.56199 |
| 4 | +6c | 291 | LightGBM | 0.53296 |
| 4 | +7c | 177 | LightGBM | 0.55026 |
| 4 | +8c | 139 | Student-T df=3 | 0.56368 |
| 4 | +9c | 87 | XGBoost | 0.51711 |
| 4 | +10c | 62 | LightGBM | 0.59285 |
| 4 | +11c | 55 | LightGBM | 0.57838 |
| 4 | +12c | 30 | PySR (Symbolic Regression) | 0.66860 |
| 4 | +13c | 22 | LightGBM | 0.71970 |
| 4 | +14c | 28 | LightGBM | 0.56267 |
| 4 | +15c | 13 | CatBoost | 0.71920 |
| 4 | +16c | 24 | LightGBM | 0.77012 |
| 4 | +17c | 17 | PySR (Symbolic Regression) | 0.62475 |
| 4 | +18c | 18 | Standard Gaussian | 0.51456 |
| 4 | +19c | 6 | Momentum-Adjusted Gaussian | 0.24465 |
| 4 | +20c | 7 | CatBoost | 0.54812 |
| 4 | +21c | 4 | LightGBM | 0.48082 |
| 4 | +22c | 4 | Machine Learning (Logistic Regression) | 0.33455 |
| 4 | +23c | 7 | LightGBM | 0.50365 |
| 4 | +24c | 1 | Momentum-Adjusted Gaussian | 0.22314 |
| 4 | +25c | 7 | PySR (Symbolic Regression) | 0.52479 |
| 4 | +26c | 3 | Machine Learning (Logistic Regression) | 0.32754 |
| 4 | +27c | 3 | Momentum-Adjusted Gaussian | 0.66271 |
| 4 | +28c | 2 | Momentum-Adjusted Gaussian | 0.51712 |
| 4 | +29c | 3 | XGBoost | 0.54996 |
| 4 | +30c | 1 | Momentum-Adjusted Gaussian | 0.40048 |
| 4 | +31c | 4 | PySR (Symbolic Regression) | 0.67813 |
| 4 | +32c | 2 | Momentum-Adjusted Gaussian | 0.57616 |
| 4 | +33c | 1 | Momentum-Adjusted Gaussian | 0.19845 |
| 4 | +34c | 2 | Machine Learning (Logistic Regression) | 0.34808 |
| 4 | +35c | 1 | XGBoost | 0.93875 |
| 4 | +39c | 2 | Machine Learning (Logistic Regression) | 0.25879 |
| 4 | +40c | 2 | Momentum-Adjusted Gaussian | 0.26730 |
| 4 | +47c | 2 | Momentum-Adjusted Gaussian | 0.18336 |
| 4 | +49c | 2 | Momentum-Adjusted Gaussian | 0.26884 |
| 4 | +55c | 1 | CatBoost | 0.24086 |
| 4 | +57c | 1 | Momentum-Adjusted Gaussian | 0.03046 |
| 4 | +58c | 2 | Momentum-Adjusted Gaussian | 0.17139 |
| 4 | +59c | 1 | XGBoost | 0.19019 |
| 4 | +63c | 1 | XGBoost | 0.14483 |
| 4 | +68c | 2 | XGBoost | 0.17253 |
| 4 | +69c | 1 | XGBoost | 0.35027 |
| 4 | +71c | 1 | XGBoost | 0.33935 |
| 4 | +77c | 1 | XGBoost | 0.23918 |
| 4 | +81c | 1 | CatBoost | 0.20179 |
| 3 | -70c | 1 | CatBoost | 0.76418 |
| 3 | -69c | 1 | LightGBM | 0.77969 |
| 3 | -67c | 1 | Momentum-Adjusted Gaussian | 0.34249 |
| 3 | -56c | 1 | Momentum-Adjusted Gaussian | 0.49430 |
| 3 | -52c | 1 | CatBoost | 0.49674 |
| 3 | -51c | 1 | PySR (Symbolic Regression) | 0.01005 |
| 3 | -46c | 2 | Machine Learning (Logistic Regression) | 0.43213 |
| 3 | -45c | 1 | Momentum-Adjusted Gaussian | 0.32850 |
| 3 | -43c | 1 | Momentum-Adjusted Gaussian | 0.46204 |
| 3 | -42c | 1 | Momentum-Adjusted Gaussian | 0.75502 |
| 3 | -40c | 1 | Momentum-Adjusted Gaussian | 0.13926 |
| 3 | -39c | 1 | Momentum-Adjusted Gaussian | 0.12783 |
| 3 | -38c | 2 | Momentum-Adjusted Gaussian | 0.40428 |
| 3 | -35c | 1 | Momentum-Adjusted Gaussian | 0.12783 |
| 3 | -34c | 1 | Momentum-Adjusted Gaussian | 0.37106 |
| 3 | -33c | 2 | PySR (Symbolic Regression) | 0.91904 |
| 3 | -32c | 4 | Machine Learning (Logistic Regression) | 0.61876 |
| 3 | -31c | 2 | CatBoost | 0.71003 |
| 3 | -30c | 4 | Momentum-Adjusted Gaussian | 0.49807 |
| 3 | -29c | 1 | Momentum-Adjusted Gaussian | 0.07257 |
| 3 | -27c | 7 | Momentum-Adjusted Gaussian | 0.37081 |
| 3 | -26c | 2 | Machine Learning (Logistic Regression) | 0.32976 |
| 3 | -25c | 4 | Momentum-Adjusted Gaussian | 0.27987 |
| 3 | -24c | 1 | LightGBM | 0.78654 |
| 3 | -23c | 4 | CatBoost | 0.52814 |
| 3 | -22c | 9 | CatBoost | 0.79496 |
| 3 | -21c | 11 | LightGBM | 0.73317 |
| 3 | -20c | 9 | Machine Learning (Logistic Regression) | 0.54062 |
| 3 | -19c | 12 | LightGBM | 0.71717 |
| 3 | -18c | 13 | Standard Gaussian | 0.46847 |
| 3 | -17c | 16 | LightGBM | 0.50999 |
| 3 | -16c | 19 | LightGBM | 0.59712 |
| 3 | -15c | 17 | XGBoost | 0.51628 |
| 3 | -14c | 30 | Standard Gaussian | 0.51263 |
| 3 | -13c | 32 | LightGBM | 0.49917 |
| 3 | -12c | 42 | Machine Learning (Logistic Regression) | 0.51912 |
| 3 | -11c | 68 | LightGBM | 0.54583 |
| 3 | -10c | 76 | LightGBM | 0.55787 |
| 3 | -9c | 124 | Machine Learning (Logistic Regression) | 0.43134 |
| 3 | -8c | 136 | Student-T df=5 | 0.55022 |
| 3 | -7c | 179 | LightGBM | 0.51789 |
| 3 | -6c | 316 | PySR (Symbolic Regression) | 0.56206 |
| 3 | -5c | 497 | LightGBM | 0.52647 |
| 3 | -4c | 995 | LightGBM | 0.52804 |
| 3 | -3c | 3,033 | LightGBM | 0.51575 |
| 3 | -2c | 7,553 | LightGBM | 0.44861 |
| 3 | -1c | 23,417 | LightGBM | 0.37567 |
| 3 | +0c | 85,443 | LightGBM | 0.32486 |
| 3 | +1c | 23,356 | LightGBM | 0.37261 |
| 3 | +2c | 7,625 | LightGBM | 0.43985 |
| 3 | +3c | 2,932 | LightGBM | 0.50219 |
| 3 | +4c | 1,104 | LightGBM | 0.50595 |
| 3 | +5c | 452 | LightGBM | 0.50631 |
| 3 | +6c | 278 | Machine Learning (Logistic Regression) | 0.51115 |
| 3 | +7c | 184 | PySR (Symbolic Regression) | 0.55428 |
| 3 | +8c | 132 | LightGBM | 0.58163 |
| 3 | +9c | 93 | LightGBM | 0.60280 |
| 3 | +10c | 73 | Machine Learning (Logistic Regression) | 0.44041 |
| 3 | +11c | 56 | LightGBM | 0.46780 |
| 3 | +12c | 43 | LightGBM | 0.63548 |
| 3 | +13c | 43 | LightGBM | 0.59408 |
| 3 | +14c | 22 | Machine Learning (Logistic Regression) | 0.42848 |
| 3 | +15c | 17 | Machine Learning (Logistic Regression) | 0.39061 |
| 3 | +16c | 12 | LightGBM | 0.52796 |
| 3 | +17c | 12 | Machine Learning (Logistic Regression) | 0.41214 |
| 3 | +18c | 14 | LightGBM | 0.56069 |
| 3 | +19c | 9 | Momentum-Adjusted Gaussian | 0.48626 |
| 3 | +20c | 8 | LightGBM | 0.66896 |
| 3 | +21c | 5 | CatBoost | 0.74465 |
| 3 | +22c | 4 | PySR (Symbolic Regression) | 0.84830 |
| 3 | +23c | 7 | Momentum-Adjusted Gaussian | 0.46027 |
| 3 | +24c | 2 | CatBoost | 0.32235 |
| 3 | +25c | 1 | Momentum-Adjusted Gaussian | 0.17435 |
| 3 | +26c | 4 | Momentum-Adjusted Gaussian | 0.45938 |
| 3 | +27c | 2 | Machine Learning (Logistic Regression) | 0.50798 |
| 3 | +28c | 5 | PySR (Symbolic Regression) | 0.68558 |
| 3 | +29c | 3 | Momentum-Adjusted Gaussian | 0.55246 |
| 3 | +30c | 1 | CatBoost | 0.16636 |
| 3 | +31c | 2 | Momentum-Adjusted Gaussian | 0.48776 |
| 3 | +32c | 1 | Momentum-Adjusted Gaussian | 0.15082 |
| 3 | +34c | 3 | LightGBM | 0.52401 |
| 3 | +39c | 1 | Momentum-Adjusted Gaussian | 0.16252 |
| 3 | +40c | 1 | Momentum-Adjusted Gaussian | 0.11653 |
| 3 | +42c | 1 | Machine Learning (Logistic Regression) | 0.34459 |
| 3 | +46c | 1 | Momentum-Adjusted Gaussian | 0.09431 |
| 3 | +50c | 2 | LightGBM | 0.58876 |
| 3 | +52c | 1 | Momentum-Adjusted Gaussian | 0.16252 |
| 3 | +58c | 1 | Machine Learning (Logistic Regression) | 0.57279 |
| 3 | +66c | 1 | Momentum-Adjusted Gaussian | 0.07257 |
| 3 | +80c | 1 | CatBoost | 0.27768 |
| 2 | -72c | 1 | CatBoost | 0.43307 |
| 2 | -69c | 1 | Standard Gaussian | 0.01005 |
| 2 | -60c | 1 | Momentum-Adjusted Gaussian | 0.19845 |
| 2 | -57c | 2 | Machine Learning (Logistic Regression) | 0.31501 |
| 2 | -47c | 2 | Momentum-Adjusted Gaussian | 0.54846 |
| 2 | -44c | 2 | CatBoost | 0.58235 |
| 2 | -43c | 1 | CatBoost | 0.68410 |
| 2 | -42c | 5 | LightGBM | 0.78535 |
| 2 | -40c | 1 | Momentum-Adjusted Gaussian | 0.35667 |
| 2 | -39c | 3 | XGBoost | 0.47801 |
| 2 | -38c | 3 | Machine Learning (Logistic Regression) | 0.36132 |
| 2 | -37c | 1 | Momentum-Adjusted Gaussian | 0.09431 |
| 2 | -36c | 4 | XGBoost | 0.71213 |
| 2 | -35c | 2 | Momentum-Adjusted Gaussian | 0.61653 |
| 2 | -34c | 4 | Momentum-Adjusted Gaussian | 0.30140 |
| 2 | -33c | 4 | Momentum-Adjusted Gaussian | 0.13944 |
| 2 | -32c | 4 | XGBoost | 0.78004 |
| 2 | -31c | 6 | Momentum-Adjusted Gaussian | 0.47001 |
| 2 | -30c | 3 | CatBoost | 1.20709 |
| 2 | -29c | 2 | LightGBM | 0.86722 |
| 2 | -28c | 3 | LightGBM | 0.76296 |
| 2 | -27c | 10 | LightGBM | 0.68310 |
| 2 | -26c | 9 | LightGBM | 0.49879 |
| 2 | -25c | 13 | Machine Learning (Logistic Regression) | 0.35182 |
| 2 | -24c | 9 | LightGBM | 0.69582 |
| 2 | -23c | 12 | Machine Learning (Logistic Regression) | 0.38907 |
| 2 | -22c | 15 | CatBoost | 0.69780 |
| 2 | -21c | 16 | LightGBM | 0.63231 |
| 2 | -20c | 8 | LightGBM | 0.50818 |
| 2 | -19c | 24 | CatBoost | 0.54858 |
| 2 | -18c | 15 | Machine Learning (Logistic Regression) | 0.39700 |
| 2 | -17c | 26 | LightGBM | 0.66149 |
| 2 | -16c | 27 | XGBoost | 0.51596 |
| 2 | -15c | 46 | LightGBM | 0.41860 |
| 2 | -14c | 36 | Momentum-Adjusted Gaussian | 0.45458 |
| 2 | -13c | 50 | LightGBM | 0.62119 |
| 2 | -12c | 60 | PySR (Symbolic Regression) | 0.62461 |
| 2 | -11c | 98 | LightGBM | 0.49217 |
| 2 | -10c | 86 | LightGBM | 0.45022 |
| 2 | -9c | 123 | Machine Learning (Logistic Regression) | 0.45608 |
| 2 | -8c | 197 | LightGBM | 0.49977 |
| 2 | -7c | 234 | Momentum-Adjusted Gaussian | 0.50871 |
| 2 | -6c | 386 | LightGBM | 0.51069 |
| 2 | -5c | 650 | LightGBM | 0.54791 |
| 2 | -4c | 1,125 | LightGBM | 0.50483 |
| 2 | -3c | 2,896 | LightGBM | 0.48063 |
| 2 | -2c | 7,065 | LightGBM | 0.41536 |
| 2 | -1c | 21,514 | LightGBM | 0.34002 |
| 2 | +0c | 86,489 | LightGBM | 0.26714 |
| 2 | +1c | 21,920 | LightGBM | 0.34233 |
| 2 | +2c | 6,965 | LightGBM | 0.42587 |
| 2 | +3c | 2,795 | LightGBM | 0.48941 |
| 2 | +4c | 1,166 | LightGBM | 0.50270 |
| 2 | +5c | 565 | Machine Learning (Logistic Regression) | 0.49494 |
| 2 | +6c | 369 | LightGBM | 0.46892 |
| 2 | +7c | 252 | LightGBM | 0.49768 |
| 2 | +8c | 191 | LightGBM | 0.48252 |
| 2 | +9c | 131 | LightGBM | 0.46058 |
| 2 | +10c | 113 | Machine Learning (Logistic Regression) | 0.45613 |
| 2 | +11c | 86 | CatBoost | 0.45525 |
| 2 | +12c | 48 | LightGBM | 0.65451 |
| 2 | +13c | 42 | Machine Learning (Logistic Regression) | 0.48656 |
| 2 | +14c | 48 | PySR (Symbolic Regression) | 0.50791 |
| 2 | +15c | 32 | LightGBM | 0.43896 |
| 2 | +16c | 22 | LightGBM | 0.59849 |
| 2 | +17c | 29 | LightGBM | 0.61459 |
| 2 | +18c | 28 | LightGBM | 0.50313 |
| 2 | +19c | 21 | Momentum-Adjusted Gaussian | 0.54062 |
| 2 | +20c | 11 | CatBoost | 0.66565 |
| 2 | +21c | 18 | LightGBM | 0.56408 |
| 2 | +22c | 6 | LightGBM | 0.43202 |
| 2 | +23c | 10 | LightGBM | 0.62053 |
| 2 | +24c | 13 | Momentum-Adjusted Gaussian | 0.32853 |
| 2 | +25c | 12 | LightGBM | 0.65691 |
| 2 | +26c | 8 | Momentum-Adjusted Gaussian | 0.35266 |
| 2 | +27c | 7 | Machine Learning (Logistic Regression) | 0.47268 |
| 2 | +28c | 9 | XGBoost | 0.73785 |
| 2 | +29c | 4 | Momentum-Adjusted Gaussian | 0.60968 |
| 2 | +30c | 5 | LightGBM | 0.54390 |
| 2 | +31c | 4 | CatBoost | 0.21950 |
| 2 | +32c | 4 | LightGBM | 0.89896 |
| 2 | +33c | 3 | PySR (Symbolic Regression) | 0.88602 |
| 2 | +34c | 4 | Momentum-Adjusted Gaussian | 0.49879 |
| 2 | +35c | 4 | PySR (Symbolic Regression) | 0.72559 |
| 2 | +36c | 4 | CatBoost | 0.79371 |
| 2 | +37c | 4 | Momentum-Adjusted Gaussian | 0.34351 |
| 2 | +39c | 1 | Machine Learning (Logistic Regression) | 0.58392 |
| 2 | +40c | 1 | Standard Gaussian | 0.01005 |
| 2 | +41c | 1 | Momentum-Adjusted Gaussian | 0.47804 |
| 2 | +54c | 1 | Momentum-Adjusted Gaussian | 0.32850 |
| 2 | +62c | 1 | Momentum-Adjusted Gaussian | 0.15082 |
| 2 | +65c | 1 | Machine Learning (Logistic Regression) | 0.39156 |
| 2 | +75c | 1 | Momentum-Adjusted Gaussian | 0.17435 |

## Full Score Matrix

| TTE (min) | Price Momentum | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | -92c | 1 | 0.04082 | 0.07278 | 0.09338 | 0.02020 | 0.22756 | 0.84485 | 0.71542 | 0.21075 | 0.01005 |
| 14 | -67c | 1 | 1.27297 | 1.22879 | 1.20244 | 1.34707 | 0.56821 | 0.56552 | 0.74873 | 0.67460 | 1.08325 |
| 14 | -59c | 1 | 0.91629 | 0.90376 | 0.89604 | 0.96758 | 0.39536 | 0.37814 | 0.55096 | 0.53811 | 0.84014 |
| 14 | -58c | 1 | 0.52763 | 0.53530 | 0.54010 | 0.49430 | 1.13786 | 0.83059 | 0.59098 | 0.64741 | 0.57794 |
| 14 | -56c | 1 | 4.60517 | 3.38851 | 2.97115 | 4.60517 | 3.13210 | 0.25589 | 0.30052 | 0.30446 | 4.60517 |
| 14 | -55c | 1 | 1.46968 | 1.39998 | 1.35954 | 1.56065 | 0.82840 | 0.63365 | 0.65619 | 0.51661 | 1.22287 |
| 14 | -53c | 1 | 0.61619 | 0.61981 | 0.62207 | 0.57982 | 1.23532 | 0.61033 | 0.60785 | 0.47226 | 0.64414 |
| 14 | -52c | 2 | 1.06138 | 1.02262 | 1.00061 | 1.09418 | 1.02968 | 0.55627 | 0.49910 | 0.50477 | 0.94702 |
| 14 | -50c | 2 | 3.45388 | 2.72171 | 2.45096 | 3.56545 | 2.40950 | 0.30873 | 0.40887 | 0.36512 | 3.28228 |
| 14 | -47c | 2 | 0.24879 | 0.27142 | 0.28552 | 0.22346 | 0.48561 | 0.62503 | 0.70811 | 0.75407 | 0.33507 |
| 14 | -46c | 1 | 0.52763 | 0.53530 | 0.54010 | 0.49430 | 0.98481 | 0.41531 | 0.47383 | 0.46444 | 0.57794 |
| 14 | -45c | 2 | 0.97316 | 0.95154 | 0.93867 | 1.03021 | 0.54973 | 0.67768 | 0.73204 | 0.68260 | 0.87794 |
| 14 | -44c | 3 | 0.37851 | 0.39065 | 0.39927 | 0.37974 | 0.42713 | 0.69323 | 0.67892 | 0.61870 | 0.40213 |
| 14 | -43c | 1 | 0.61619 | 0.61981 | 0.62207 | 0.57982 | 1.10106 | 0.62721 | 0.63915 | 0.52758 | 0.64414 |
| 14 | -41c | 2 | 0.35160 | 0.36374 | 0.37288 | 0.37201 | 0.19537 | 0.77385 | 0.89669 | 0.65802 | 0.34833 |
| 14 | -40c | 5 | 0.81112 | 0.80375 | 0.79925 | 0.84187 | 0.61985 | 0.71926 | 0.68994 | 0.72167 | 0.77008 |
| 14 | -39c | 3 | 1.55342 | 1.44010 | 1.38015 | 1.64791 | 1.36267 | 0.82675 | 0.90214 | 0.71125 | 1.32677 |
| 14 | -38c | 6 | 0.91056 | 0.87017 | 0.85017 | 0.96049 | 0.75845 | 0.71719 | 0.79294 | 0.73149 | 0.84044 |
| 14 | -37c | 4 | 1.04201 | 0.96850 | 0.93390 | 1.11158 | 0.94368 | 0.71586 | 0.76696 | 0.71959 | 0.99460 |
| 14 | -36c | 1 | 0.38566 | 0.40024 | 0.40938 | 0.35667 | 0.63351 | 0.51884 | 0.36790 | 0.54290 | 0.46370 |
| 14 | -35c | 1 | 0.34249 | 0.35942 | 0.37003 | 0.31471 | 0.55367 | 0.47450 | 0.49479 | 0.49078 | 0.42590 |
| 14 | -34c | 7 | 1.16834 | 1.11786 | 1.09002 | 1.19235 | 1.24982 | 0.77218 | 0.79495 | 0.66610 | 1.03302 |
| 14 | -33c | 8 | 0.47990 | 0.48819 | 0.49404 | 0.47297 | 0.54309 | 0.59522 | 0.58939 | 0.61890 | 0.46092 |
| 14 | -32c | 4 | 0.81528 | 0.77761 | 0.75934 | 0.83869 | 0.86840 | 0.71554 | 0.71107 | 0.69551 | 0.77000 |
| 14 | -31c | 7 | 0.49005 | 0.50000 | 0.50623 | 0.47747 | 0.59300 | 0.53362 | 0.57939 | 0.56718 | 0.52005 |
| 14 | -30c | 5 | 1.27829 | 1.15711 | 1.10431 | 1.42728 | 1.14420 | 0.75979 | 0.68631 | 0.65283 | 1.50230 |
| 14 | -29c | 1 | 2.99573 | 2.52000 | 2.30987 | 3.50656 | 2.49018 | 1.68866 | 1.78581 | 1.33471 | 3.26424 |
| 14 | -28c | 8 | 1.46723 | 1.29090 | 1.21857 | 1.63015 | 1.29302 | 0.80854 | 0.73488 | 0.75595 | 1.55615 |
| 14 | -27c | 10 | 0.66717 | 0.66160 | 0.65889 | 0.67518 | 0.66771 | 0.62085 | 0.65049 | 0.62101 | 0.63452 |
| 14 | -26c | 9 | 0.57160 | 0.57490 | 0.57721 | 0.56854 | 0.61920 | 0.65320 | 0.64275 | 0.64137 | 0.57715 |
| 14 | -25c | 15 | 0.62224 | 0.62281 | 0.62341 | 0.61900 | 0.68579 | 0.65142 | 0.67654 | 0.67976 | 0.62448 |
| 14 | -24c | 10 | 0.51720 | 0.52360 | 0.52787 | 0.52883 | 0.45677 | 0.58732 | 0.66384 | 0.59910 | 0.52217 |
| 14 | -23c | 13 | 0.75900 | 0.75247 | 0.74866 | 0.78731 | 0.64348 | 0.69807 | 0.70329 | 0.69454 | 0.71906 |
| 14 | -22c | 16 | 0.83485 | 0.76788 | 0.74131 | 0.93243 | 0.80090 | 0.59179 | 0.51772 | 0.57264 | 0.93875 |
| 14 | -21c | 12 | 0.55720 | 0.56165 | 0.56464 | 0.54852 | 0.61163 | 0.57963 | 0.57370 | 0.59701 | 0.57994 |
| 14 | -20c | 23 | 0.66666 | 0.66570 | 0.66528 | 0.67069 | 0.67483 | 0.65896 | 0.65868 | 0.66400 | 0.65895 |
| 14 | -19c | 23 | 0.72917 | 0.72479 | 0.72225 | 0.73038 | 0.75792 | 0.72205 | 0.70131 | 0.70185 | 0.71071 |
| 14 | -18c | 25 | 0.59519 | 0.59903 | 0.60155 | 0.59922 | 0.58486 | 0.63544 | 0.62148 | 0.61700 | 0.59717 |
| 14 | -17c | 33 | 0.70953 | 0.70486 | 0.70225 | 0.71241 | 0.71688 | 0.70158 | 0.67356 | 0.68106 | 0.68906 |
| 14 | -16c | 38 | 0.59617 | 0.59863 | 0.60039 | 0.59401 | 0.61224 | 0.61146 | 0.61192 | 0.60789 | 0.60922 |
| 14 | -15c | 47 | 0.73207 | 0.72363 | 0.71922 | 0.74184 | 0.73263 | 0.70610 | 0.69179 | 0.69320 | 0.71224 |
| 14 | -14c | 51 | 0.69410 | 0.69181 | 0.69055 | 0.69124 | 0.71752 | 0.69392 | 0.67045 | 0.68094 | 0.68448 |
| 14 | -13c | 69 | 0.66320 | 0.66042 | 0.65926 | 0.67008 | 0.66080 | 0.66248 | 0.66779 | 0.65980 | 0.66055 |
| 14 | -12c | 64 | 0.63385 | 0.63478 | 0.63554 | 0.63355 | 0.64196 | 0.63798 | 0.63356 | 0.63808 | 0.64032 |
| 14 | -11c | 95 | 0.67216 | 0.67122 | 0.67080 | 0.67217 | 0.68119 | 0.66785 | 0.65814 | 0.66673 | 0.66826 |
| 14 | -10c | 136 | 0.63892 | 0.64027 | 0.64121 | 0.64017 | 0.64071 | 0.64604 | 0.64952 | 0.64430 | 0.64083 |
| 14 | -9c | 172 | 0.70204 | 0.69878 | 0.69702 | 0.70765 | 0.69781 | 0.68602 | 0.68101 | 0.68425 | 0.69046 |
| 14 | -8c | 205 | 0.66238 | 0.66179 | 0.66162 | 0.66302 | 0.66539 | 0.65972 | 0.64860 | 0.65871 | 0.66263 |
| 14 | -7c | 350 | 0.67090 | 0.67034 | 0.67012 | 0.67078 | 0.67569 | 0.66666 | 0.66023 | 0.66612 | 0.66905 |
| 14 | -6c | 495 | 0.69208 | 0.68964 | 0.68838 | 0.69647 | 0.69132 | 0.68025 | 0.67535 | 0.68029 | 0.68437 |
| 14 | -5c | 871 | 0.65533 | 0.65572 | 0.65610 | 0.65774 | 0.65368 | 0.65552 | 0.65239 | 0.65473 | 0.65824 |
| 14 | -4c | 1,956 | 0.67030 | 0.66988 | 0.66976 | 0.67278 | 0.67009 | 0.66794 | 0.66584 | 0.66738 | 0.67016 |
| 14 | -3c | 4,872 | 0.65971 | 0.65992 | 0.66019 | 0.66134 | 0.65905 | 0.65856 | 0.65609 | 0.65794 | 0.66278 |
| 14 | -2c | 11,578 | 0.66467 | 0.66462 | 0.66473 | 0.66751 | 0.66389 | 0.66269 | 0.66134 | 0.66230 | 0.66612 |
| 14 | -1c | 32,322 | 0.65818 | 0.65843 | 0.65872 | 0.65991 | 0.65804 | 0.65702 | 0.65572 | 0.65686 | 0.66073 |
| 14 | +0c | 93,381 | 0.65832 | 0.65832 | 0.65850 | 0.65832 | 0.65841 | 0.65640 | 0.65423 | 0.65617 | 0.66067 |
| 14 | +1c | 31,891 | 0.65990 | 0.66005 | 0.66029 | 0.66026 | 0.65995 | 0.65856 | 0.65720 | 0.65839 | 0.66219 |
| 14 | +2c | 11,573 | 0.66528 | 0.66522 | 0.66531 | 0.66613 | 0.66547 | 0.66422 | 0.66315 | 0.66399 | 0.66674 |
| 14 | +3c | 4,885 | 0.66217 | 0.66230 | 0.66250 | 0.66444 | 0.66169 | 0.65976 | 0.65718 | 0.65966 | 0.66502 |
| 14 | +4c | 1,973 | 0.65563 | 0.65618 | 0.65663 | 0.65588 | 0.65549 | 0.65529 | 0.65184 | 0.65491 | 0.65950 |
| 14 | +5c | 898 | 0.67661 | 0.67546 | 0.67497 | 0.67869 | 0.67718 | 0.67288 | 0.66679 | 0.67262 | 0.67500 |
| 14 | +6c | 472 | 0.68003 | 0.67841 | 0.67761 | 0.68343 | 0.67944 | 0.67084 | 0.66420 | 0.66890 | 0.67689 |
| 14 | +7c | 308 | 0.68253 | 0.67980 | 0.67852 | 0.69108 | 0.67529 | 0.66574 | 0.65572 | 0.66585 | 0.67952 |
| 14 | +8c | 220 | 0.61312 | 0.61601 | 0.61789 | 0.61559 | 0.60957 | 0.61650 | 0.61672 | 0.62043 | 0.62614 |
| 14 | +9c | 151 | 0.72091 | 0.70776 | 0.70255 | 0.72406 | 0.72125 | 0.67240 | 0.65983 | 0.67007 | 0.71442 |
| 14 | +10c | 131 | 0.69028 | 0.68570 | 0.68343 | 0.69994 | 0.67859 | 0.66079 | 0.64586 | 0.65501 | 0.68506 |
| 14 | +11c | 106 | 0.64358 | 0.64428 | 0.64486 | 0.64451 | 0.64874 | 0.65283 | 0.64069 | 0.64899 | 0.64761 |
| 14 | +12c | 65 | 0.64317 | 0.64422 | 0.64498 | 0.63864 | 0.66694 | 0.66433 | 0.65354 | 0.66181 | 0.64657 |
| 14 | +13c | 55 | 0.69181 | 0.67854 | 0.67300 | 0.71084 | 0.67774 | 0.65209 | 0.65094 | 0.64329 | 0.71758 |
| 14 | +14c | 52 | 0.62952 | 0.62508 | 0.62380 | 0.63467 | 0.64087 | 0.63591 | 0.61341 | 0.62371 | 0.64806 |
| 14 | +15c | 45 | 0.71064 | 0.70513 | 0.70223 | 0.72696 | 0.67173 | 0.65872 | 0.66167 | 0.66108 | 0.69930 |
| 14 | +16c | 26 | 0.66532 | 0.66266 | 0.66144 | 0.66805 | 0.67945 | 0.67689 | 0.67773 | 0.68545 | 0.65583 |
| 14 | +17c | 29 | 0.72098 | 0.70600 | 0.69894 | 0.73647 | 0.71003 | 0.64460 | 0.61193 | 0.62606 | 0.70876 |
| 14 | +18c | 23 | 0.71031 | 0.70598 | 0.70361 | 0.73557 | 0.60834 | 0.65104 | 0.65327 | 0.65675 | 0.68533 |
| 14 | +19c | 22 | 0.75147 | 0.73769 | 0.73078 | 0.77630 | 0.68300 | 0.68965 | 0.67613 | 0.64732 | 0.73582 |
| 14 | +20c | 24 | 0.62435 | 0.62617 | 0.62743 | 0.63058 | 0.61362 | 0.62428 | 0.63433 | 0.62613 | 0.62753 |
| 14 | +21c | 7 | 0.73155 | 0.72706 | 0.72446 | 0.74892 | 0.66515 | 0.69180 | 0.63224 | 0.66782 | 0.70989 |
| 14 | +22c | 15 | 0.81065 | 0.79972 | 0.79330 | 0.84654 | 0.63392 | 0.63529 | 0.60745 | 0.66655 | 0.76182 |
| 14 | +23c | 12 | 0.81800 | 0.79963 | 0.79004 | 0.82670 | 0.86496 | 0.71879 | 0.70545 | 0.70720 | 0.78423 |
| 14 | +24c | 6 | 0.56138 | 0.56432 | 0.56646 | 0.57223 | 0.52657 | 0.58627 | 0.59393 | 0.58989 | 0.58073 |
| 14 | +25c | 13 | 0.66660 | 0.66477 | 0.66391 | 0.68259 | 0.59480 | 0.68897 | 0.68439 | 0.70118 | 0.65238 |
| 14 | +26c | 14 | 0.96426 | 0.89706 | 0.87052 | 1.02605 | 0.85439 | 0.67264 | 0.69522 | 0.69593 | 0.98628 |
| 14 | +27c | 10 | 0.63655 | 0.63682 | 0.63719 | 0.66238 | 0.47788 | 0.55730 | 0.53726 | 0.56853 | 0.63870 |
| 14 | +28c | 7 | 0.55045 | 0.55461 | 0.55748 | 0.56880 | 0.43357 | 0.50313 | 0.50963 | 0.59410 | 0.55739 |
| 14 | +29c | 6 | 0.50075 | 0.50986 | 0.51556 | 0.49574 | 0.58438 | 0.53465 | 0.49678 | 0.56264 | 0.53047 |
| 14 | +30c | 7 | 0.93553 | 0.90248 | 0.88457 | 0.95903 | 0.92087 | 0.83486 | 0.86556 | 0.74375 | 0.85371 |
| 14 | +31c | 4 | 0.56240 | 0.56878 | 0.57276 | 0.56835 | 0.56087 | 0.58247 | 0.57605 | 0.56171 | 0.59045 |
| 14 | +32c | 2 | 1.43060 | 1.32068 | 1.26472 | 1.51207 | 1.31045 | 0.71576 | 0.90667 | 0.67772 | 1.27818 |
| 14 | +33c | 3 | 1.91073 | 1.70603 | 1.60821 | 2.12181 | 1.24849 | 0.52835 | 0.54800 | 0.54103 | 1.81370 |
| 14 | +34c | 5 | 0.53462 | 0.53854 | 0.54133 | 0.52844 | 0.67701 | 0.67682 | 0.72311 | 0.74888 | 0.54311 |
| 14 | +35c | 8 | 1.19118 | 1.10309 | 1.06091 | 1.27066 | 1.04277 | 0.56639 | 0.53624 | 0.55641 | 1.12171 |
| 14 | +36c | 1 | 0.21072 | 0.23583 | 0.25143 | 0.18633 | 0.46548 | 0.54979 | 0.62336 | 0.58020 | 0.28569 |
| 14 | +37c | 4 | 0.64256 | 0.63036 | 0.62472 | 0.65174 | 0.69124 | 0.82312 | 0.81155 | 0.89724 | 0.58733 |
| 14 | +38c | 4 | 0.82505 | 0.78124 | 0.76050 | 0.87391 | 0.62688 | 0.62709 | 0.57531 | 0.55728 | 0.75991 |
| 14 | +39c | 4 | 1.00686 | 0.91565 | 0.87587 | 1.10228 | 0.86166 | 0.46091 | 0.44117 | 0.53891 | 1.04753 |
| 14 | +40c | 1 | 2.99573 | 2.52000 | 2.30987 | 3.50656 | 1.88686 | 0.41393 | 0.61019 | 0.52481 | 3.45259 |
| 14 | +42c | 1 | 0.79851 | 0.79308 | 0.78972 | 0.75502 | 1.57954 | 0.71442 | 0.69650 | 0.73887 | 0.75982 |
| 14 | +43c | 3 | 1.32999 | 1.17786 | 1.11185 | 1.50499 | 0.94019 | 0.61732 | 0.56621 | 0.53470 | 1.51013 |
| 14 | +45c | 1 | 0.63488 | 0.63764 | 0.63936 | 0.59784 | 1.36939 | 0.58809 | 0.64851 | 0.62538 | 0.64519 |
| 14 | +47c | 1 | 0.69315 | 0.69315 | 0.69315 | 0.65393 | 1.50831 | 0.56347 | 0.70086 | 0.65919 | 0.68661 |
| 14 | +49c | 1 | 1.38629 | 1.32817 | 1.29403 | 1.46968 | 0.60084 | 0.70285 | 0.64953 | 0.79932 | 1.18411 |
| 14 | +55c | 2 | 0.57003 | 0.57588 | 0.57955 | 0.58238 | 0.54216 | 0.72194 | 0.63897 | 0.67411 | 0.58236 |
| 14 | +56c | 1 | 0.01005 | 0.03434 | 0.05260 | 0.01005 | 0.08274 | 0.62933 | 0.57689 | 0.30134 | 0.01005 |
| 14 | +57c | 1 | 0.26136 | 0.28316 | 0.29676 | 0.23572 | 0.86434 | 0.32238 | 0.41280 | 0.34644 | 0.33943 |
| 14 | +59c | 2 | 0.26044 | 0.27680 | 0.28858 | 0.24404 | 0.77070 | 0.50809 | 0.46281 | 0.39485 | 0.28180 |
| 14 | +94c | 1 | 4.60517 | 3.38851 | 2.97115 | 4.60517 | 1.58342 | 0.44228 | 0.68513 | 0.20741 | 4.60517 |
| 13 | -53c | 1 | 1.04982 | 1.02744 | 1.01377 | 1.10866 | 0.53013 | 0.40389 | 0.52050 | 0.37411 | 0.93055 |
| 13 | -50c | 1 | 0.79851 | 0.79308 | 0.78972 | 0.84397 | 0.37643 | 0.58699 | 0.77053 | 0.70751 | 0.75982 |
| 13 | -49c | 1 | 0.59784 | 0.60230 | 0.60509 | 0.56212 | 1.14661 | 0.49561 | 0.55141 | 0.44495 | 0.63065 |
| 13 | -46c | 1 | 1.51413 | 1.43780 | 1.39382 | 1.60944 | 0.96993 | 0.39558 | 0.56836 | 0.38744 | 1.25542 |
| 13 | -45c | 2 | 1.75328 | 1.54855 | 1.45514 | 1.97642 | 1.20169 | 0.51522 | 0.64228 | 0.58474 | 1.72013 |
| 13 | -44c | 1 | 0.65393 | 0.65580 | 0.65696 | 0.61619 | 1.17643 | 0.53031 | 0.67558 | 0.53519 | 0.67158 |
| 13 | -43c | 1 | 0.57982 | 0.58510 | 0.58841 | 0.61619 | 0.27944 | 0.74489 | 0.52823 | 0.75939 | 0.60520 |
| 13 | -40c | 2 | 0.42455 | 0.43883 | 0.44775 | 0.43614 | 0.28300 | 0.53201 | 0.57790 | 0.55734 | 0.41732 |
| 13 | -35c | 1 | 0.65393 | 0.65580 | 0.65696 | 0.69315 | 0.36859 | 0.39303 | 0.49266 | 0.48997 | 0.65882 |
| 13 | -34c | 2 | 0.27409 | 0.29555 | 0.30888 | 0.24797 | 0.43757 | 0.53485 | 0.47555 | 0.55806 | 0.34987 |
| 13 | -33c | 3 | 0.53255 | 0.54016 | 0.54493 | 0.52594 | 0.60516 | 0.51550 | 0.57387 | 0.48462 | 0.57242 |
| 13 | -32c | 4 | 1.10732 | 1.01087 | 0.96786 | 1.18202 | 1.19220 | 0.64605 | 0.77187 | 0.61544 | 1.09133 |
| 13 | -31c | 1 | 0.82098 | 0.81430 | 0.81015 | 0.77653 | 1.23980 | 0.56616 | 0.68213 | 0.60206 | 0.78958 |
| 13 | -30c | 4 | 0.38927 | 0.40537 | 0.41535 | 0.35915 | 0.59203 | 0.49564 | 0.50920 | 0.51645 | 0.42321 |
| 13 | -29c | 2 | 0.28282 | 0.30342 | 0.31628 | 0.25658 | 0.41535 | 0.69022 | 0.70207 | 0.61608 | 0.36712 |
| 13 | -28c | 2 | 0.38740 | 0.40237 | 0.41203 | 0.39837 | 0.27317 | 0.39838 | 0.51128 | 0.52294 | 0.36262 |
| 13 | -27c | 2 | 0.69475 | 0.68824 | 0.68469 | 0.72069 | 0.57583 | 0.68715 | 0.52302 | 0.69346 | 0.62672 |
| 13 | -26c | 3 | 0.48149 | 0.49160 | 0.49794 | 0.51471 | 0.30367 | 0.40700 | 0.47482 | 0.49893 | 0.52595 |
| 13 | -25c | 6 | 0.75904 | 0.75344 | 0.75011 | 0.75123 | 0.84191 | 0.75381 | 0.70890 | 0.71828 | 0.73154 |
| 13 | -24c | 5 | 0.94351 | 0.92666 | 0.91647 | 0.97936 | 0.81684 | 0.76273 | 0.78674 | 0.79111 | 0.86082 |
| 13 | -23c | 4 | 0.99156 | 0.96936 | 0.95618 | 1.00472 | 1.02471 | 0.85859 | 0.84014 | 0.85108 | 0.89939 |
| 13 | -22c | 6 | 0.96870 | 0.94041 | 0.92425 | 0.99186 | 0.94738 | 0.82234 | 0.80597 | 0.78451 | 0.87188 |
| 13 | -21c | 6 | 0.56398 | 0.56871 | 0.57179 | 0.56458 | 0.56131 | 0.59169 | 0.60002 | 0.60196 | 0.58148 |
| 13 | -20c | 8 | 0.97928 | 0.91236 | 0.88236 | 1.05490 | 0.91751 | 0.72436 | 0.69082 | 0.71975 | 0.99950 |
| 13 | -19c | 10 | 0.55744 | 0.56313 | 0.56691 | 0.54130 | 0.63964 | 0.66904 | 0.69871 | 0.64311 | 0.56724 |
| 13 | -18c | 8 | 0.42366 | 0.43656 | 0.44465 | 0.39293 | 0.53330 | 0.57258 | 0.56748 | 0.55531 | 0.48567 |
| 13 | -17c | 16 | 0.51555 | 0.52426 | 0.52970 | 0.51937 | 0.48385 | 0.53645 | 0.52936 | 0.54337 | 0.54317 |
| 13 | -16c | 18 | 0.78412 | 0.76790 | 0.75923 | 0.80313 | 0.77273 | 0.67329 | 0.66433 | 0.69925 | 0.74461 |
| 13 | -15c | 22 | 0.69968 | 0.69469 | 0.69198 | 0.70526 | 0.69167 | 0.66441 | 0.68554 | 0.66340 | 0.68386 |
| 13 | -14c | 26 | 0.62793 | 0.62824 | 0.62874 | 0.61841 | 0.67169 | 0.66645 | 0.64820 | 0.64991 | 0.63691 |
| 13 | -13c | 35 | 0.69433 | 0.68999 | 0.68769 | 0.69512 | 0.70616 | 0.68795 | 0.68927 | 0.68486 | 0.67844 |
| 13 | -12c | 50 | 0.57182 | 0.57608 | 0.57889 | 0.57609 | 0.56369 | 0.57496 | 0.57425 | 0.58350 | 0.59233 |
| 13 | -11c | 43 | 0.61520 | 0.61491 | 0.61521 | 0.61754 | 0.62169 | 0.62616 | 0.61134 | 0.62113 | 0.61350 |
| 13 | -10c | 59 | 0.65610 | 0.65475 | 0.65424 | 0.66962 | 0.63281 | 0.62225 | 0.61098 | 0.62964 | 0.64718 |
| 13 | -9c | 103 | 0.69848 | 0.69047 | 0.68716 | 0.70600 | 0.71119 | 0.68395 | 0.68123 | 0.68082 | 0.70572 |
| 13 | -8c | 138 | 0.65595 | 0.64892 | 0.64665 | 0.66444 | 0.65492 | 0.63382 | 0.63027 | 0.63377 | 0.66588 |
| 13 | -7c | 215 | 0.63982 | 0.64098 | 0.64182 | 0.63846 | 0.64483 | 0.64396 | 0.63836 | 0.64344 | 0.64368 |
| 13 | -6c | 302 | 0.62963 | 0.63091 | 0.63187 | 0.62854 | 0.63099 | 0.62991 | 0.62874 | 0.62957 | 0.63621 |
| 13 | -5c | 604 | 0.62587 | 0.62580 | 0.62628 | 0.62952 | 0.62259 | 0.62282 | 0.62156 | 0.62208 | 0.63638 |
| 13 | -4c | 1,512 | 0.65464 | 0.65328 | 0.65283 | 0.65751 | 0.65517 | 0.65148 | 0.64877 | 0.65258 | 0.65529 |
| 13 | -3c | 4,319 | 0.64629 | 0.64604 | 0.64621 | 0.64887 | 0.64560 | 0.64290 | 0.64067 | 0.64269 | 0.64928 |
| 13 | -2c | 10,770 | 0.63904 | 0.63943 | 0.63990 | 0.64174 | 0.63860 | 0.63814 | 0.63693 | 0.63811 | 0.64192 |
| 13 | -1c | 30,000 | 0.63595 | 0.63602 | 0.63639 | 0.63763 | 0.63572 | 0.63388 | 0.63216 | 0.63383 | 0.63902 |
| 13 | +0c | 89,299 | 0.62899 | 0.62934 | 0.62992 | 0.62899 | 0.62854 | 0.62667 | 0.62451 | 0.62668 | 0.63385 |
| 13 | +1c | 30,067 | 0.63063 | 0.63118 | 0.63180 | 0.63139 | 0.63023 | 0.62921 | 0.62770 | 0.62920 | 0.63538 |
| 13 | +2c | 11,127 | 0.63888 | 0.63897 | 0.63933 | 0.64006 | 0.63833 | 0.63609 | 0.63432 | 0.63601 | 0.64297 |
| 13 | +3c | 4,336 | 0.62675 | 0.62789 | 0.62882 | 0.62880 | 0.62515 | 0.62469 | 0.62229 | 0.62453 | 0.63466 |
| 13 | +4c | 1,522 | 0.64453 | 0.64456 | 0.64482 | 0.64672 | 0.64457 | 0.64418 | 0.64240 | 0.64435 | 0.64742 |
| 13 | +5c | 591 | 0.64132 | 0.64042 | 0.64031 | 0.64330 | 0.64089 | 0.63563 | 0.62969 | 0.63587 | 0.64614 |
| 13 | +6c | 361 | 0.64315 | 0.64283 | 0.64293 | 0.64148 | 0.64745 | 0.64457 | 0.64154 | 0.64222 | 0.64564 |
| 13 | +7c | 199 | 0.66437 | 0.66352 | 0.66318 | 0.66591 | 0.66725 | 0.66506 | 0.65474 | 0.66204 | 0.66036 |
| 13 | +8c | 119 | 0.63771 | 0.63708 | 0.63700 | 0.64898 | 0.62261 | 0.63098 | 0.62372 | 0.62814 | 0.63720 |
| 13 | +9c | 93 | 0.68286 | 0.67472 | 0.67109 | 0.68032 | 0.69921 | 0.66539 | 0.64178 | 0.65850 | 0.67796 |
| 13 | +10c | 79 | 0.68521 | 0.68193 | 0.68029 | 0.69122 | 0.68109 | 0.68047 | 0.67887 | 0.67805 | 0.67447 |
| 13 | +11c | 52 | 0.80356 | 0.78466 | 0.77476 | 0.81238 | 0.81315 | 0.75904 | 0.73172 | 0.74154 | 0.76822 |
| 13 | +12c | 34 | 0.60593 | 0.60771 | 0.60910 | 0.61449 | 0.58756 | 0.62171 | 0.62514 | 0.61958 | 0.60681 |
| 13 | +13c | 33 | 0.87051 | 0.84652 | 0.83394 | 0.88926 | 0.86388 | 0.78552 | 0.78969 | 0.78620 | 0.82364 |
| 13 | +14c | 23 | 0.72911 | 0.72359 | 0.72044 | 0.74166 | 0.70492 | 0.70121 | 0.70297 | 0.69162 | 0.70425 |
| 13 | +15c | 24 | 0.57921 | 0.57992 | 0.58097 | 0.58408 | 0.57549 | 0.56822 | 0.56953 | 0.57622 | 0.58928 |
| 13 | +16c | 24 | 0.74210 | 0.73448 | 0.73020 | 0.73713 | 0.79382 | 0.76658 | 0.76304 | 0.75205 | 0.71249 |
| 13 | +17c | 17 | 0.70517 | 0.70285 | 0.70152 | 0.71057 | 0.71186 | 0.67157 | 0.65797 | 0.67103 | 0.69468 |
| 13 | +18c | 8 | 0.54842 | 0.55411 | 0.55776 | 0.54904 | 0.57243 | 0.57908 | 0.54313 | 0.56794 | 0.54769 |
| 13 | +19c | 6 | 0.55868 | 0.56557 | 0.56985 | 0.54982 | 0.62024 | 0.58338 | 0.55992 | 0.58733 | 0.57191 |
| 13 | +20c | 11 | 0.53785 | 0.54511 | 0.54967 | 0.51207 | 0.70606 | 0.56761 | 0.57661 | 0.57653 | 0.55876 |
| 13 | +21c | 5 | 0.63753 | 0.63775 | 0.63807 | 0.65222 | 0.58570 | 0.58281 | 0.58343 | 0.58351 | 0.64492 |
| 13 | +22c | 12 | 0.52422 | 0.53184 | 0.53665 | 0.53626 | 0.46551 | 0.50315 | 0.47758 | 0.51803 | 0.55059 |
| 13 | +23c | 11 | 0.43845 | 0.45150 | 0.45961 | 0.44245 | 0.41120 | 0.52665 | 0.51106 | 0.52426 | 0.47502 |
| 13 | +24c | 6 | 0.62380 | 0.62335 | 0.62364 | 0.65085 | 0.46960 | 0.73168 | 0.64779 | 0.73897 | 0.60734 |
| 13 | +25c | 1 | 0.43078 | 0.44305 | 0.45075 | 0.40048 | 0.68713 | 0.56564 | 0.58974 | 0.59780 | 0.49066 |
| 13 | +26c | 2 | 0.37900 | 0.39398 | 0.40338 | 0.37836 | 0.42472 | 0.47964 | 0.53096 | 0.46899 | 0.45205 |
| 13 | +27c | 5 | 0.54844 | 0.55443 | 0.55826 | 0.52906 | 0.74208 | 0.64922 | 0.66949 | 0.62814 | 0.56688 |
| 13 | +28c | 3 | 0.29818 | 0.31783 | 0.33010 | 0.27155 | 0.52527 | 0.43189 | 0.42533 | 0.42688 | 0.37287 |
| 13 | +29c | 5 | 0.72306 | 0.71653 | 0.71290 | 0.73671 | 0.71112 | 0.56777 | 0.65529 | 0.57659 | 0.70926 |
| 13 | +31c | 1 | 0.79851 | 0.79308 | 0.78972 | 0.75502 | 1.33866 | 0.71084 | 0.73494 | 0.72202 | 0.75982 |
| 13 | +32c | 2 | 0.62728 | 0.63015 | 0.63197 | 0.66590 | 0.31811 | 0.47423 | 0.47245 | 0.53480 | 0.64901 |
| 13 | +33c | 1 | 0.37106 | 0.38642 | 0.39605 | 0.34249 | 0.71026 | 0.47292 | 0.56674 | 0.51091 | 0.44087 |
| 13 | +34c | 1 | 0.56212 | 0.56821 | 0.57202 | 0.52763 | 1.03350 | 0.49826 | 0.50268 | 0.42941 | 0.59213 |
| 13 | +35c | 2 | 0.71858 | 0.71705 | 0.71611 | 0.76071 | 0.35056 | 0.71774 | 0.67751 | 0.69964 | 0.71686 |
| 13 | +36c | 2 | 0.46317 | 0.46905 | 0.47432 | 0.43878 | 0.84019 | 0.82100 | 0.63611 | 0.55625 | 0.42509 |
| 13 | +37c | 3 | 0.77716 | 0.77285 | 0.77017 | 0.76234 | 1.09361 | 0.68814 | 0.63866 | 0.62509 | 0.74954 |
| 13 | +41c | 1 | 0.41552 | 0.42855 | 0.43673 | 0.38566 | 0.91522 | 0.50194 | 0.45412 | 0.48695 | 0.47820 |
| 13 | +42c | 1 | 2.20727 | 1.98557 | 1.87247 | 2.40795 | 1.27051 | 0.67090 | 0.57380 | 0.37542 | 1.89753 |
| 13 | +44c | 2 | 1.47062 | 1.28116 | 1.19717 | 1.66212 | 1.01616 | 0.41728 | 0.42939 | 0.41371 | 1.52106 |
| 13 | +45c | 1 | 0.67334 | 0.67430 | 0.67489 | 0.63488 | 1.43430 | 0.55944 | 0.62190 | 0.49333 | 0.67262 |
| 13 | +49c | 1 | 1.04982 | 1.02744 | 1.01377 | 1.10866 | 0.40959 | 0.71913 | 0.70892 | 0.91838 | 0.94733 |
| 13 | +53c | 1 | 1.17118 | 1.13784 | 1.11772 | 1.23787 | 0.43322 | 0.92912 | 0.80660 | 0.80205 | 1.03135 |
| 13 | +56c | 1 | 0.09431 | 0.12652 | 0.14646 | 0.07257 | 0.39267 | 0.37701 | 0.29665 | 0.48763 | 0.12516 |
| 12 | -64c | 1 | 1.07881 | 1.05399 | 1.03888 | 1.13943 | 0.46626 | 0.44230 | 0.39325 | 0.45441 | 0.95020 |
| 12 | -63c | 1 | 0.11653 | 0.14769 | 0.16693 | 0.09431 | 0.31490 | 0.80714 | 0.74156 | 1.15328 | 0.17020 |
| 12 | -56c | 1 | 1.56065 | 1.47704 | 1.42920 | 1.66073 | 0.89196 | 0.21587 | 0.25327 | 0.21417 | 1.28997 |
| 12 | -53c | 1 | 0.18633 | 0.21308 | 0.22966 | 0.16252 | 0.40555 | 1.08353 | 0.89634 | 1.11160 | 0.26593 |
| 12 | -52c | 1 | 2.81341 | 2.40396 | 2.21714 | 3.21888 | 1.95146 | 0.48282 | 0.26285 | 0.33287 | 2.75366 |
| 12 | -49c | 1 | 0.17435 | 0.20191 | 0.21896 | 0.15082 | 0.35744 | 0.84375 | 0.66129 | 0.73590 | 0.25118 |
| 12 | -47c | 1 | 1.71480 | 1.60448 | 1.54295 | 1.83258 | 1.12921 | 0.59702 | 0.66224 | 0.52982 | 1.40863 |
| 12 | -42c | 1 | 1.20397 | 1.16731 | 1.14526 | 1.27297 | 0.75342 | 0.42783 | 0.52819 | 0.41217 | 1.03559 |
| 12 | -41c | 1 | 1.56065 | 1.47704 | 1.42920 | 1.66073 | 1.07457 | 0.36459 | 0.30207 | 0.42434 | 1.28997 |
| 12 | -40c | 1 | 0.91629 | 0.90376 | 0.89604 | 0.96758 | 0.53368 | 0.59125 | 0.68017 | 0.71381 | 0.84014 |
| 12 | -39c | 1 | 1.38629 | 1.32817 | 1.29403 | 1.46968 | 0.94414 | 0.63301 | 0.40833 | 0.51958 | 1.16289 |
| 12 | -38c | 2 | 0.70735 | 0.69899 | 0.69443 | 0.73484 | 0.53746 | 0.58850 | 0.61718 | 0.58376 | 0.63026 |
| 12 | -37c | 1 | 0.22314 | 0.24743 | 0.26253 | 0.19845 | 0.37160 | 0.55645 | 0.63278 | 0.59131 | 0.30833 |
| 12 | -36c | 3 | 1.04689 | 1.00436 | 0.98068 | 1.10440 | 0.79617 | 0.59034 | 0.55454 | 0.58601 | 0.89581 |
| 12 | -34c | 1 | 0.99425 | 0.97623 | 0.96518 | 1.04982 | 0.65181 | 0.67141 | 0.51150 | 0.49128 | 0.89293 |
| 12 | -33c | 1 | 0.31471 | 0.33324 | 0.34484 | 0.28768 | 0.49123 | 0.59342 | 0.61976 | 0.62135 | 0.40051 |
| 12 | -32c | 3 | 1.21913 | 1.13671 | 1.09544 | 1.29845 | 1.06604 | 0.84912 | 0.79098 | 0.82928 | 1.10512 |
| 12 | -30c | 5 | 1.47991 | 1.34732 | 1.28296 | 1.61707 | 1.19225 | 0.83238 | 0.72900 | 0.67709 | 1.31815 |
| 12 | -29c | 1 | 0.08338 | 0.11596 | 0.13619 | 0.06188 | 0.12959 | 0.29568 | 0.40446 | 0.39694 | 0.11207 |
| 12 | -28c | 5 | 0.51119 | 0.51987 | 0.52532 | 0.51480 | 0.48107 | 0.54981 | 0.49551 | 0.53898 | 0.52956 |
| 12 | -27c | 1 | 0.16252 | 0.19086 | 0.20838 | 0.13926 | 0.23079 | 0.39717 | 0.39723 | 0.46275 | 0.23603 |
| 12 | -26c | 2 | 0.37486 | 0.39030 | 0.39997 | 0.37783 | 0.34443 | 0.60974 | 0.70270 | 0.62355 | 0.44523 |
| 12 | -25c | 7 | 0.33647 | 0.35467 | 0.36599 | 0.33983 | 0.29115 | 0.39828 | 0.32995 | 0.38926 | 0.37434 |
| 12 | -24c | 7 | 0.86833 | 0.85354 | 0.84483 | 0.87762 | 0.89898 | 0.75567 | 0.72537 | 0.75301 | 0.80812 |
| 12 | -23c | 3 | 0.20771 | 0.23307 | 0.24881 | 0.20092 | 0.21121 | 0.33502 | 0.27960 | 0.31814 | 0.28493 |
| 12 | -22c | 8 | 0.93996 | 0.89818 | 0.87682 | 0.98677 | 0.86389 | 0.78301 | 0.70822 | 0.70864 | 0.86311 |
| 12 | -21c | 7 | 0.64476 | 0.64393 | 0.64366 | 0.64392 | 0.66380 | 0.64607 | 0.67238 | 0.65241 | 0.63169 |
| 12 | -20c | 7 | 0.80731 | 0.78778 | 0.77731 | 0.83255 | 0.77900 | 0.70592 | 0.72520 | 0.72073 | 0.76608 |
| 12 | -19c | 12 | 0.52321 | 0.52342 | 0.52484 | 0.52658 | 0.52767 | 0.56004 | 0.55494 | 0.56095 | 0.53005 |
| 12 | -18c | 8 | 0.91401 | 0.89899 | 0.88991 | 0.94338 | 0.83709 | 0.81029 | 0.77574 | 0.80737 | 0.84093 |
| 12 | -17c | 14 | 0.74368 | 0.73470 | 0.72971 | 0.75330 | 0.74509 | 0.69335 | 0.76371 | 0.69699 | 0.70805 |
| 12 | -16c | 16 | 0.68257 | 0.67490 | 0.67128 | 0.68911 | 0.69128 | 0.67475 | 0.64467 | 0.66597 | 0.66459 |
| 12 | -15c | 16 | 0.65295 | 0.65144 | 0.65078 | 0.66385 | 0.63096 | 0.63596 | 0.64407 | 0.63424 | 0.65237 |
| 12 | -14c | 28 | 0.65134 | 0.64776 | 0.64624 | 0.67204 | 0.60363 | 0.61846 | 0.60814 | 0.62061 | 0.62637 |
| 12 | -13c | 29 | 0.65063 | 0.64489 | 0.64228 | 0.66605 | 0.63048 | 0.61942 | 0.62619 | 0.62151 | 0.64354 |
| 12 | -12c | 37 | 0.61511 | 0.61555 | 0.61618 | 0.62849 | 0.58739 | 0.60166 | 0.59957 | 0.59335 | 0.61133 |
| 12 | -11c | 53 | 0.52486 | 0.53218 | 0.53684 | 0.51930 | 0.53658 | 0.53325 | 0.53518 | 0.54196 | 0.54246 |
| 12 | -10c | 73 | 0.68898 | 0.67845 | 0.67390 | 0.70384 | 0.68630 | 0.67002 | 0.68463 | 0.67293 | 0.70091 |
| 12 | -9c | 93 | 0.66156 | 0.65763 | 0.65584 | 0.66386 | 0.66983 | 0.66097 | 0.65942 | 0.66247 | 0.64956 |
| 12 | -8c | 144 | 0.70506 | 0.69695 | 0.69283 | 0.70356 | 0.71469 | 0.68959 | 0.68210 | 0.68327 | 0.68764 |
| 12 | -7c | 196 | 0.61037 | 0.61059 | 0.61125 | 0.61087 | 0.60942 | 0.60535 | 0.59696 | 0.60396 | 0.61294 |
| 12 | -6c | 347 | 0.61919 | 0.61841 | 0.61854 | 0.61911 | 0.61965 | 0.61619 | 0.60689 | 0.61384 | 0.62483 |
| 12 | -5c | 578 | 0.64392 | 0.64172 | 0.64095 | 0.64798 | 0.64604 | 0.64134 | 0.63893 | 0.63956 | 0.64309 |
| 12 | -4c | 1,531 | 0.61267 | 0.61438 | 0.61567 | 0.61519 | 0.61078 | 0.61175 | 0.60830 | 0.61174 | 0.62030 |
| 12 | -3c | 4,548 | 0.62553 | 0.62581 | 0.62633 | 0.62707 | 0.62580 | 0.62500 | 0.62401 | 0.62498 | 0.63084 |
| 12 | -2c | 10,993 | 0.61608 | 0.61668 | 0.61742 | 0.61734 | 0.61601 | 0.61501 | 0.61360 | 0.61482 | 0.62130 |
| 12 | -1c | 29,155 | 0.61370 | 0.61412 | 0.61481 | 0.61537 | 0.61379 | 0.61216 | 0.61037 | 0.61179 | 0.61751 |
| 12 | +0c | 84,964 | 0.60923 | 0.60951 | 0.61020 | 0.60923 | 0.60962 | 0.60776 | 0.60448 | 0.60728 | 0.61339 |
| 12 | +1c | 29,282 | 0.61022 | 0.61093 | 0.61179 | 0.61049 | 0.60996 | 0.60874 | 0.60692 | 0.60845 | 0.61532 |
| 12 | +2c | 10,675 | 0.62346 | 0.62340 | 0.62380 | 0.62418 | 0.62349 | 0.62174 | 0.62042 | 0.62168 | 0.62724 |
| 12 | +3c | 4,507 | 0.62695 | 0.62726 | 0.62777 | 0.62811 | 0.62717 | 0.62565 | 0.62427 | 0.62541 | 0.63087 |
| 12 | +4c | 1,548 | 0.62129 | 0.62195 | 0.62268 | 0.62274 | 0.62112 | 0.62094 | 0.61944 | 0.62075 | 0.62670 |
| 12 | +5c | 607 | 0.62892 | 0.62801 | 0.62800 | 0.63423 | 0.62606 | 0.62448 | 0.61829 | 0.62440 | 0.63575 |
| 12 | +6c | 337 | 0.61161 | 0.61105 | 0.61147 | 0.61724 | 0.60723 | 0.60102 | 0.59565 | 0.60224 | 0.62342 |
| 12 | +7c | 182 | 0.70811 | 0.69598 | 0.69061 | 0.72963 | 0.69381 | 0.66589 | 0.63853 | 0.66564 | 0.70969 |
| 12 | +8c | 106 | 0.67786 | 0.67068 | 0.66748 | 0.67963 | 0.69041 | 0.66460 | 0.64469 | 0.66113 | 0.66924 |
| 12 | +9c | 76 | 0.76136 | 0.73509 | 0.72508 | 0.77579 | 0.74616 | 0.67662 | 0.65634 | 0.68483 | 0.77330 |
| 12 | +10c | 59 | 0.64035 | 0.63947 | 0.63925 | 0.63997 | 0.65068 | 0.64337 | 0.64535 | 0.63922 | 0.63465 |
| 12 | +11c | 51 | 0.70065 | 0.68804 | 0.68322 | 0.72505 | 0.69355 | 0.66388 | 0.65303 | 0.66522 | 0.72452 |
| 12 | +12c | 36 | 0.75476 | 0.72892 | 0.71896 | 0.76674 | 0.77471 | 0.70316 | 0.68785 | 0.70449 | 0.76437 |
| 12 | +13c | 23 | 0.63627 | 0.63618 | 0.63644 | 0.62899 | 0.67819 | 0.67159 | 0.64879 | 0.67069 | 0.62896 |
| 12 | +14c | 26 | 0.70167 | 0.69062 | 0.68562 | 0.70699 | 0.72395 | 0.68442 | 0.69332 | 0.67530 | 0.69808 |
| 12 | +15c | 20 | 0.74756 | 0.73252 | 0.72495 | 0.75956 | 0.74931 | 0.71201 | 0.71982 | 0.69487 | 0.71647 |
| 12 | +16c | 12 | 0.61877 | 0.62054 | 0.62178 | 0.63653 | 0.55871 | 0.59904 | 0.60563 | 0.59493 | 0.62691 |
| 12 | +17c | 15 | 0.64758 | 0.64034 | 0.63717 | 0.64656 | 0.68940 | 0.61113 | 0.56684 | 0.59919 | 0.64419 |
| 12 | +18c | 10 | 0.45978 | 0.47109 | 0.47826 | 0.46515 | 0.42880 | 0.48308 | 0.45469 | 0.51253 | 0.48282 |
| 12 | +19c | 10 | 0.72972 | 0.71936 | 0.71398 | 0.75323 | 0.66168 | 0.64191 | 0.55061 | 0.61923 | 0.71636 |
| 12 | +20c | 10 | 0.76898 | 0.75779 | 0.75145 | 0.78013 | 0.75902 | 0.75493 | 0.82721 | 0.77453 | 0.71940 |
| 12 | +21c | 5 | 1.20292 | 1.13936 | 1.10555 | 1.25980 | 1.07195 | 0.94850 | 1.06806 | 0.93906 | 1.03994 |
| 12 | +22c | 6 | 0.76222 | 0.75608 | 0.75243 | 0.76551 | 0.78897 | 0.79546 | 0.82258 | 0.78262 | 0.73578 |
| 12 | +23c | 5 | 1.72927 | 1.52279 | 1.43763 | 1.92481 | 1.26861 | 0.91651 | 0.92321 | 0.92690 | 1.75581 |
| 12 | +24c | 1 | 1.34707 | 1.29401 | 1.26267 | 1.42712 | 0.95502 | 1.12333 | 1.04536 | 1.09886 | 1.15574 |
| 12 | +25c | 4 | 0.41799 | 0.43151 | 0.43996 | 0.40597 | 0.52266 | 0.45604 | 0.42631 | 0.43098 | 0.47483 |
| 12 | +26c | 4 | 0.47564 | 0.48618 | 0.49277 | 0.47442 | 0.52296 | 0.48315 | 0.40031 | 0.49727 | 0.52163 |
| 12 | +27c | 6 | 0.34555 | 0.36419 | 0.37573 | 0.34462 | 0.33039 | 0.53772 | 0.47938 | 0.57575 | 0.37912 |
| 12 | +28c | 2 | 1.13895 | 1.10442 | 1.08398 | 1.15682 | 1.16781 | 0.90956 | 0.90002 | 0.83811 | 1.00374 |
| 12 | +29c | 5 | 0.97766 | 0.90126 | 0.86761 | 1.05743 | 0.85113 | 0.60800 | 0.57784 | 0.65204 | 0.98285 |
| 12 | +30c | 3 | 1.01119 | 0.96960 | 0.94711 | 1.05792 | 0.83633 | 0.72271 | 0.68560 | 0.68776 | 0.92881 |
| 12 | +31c | 2 | 0.93217 | 0.90809 | 0.89415 | 0.99025 | 0.54604 | 0.69895 | 0.74097 | 0.63678 | 0.86176 |
| 12 | +32c | 1 | 0.57982 | 0.58510 | 0.58841 | 0.61619 | 0.28437 | 0.43068 | 0.39646 | 0.47039 | 0.61729 |
| 12 | +33c | 2 | 0.14880 | 0.17694 | 0.19455 | 0.12577 | 0.31906 | 0.67337 | 0.49001 | 0.62714 | 0.18589 |
| 12 | +34c | 1 | 1.51413 | 1.43780 | 1.39382 | 1.60944 | 0.91237 | 0.96474 | 1.05053 | 0.92399 | 1.27872 |
| 12 | +36c | 1 | 0.43078 | 0.44305 | 0.45075 | 0.40048 | 0.85881 | 0.39893 | 0.37933 | 0.36589 | 0.49066 |
| 12 | +37c | 2 | 0.66364 | 0.66254 | 0.66203 | 0.68129 | 0.55366 | 0.61058 | 0.65885 | 0.63788 | 0.64680 |
| 12 | +38c | 1 | 0.99425 | 0.97623 | 0.96518 | 1.04982 | 0.49195 | 0.86651 | 0.92508 | 0.83478 | 0.90908 |
| 12 | +39c | 1 | 0.73397 | 0.73195 | 0.73069 | 0.69315 | 1.40777 | 0.61231 | 0.62671 | 0.54328 | 0.71519 |
| 12 | +40c | 1 | 0.54473 | 0.55161 | 0.55592 | 0.57982 | 0.21301 | 0.45617 | 0.47445 | 0.50684 | 0.59095 |
| 12 | +41c | 3 | 0.51895 | 0.52713 | 0.53227 | 0.51231 | 0.72672 | 0.46208 | 0.49468 | 0.42510 | 0.56042 |
| 12 | +42c | 1 | 0.54473 | 0.55161 | 0.55592 | 0.51083 | 1.16012 | 0.44894 | 0.38940 | 0.51808 | 0.57917 |
| 12 | +43c | 2 | 2.17140 | 1.95778 | 1.84848 | 2.36650 | 1.22230 | 0.56382 | 0.58291 | 0.51251 | 1.86582 |
| 12 | +44c | 2 | 0.35079 | 0.36736 | 0.37774 | 0.32275 | 0.84519 | 0.34157 | 0.38482 | 0.35868 | 0.42192 |
| 12 | +45c | 2 | 0.63773 | 0.64028 | 0.64188 | 0.64133 | 0.76093 | 0.56696 | 0.64172 | 0.56259 | 0.65312 |
| 12 | +46c | 1 | 0.52763 | 0.53530 | 0.54010 | 0.49430 | 1.21181 | 0.38423 | 0.43594 | 0.32748 | 0.56631 |
| 12 | +47c | 1 | 0.44629 | 0.45779 | 0.46501 | 0.41552 | 1.08318 | 0.29926 | 0.30015 | 0.27168 | 0.50314 |
| 12 | +57c | 1 | 0.30111 | 0.32044 | 0.33254 | 0.27444 | 0.96355 | 0.15224 | 0.27398 | 0.24368 | 0.37808 |
| 12 | +58c | 1 | 1.34707 | 1.29401 | 1.26267 | 1.42712 | 0.46954 | 1.83841 | 1.42263 | 1.47598 | 1.15574 |
| 12 | +62c | 1 | 1.34707 | 1.29401 | 1.26267 | 1.42712 | 0.42800 | 1.47952 | 1.11783 | 1.21571 | 1.15574 |
| 12 | +63c | 1 | 0.32850 | 0.34623 | 0.35733 | 0.30111 | 1.14197 | 0.15538 | 0.29667 | 0.18512 | 0.40336 |
| 11 | -55c | 3 | 1.26158 | 1.18277 | 1.14212 | 1.33697 | 1.01208 | 0.45778 | 0.53696 | 0.38943 | 1.12106 |
| 11 | -48c | 1 | 0.99425 | 0.97623 | 0.96518 | 1.04982 | 0.53116 | 0.77800 | 0.87190 | 0.65415 | 0.89293 |
| 11 | -41c | 1 | 0.75502 | 0.75192 | 0.74999 | 0.71335 | 1.28654 | 0.74775 | 0.74445 | 0.57976 | 0.74352 |
| 11 | -39c | 2 | 0.72408 | 0.72253 | 0.72157 | 0.68364 | 1.21478 | 0.56126 | 0.81528 | 0.56063 | 0.72163 |
| 11 | -38c | 1 | 1.46968 | 1.39998 | 1.35954 | 1.56065 | 1.03204 | 0.54811 | 0.55591 | 0.49271 | 1.22287 |
| 11 | -37c | 2 | 0.28600 | 0.30666 | 0.31952 | 0.25960 | 0.47523 | 0.47853 | 0.43914 | 0.56076 | 0.36481 |
| 11 | -36c | 1 | 0.65393 | 0.65580 | 0.65696 | 0.69315 | 0.36351 | 0.38868 | 0.37098 | 0.45970 | 0.65882 |
| 11 | -35c | 3 | 0.81204 | 0.80458 | 0.80002 | 0.83219 | 0.73505 | 0.50012 | 0.60762 | 0.56193 | 0.77229 |
| 11 | -34c | 3 | 0.41249 | 0.42650 | 0.43523 | 0.42727 | 0.29670 | 0.50760 | 0.50587 | 0.58586 | 0.46679 |
| 11 | -33c | 2 | 0.68482 | 0.68434 | 0.68410 | 0.72622 | 0.41926 | 0.57890 | 0.55451 | 0.59139 | 0.67580 |
| 11 | -32c | 2 | 0.78032 | 0.77561 | 0.77269 | 0.73745 | 1.19522 | 0.61013 | 0.83824 | 0.66250 | 0.76077 |
| 11 | -31c | 3 | 1.32339 | 1.26012 | 1.22494 | 1.37375 | 1.24538 | 0.80937 | 0.81650 | 0.78285 | 1.13959 |
| 11 | -30c | 2 | 1.07279 | 1.04734 | 1.03193 | 1.13361 | 0.76359 | 0.67547 | 0.60212 | 0.66158 | 0.94644 |
| 11 | -29c | 1 | 0.10536 | 0.13709 | 0.15670 | 0.08338 | 0.15901 | 0.41585 | 0.30814 | 0.44923 | 0.15192 |
| 11 | -28c | 3 | 0.95329 | 0.93790 | 0.92846 | 0.90257 | 1.36231 | 0.63593 | 0.73893 | 0.65611 | 0.88085 |
| 11 | -27c | 5 | 0.72878 | 0.71960 | 0.71469 | 0.71961 | 0.85590 | 0.54382 | 0.52838 | 0.54569 | 0.67443 |
| 11 | -26c | 3 | 0.75592 | 0.75192 | 0.74948 | 0.73625 | 0.93165 | 0.56805 | 0.68389 | 0.63120 | 0.73771 |
| 11 | -25c | 7 | 0.68629 | 0.67830 | 0.67431 | 0.69734 | 0.69566 | 0.44685 | 0.50833 | 0.49860 | 0.64803 |
| 11 | -24c | 2 | 0.35160 | 0.36819 | 0.37858 | 0.38050 | 0.22210 | 0.27867 | 0.28532 | 0.27791 | 0.42175 |
| 11 | -23c | 4 | 0.49958 | 0.50855 | 0.51419 | 0.46479 | 0.67338 | 0.59054 | 0.55725 | 0.60881 | 0.51578 |
| 11 | -22c | 7 | 0.61030 | 0.61307 | 0.61511 | 0.63353 | 0.51349 | 0.64709 | 0.61198 | 0.56974 | 0.59626 |
| 11 | -21c | 2 | 0.39637 | 0.41062 | 0.41955 | 0.39925 | 0.36571 | 0.47916 | 0.43050 | 0.46110 | 0.46436 |
| 11 | -20c | 5 | 1.04494 | 1.00615 | 0.98446 | 1.09547 | 0.95178 | 0.76940 | 0.84524 | 0.81337 | 0.93426 |
| 11 | -19c | 9 | 0.81103 | 0.79676 | 0.78871 | 0.82197 | 0.79845 | 0.78511 | 0.74764 | 0.76078 | 0.75067 |
| 11 | -18c | 10 | 0.41139 | 0.42527 | 0.43394 | 0.41835 | 0.37961 | 0.44064 | 0.45891 | 0.43542 | 0.46949 |
| 11 | -17c | 10 | 0.59736 | 0.59525 | 0.59482 | 0.61983 | 0.53580 | 0.54253 | 0.50789 | 0.53804 | 0.58326 |
| 11 | -16c | 13 | 0.50339 | 0.50964 | 0.51387 | 0.50462 | 0.49792 | 0.52953 | 0.52960 | 0.52867 | 0.51987 |
| 11 | -15c | 14 | 0.60658 | 0.60561 | 0.60565 | 0.60279 | 0.63088 | 0.61999 | 0.61215 | 0.62182 | 0.61253 |
| 11 | -14c | 18 | 0.54070 | 0.54484 | 0.54777 | 0.55259 | 0.51579 | 0.53412 | 0.52439 | 0.52818 | 0.55667 |
| 11 | -13c | 24 | 0.54943 | 0.55205 | 0.55433 | 0.54318 | 0.57768 | 0.55688 | 0.55542 | 0.57034 | 0.57702 |
| 11 | -12c | 33 | 0.76808 | 0.75420 | 0.74688 | 0.78262 | 0.76990 | 0.73013 | 0.73148 | 0.73290 | 0.73377 |
| 11 | -11c | 50 | 0.63024 | 0.62959 | 0.62959 | 0.63803 | 0.62456 | 0.62357 | 0.62080 | 0.62067 | 0.62897 |
| 11 | -10c | 57 | 0.63329 | 0.63176 | 0.63130 | 0.63679 | 0.63532 | 0.62906 | 0.62037 | 0.62975 | 0.62470 |
| 11 | -9c | 86 | 0.62104 | 0.62045 | 0.62061 | 0.62727 | 0.61783 | 0.61752 | 0.61372 | 0.61879 | 0.61863 |
| 11 | -8c | 123 | 0.63194 | 0.62816 | 0.62666 | 0.63884 | 0.62811 | 0.62166 | 0.61681 | 0.61664 | 0.62952 |
| 11 | -7c | 148 | 0.58723 | 0.58818 | 0.58924 | 0.59273 | 0.58288 | 0.58145 | 0.58221 | 0.57987 | 0.59217 |
| 11 | -6c | 349 | 0.59421 | 0.59464 | 0.59547 | 0.59964 | 0.59100 | 0.58662 | 0.58678 | 0.58632 | 0.60085 |
| 11 | -5c | 540 | 0.60915 | 0.60941 | 0.61002 | 0.61145 | 0.60926 | 0.60770 | 0.60616 | 0.60722 | 0.61106 |
| 11 | -4c | 1,369 | 0.61686 | 0.61668 | 0.61706 | 0.61939 | 0.61623 | 0.61462 | 0.60977 | 0.61477 | 0.61964 |
| 11 | -3c | 3,998 | 0.61096 | 0.61139 | 0.61210 | 0.61224 | 0.61165 | 0.61034 | 0.60962 | 0.61035 | 0.61637 |
| 11 | -2c | 9,594 | 0.60015 | 0.60069 | 0.60156 | 0.60213 | 0.60010 | 0.59967 | 0.59847 | 0.59937 | 0.60491 |
| 11 | -1c | 26,402 | 0.59619 | 0.59637 | 0.59711 | 0.59854 | 0.59670 | 0.59518 | 0.59365 | 0.59503 | 0.59952 |
| 11 | +0c | 77,590 | 0.58461 | 0.58529 | 0.58636 | 0.58461 | 0.58497 | 0.58357 | 0.58109 | 0.58337 | 0.58900 |
| 11 | +1c | 26,530 | 0.59129 | 0.59178 | 0.59269 | 0.59214 | 0.59171 | 0.59031 | 0.58874 | 0.59001 | 0.59571 |
| 11 | +2c | 9,817 | 0.60258 | 0.60290 | 0.60363 | 0.60412 | 0.60251 | 0.60228 | 0.60135 | 0.60218 | 0.60733 |
| 11 | +3c | 3,960 | 0.61598 | 0.61614 | 0.61666 | 0.61688 | 0.61647 | 0.61506 | 0.61284 | 0.61474 | 0.62034 |
| 11 | +4c | 1,306 | 0.61368 | 0.61385 | 0.61445 | 0.61654 | 0.61286 | 0.61008 | 0.60733 | 0.60992 | 0.62136 |
| 11 | +5c | 558 | 0.61435 | 0.61446 | 0.61501 | 0.61722 | 0.61401 | 0.61371 | 0.61010 | 0.61308 | 0.62019 |
| 11 | +6c | 282 | 0.62933 | 0.62676 | 0.62606 | 0.63014 | 0.63586 | 0.62828 | 0.62505 | 0.62952 | 0.63318 |
| 11 | +7c | 172 | 0.59473 | 0.59693 | 0.59855 | 0.59889 | 0.59153 | 0.59804 | 0.59314 | 0.59756 | 0.60040 |
| 11 | +8c | 124 | 0.64724 | 0.64179 | 0.63959 | 0.65567 | 0.64415 | 0.62310 | 0.62114 | 0.62267 | 0.64508 |
| 11 | +9c | 70 | 0.56447 | 0.56909 | 0.57213 | 0.55764 | 0.58351 | 0.58517 | 0.56739 | 0.58062 | 0.57598 |
| 11 | +10c | 67 | 0.57634 | 0.57909 | 0.58110 | 0.57532 | 0.58351 | 0.58599 | 0.58156 | 0.58638 | 0.58441 |
| 11 | +11c | 37 | 0.65515 | 0.65243 | 0.65118 | 0.65505 | 0.66924 | 0.67264 | 0.65022 | 0.66443 | 0.63528 |
| 11 | +12c | 34 | 0.73212 | 0.71849 | 0.71153 | 0.74085 | 0.73021 | 0.70725 | 0.70782 | 0.71266 | 0.69810 |
| 11 | +13c | 22 | 0.60271 | 0.60360 | 0.60455 | 0.60680 | 0.60120 | 0.62263 | 0.59255 | 0.60823 | 0.59801 |
| 11 | +14c | 21 | 0.55935 | 0.56216 | 0.56429 | 0.57515 | 0.50997 | 0.53765 | 0.56496 | 0.55123 | 0.56964 |
| 11 | +15c | 14 | 0.44032 | 0.45271 | 0.46046 | 0.43463 | 0.46316 | 0.48575 | 0.46709 | 0.47783 | 0.47242 |
| 11 | +16c | 14 | 0.50658 | 0.50036 | 0.49966 | 0.51401 | 0.52443 | 0.49515 | 0.44906 | 0.46818 | 0.53392 |
| 11 | +17c | 11 | 0.45745 | 0.46556 | 0.47105 | 0.44069 | 0.54526 | 0.52417 | 0.46490 | 0.49038 | 0.47009 |
| 11 | +18c | 10 | 0.49536 | 0.50512 | 0.51121 | 0.48437 | 0.55803 | 0.52396 | 0.52642 | 0.52238 | 0.52949 |
| 11 | +19c | 6 | 0.60341 | 0.60036 | 0.59927 | 0.62194 | 0.54049 | 0.60891 | 0.63831 | 0.61639 | 0.58257 |
| 11 | +20c | 5 | 0.50237 | 0.51140 | 0.51706 | 0.50345 | 0.50263 | 0.52944 | 0.45704 | 0.52533 | 0.52520 |
| 11 | +21c | 7 | 0.51505 | 0.51361 | 0.51420 | 0.52336 | 0.49863 | 0.54477 | 0.45808 | 0.51323 | 0.50437 |
| 11 | +22c | 6 | 1.00032 | 0.97160 | 0.95498 | 1.02522 | 0.95620 | 0.84992 | 0.83585 | 0.86293 | 0.90543 |
| 11 | +23c | 4 | 1.17320 | 1.04034 | 0.98542 | 1.34226 | 1.02705 | 0.78842 | 0.82696 | 0.76837 | 1.54577 |
| 11 | +24c | 4 | 0.79141 | 0.74995 | 0.73043 | 0.82087 | 0.79687 | 0.64404 | 0.60217 | 0.60575 | 0.74081 |
| 11 | +25c | 4 | 0.92613 | 0.89221 | 0.87416 | 0.95879 | 0.85429 | 0.74736 | 0.71703 | 0.75876 | 0.86242 |
| 11 | +26c | 5 | 0.55839 | 0.54784 | 0.54384 | 0.57918 | 0.48820 | 0.50640 | 0.49907 | 0.52948 | 0.53354 |
| 11 | +27c | 2 | 0.66364 | 0.66380 | 0.66399 | 0.70488 | 0.38850 | 0.45577 | 0.43210 | 0.51769 | 0.66829 |
| 11 | +28c | 4 | 0.16461 | 0.19210 | 0.20925 | 0.14119 | 0.30891 | 0.53818 | 0.45830 | 0.55273 | 0.20689 |
| 11 | +29c | 1 | 2.81341 | 2.40396 | 2.21714 | 3.21888 | 2.01937 | 0.66053 | 0.63962 | 0.61039 | 2.86239 |
| 11 | +30c | 3 | 0.05163 | 0.08287 | 0.10306 | 0.03413 | 0.12549 | 0.39814 | 0.34299 | 0.45731 | 0.04899 |
| 11 | +31c | 1 | 1.96611 | 1.80384 | 1.71729 | 2.12026 | 1.31680 | 0.63042 | 0.74748 | 0.49385 | 1.65404 |
| 11 | +32c | 3 | 0.95094 | 0.90135 | 0.87620 | 1.01239 | 0.66643 | 0.47046 | 0.57008 | 0.47626 | 0.86882 |
| 11 | +33c | 2 | 0.61894 | 0.62238 | 0.62454 | 0.62240 | 0.67143 | 0.44198 | 0.43644 | 0.48087 | 0.63945 |
| 11 | +34c | 3 | 0.87884 | 0.82125 | 0.79375 | 0.92925 | 0.77532 | 0.45008 | 0.52884 | 0.49196 | 0.80291 |
| 11 | +35c | 3 | 2.28632 | 1.88512 | 1.72380 | 2.62076 | 1.66612 | 0.59556 | 0.73452 | 0.61721 | 2.50098 |
| 11 | +36c | 3 | 0.65872 | 0.65842 | 0.65835 | 0.67968 | 0.53368 | 0.60321 | 0.56191 | 0.63417 | 0.66352 |
| 11 | +39c | 2 | 1.46031 | 1.29281 | 1.21591 | 1.61573 | 1.13126 | 0.36454 | 0.42841 | 0.36615 | 1.43594 |
| 11 | +40c | 1 | 2.52573 | 2.21192 | 2.06102 | 2.81341 | 1.54200 | 0.30107 | 0.33150 | 0.24315 | 2.30802 |
| 11 | +42c | 4 | 2.23776 | 1.89420 | 1.74556 | 2.68021 | 1.42281 | 0.58407 | 0.58162 | 0.45709 | 2.47997 |
| 11 | +45c | 1 | 0.22314 | 0.24743 | 0.26253 | 0.19845 | 0.59973 | 0.30216 | 0.22941 | 0.29762 | 0.29944 |
| 11 | +48c | 1 | 2.12026 | 1.92106 | 1.81778 | 2.30259 | 1.08620 | 0.57568 | 0.51115 | 0.58774 | 1.80478 |
| 11 | +52c | 1 | 0.01005 | 0.03434 | 0.05260 | 0.01005 | 0.07461 | 1.32813 | 1.01220 | 0.85733 | 0.01005 |
| 11 | +55c | 1 | 0.09431 | 0.12652 | 0.14646 | 0.07257 | 0.38454 | 0.64182 | 0.57222 | 0.58533 | 0.12516 |
| 11 | +56c | 1 | 3.91202 | 3.04505 | 2.71593 | 4.60517 | 2.09624 | 0.55482 | 0.64838 | 0.46353 | 4.60517 |
| 10 | -62c | 1 | 0.99425 | 0.97623 | 0.96518 | 1.04982 | 0.42843 | 0.63846 | 0.59062 | 0.93729 | 0.89293 |
| 10 | -42c | 1 | 0.35667 | 0.37281 | 0.38293 | 0.32850 | 0.63470 | 0.46310 | 0.44507 | 0.44171 | 0.43853 |
| 10 | -40c | 1 | 2.30259 | 2.05490 | 1.93077 | 2.52573 | 1.73925 | 1.11457 | 1.07131 | 0.86518 | 1.95940 |
| 10 | -39c | 1 | 0.69315 | 0.69315 | 0.69315 | 0.65393 | 1.16864 | 0.60662 | 0.55718 | 0.40172 | 0.69973 |
| 10 | -38c | 1 | 0.11653 | 0.14769 | 0.16693 | 0.09431 | 0.20429 | 0.38551 | 0.31511 | 0.50439 | 0.17020 |
| 10 | -36c | 4 | 1.24207 | 1.10484 | 1.04726 | 1.41322 | 1.11914 | 0.79957 | 0.77779 | 0.65098 | 1.60578 |
| 10 | -34c | 2 | 0.59372 | 0.59833 | 0.60122 | 0.59078 | 0.68721 | 0.61366 | 0.58017 | 0.51547 | 0.62029 |
| 10 | -33c | 1 | 0.77653 | 0.77230 | 0.76967 | 0.73397 | 1.20317 | 0.76254 | 0.74997 | 0.62057 | 0.75860 |
| 10 | -32c | 1 | 0.69315 | 0.69315 | 0.69315 | 0.65393 | 1.07706 | 0.71768 | 0.76756 | 0.53801 | 0.69973 |
| 10 | -31c | 1 | 0.65393 | 0.65580 | 0.65696 | 0.61619 | 1.00392 | 0.55002 | 0.60374 | 0.56660 | 0.67158 |
| 10 | -30c | 1 | 0.26136 | 0.28316 | 0.29676 | 0.23572 | 0.38704 | 0.57157 | 0.49852 | 0.51520 | 0.34868 |
| 10 | -28c | 2 | 1.33682 | 1.22066 | 1.16341 | 1.46310 | 1.07706 | 0.78202 | 0.92291 | 0.71819 | 1.20013 |
| 10 | -27c | 3 | 0.38276 | 0.39824 | 0.40790 | 0.37776 | 0.38575 | 0.55466 | 0.56529 | 0.51049 | 0.44696 |
| 10 | -26c | 1 | 0.57982 | 0.58510 | 0.58841 | 0.54473 | 0.83302 | 0.55378 | 0.56453 | 0.57308 | 0.61729 |
| 10 | -25c | 6 | 0.78690 | 0.77470 | 0.76781 | 0.81722 | 0.67610 | 0.72583 | 0.76542 | 0.72948 | 0.74331 |
| 10 | -24c | 4 | 0.85968 | 0.84809 | 0.84108 | 0.82581 | 1.10288 | 0.74376 | 0.57092 | 0.65345 | 0.80532 |
| 10 | -23c | 4 | 0.54225 | 0.55008 | 0.55494 | 0.52780 | 0.62862 | 0.59093 | 0.57077 | 0.58474 | 0.57056 |
| 10 | -22c | 7 | 0.51057 | 0.51336 | 0.51587 | 0.53122 | 0.43018 | 0.51746 | 0.52465 | 0.49900 | 0.49738 |
| 10 | -21c | 6 | 0.54577 | 0.55286 | 0.55738 | 0.50975 | 0.73109 | 0.52108 | 0.50894 | 0.55096 | 0.53208 |
| 10 | -20c | 10 | 0.89715 | 0.84208 | 0.81709 | 0.94231 | 0.92516 | 0.73978 | 0.75799 | 0.73326 | 0.87118 |
| 10 | -19c | 10 | 0.57453 | 0.57144 | 0.57076 | 0.59476 | 0.54816 | 0.50994 | 0.51758 | 0.50817 | 0.58815 |
| 10 | -18c | 10 | 0.62562 | 0.62350 | 0.62275 | 0.65102 | 0.55585 | 0.55979 | 0.53110 | 0.55819 | 0.62730 |
| 10 | -17c | 13 | 0.62330 | 0.61268 | 0.60848 | 0.63146 | 0.64701 | 0.61593 | 0.59957 | 0.60472 | 0.63199 |
| 10 | -16c | 16 | 0.51907 | 0.52537 | 0.52949 | 0.52726 | 0.49476 | 0.49804 | 0.48268 | 0.50835 | 0.53376 |
| 10 | -15c | 18 | 0.50586 | 0.49546 | 0.49378 | 0.52332 | 0.50324 | 0.49666 | 0.49221 | 0.51240 | 0.57291 |
| 10 | -14c | 21 | 0.56646 | 0.56745 | 0.56879 | 0.58513 | 0.52231 | 0.54490 | 0.54611 | 0.53991 | 0.57112 |
| 10 | -13c | 23 | 0.60796 | 0.60533 | 0.60481 | 0.62780 | 0.57061 | 0.59187 | 0.55395 | 0.59803 | 0.61449 |
| 10 | -12c | 38 | 0.43792 | 0.45043 | 0.45828 | 0.42768 | 0.45322 | 0.46534 | 0.44856 | 0.46451 | 0.46477 |
| 10 | -11c | 45 | 0.75430 | 0.72929 | 0.71798 | 0.77985 | 0.74527 | 0.71533 | 0.69960 | 0.71076 | 0.76372 |
| 10 | -10c | 63 | 0.66687 | 0.65918 | 0.65562 | 0.68599 | 0.64289 | 0.64308 | 0.63596 | 0.64175 | 0.64935 |
| 10 | -9c | 84 | 0.59711 | 0.59814 | 0.59928 | 0.60727 | 0.58587 | 0.59033 | 0.59077 | 0.59515 | 0.59877 |
| 10 | -8c | 136 | 0.54416 | 0.54937 | 0.55284 | 0.54792 | 0.53598 | 0.54474 | 0.53511 | 0.54177 | 0.55690 |
| 10 | -7c | 176 | 0.61543 | 0.61383 | 0.61353 | 0.61606 | 0.61715 | 0.61403 | 0.61029 | 0.61137 | 0.61423 |
| 10 | -6c | 296 | 0.57314 | 0.57521 | 0.57704 | 0.57325 | 0.57328 | 0.57660 | 0.57195 | 0.57449 | 0.58330 |
| 10 | -5c | 557 | 0.61408 | 0.61330 | 0.61343 | 0.61616 | 0.61714 | 0.61383 | 0.61112 | 0.61374 | 0.61467 |
| 10 | -4c | 1,339 | 0.59317 | 0.59471 | 0.59609 | 0.59291 | 0.59371 | 0.59296 | 0.59082 | 0.59275 | 0.60073 |
| 10 | -3c | 4,011 | 0.59914 | 0.60009 | 0.60113 | 0.59927 | 0.60082 | 0.59886 | 0.59710 | 0.59859 | 0.60402 |
| 10 | -2c | 10,144 | 0.58154 | 0.58305 | 0.58450 | 0.58263 | 0.58191 | 0.58105 | 0.58003 | 0.58087 | 0.58674 |
| 10 | -1c | 27,429 | 0.57365 | 0.57460 | 0.57594 | 0.57504 | 0.57403 | 0.57184 | 0.57044 | 0.57150 | 0.57833 |
| 10 | +0c | 81,884 | 0.56689 | 0.56780 | 0.56922 | 0.56689 | 0.56726 | 0.56400 | 0.56203 | 0.56380 | 0.57228 |
| 10 | +1c | 27,595 | 0.56823 | 0.56986 | 0.57153 | 0.56918 | 0.56821 | 0.56702 | 0.56582 | 0.56687 | 0.57441 |
| 10 | +2c | 10,295 | 0.58898 | 0.58956 | 0.59053 | 0.59041 | 0.58963 | 0.58819 | 0.58673 | 0.58801 | 0.59280 |
| 10 | +3c | 4,065 | 0.60396 | 0.60448 | 0.60527 | 0.60706 | 0.60425 | 0.60210 | 0.60163 | 0.60225 | 0.60744 |
| 10 | +4c | 1,327 | 0.61026 | 0.61003 | 0.61046 | 0.61281 | 0.61009 | 0.60965 | 0.60788 | 0.60972 | 0.61454 |
| 10 | +5c | 537 | 0.61517 | 0.61357 | 0.61343 | 0.62120 | 0.61321 | 0.61094 | 0.61004 | 0.61018 | 0.62172 |
| 10 | +6c | 317 | 0.58947 | 0.59051 | 0.59169 | 0.59288 | 0.58840 | 0.58808 | 0.58030 | 0.58902 | 0.59452 |
| 10 | +7c | 200 | 0.57465 | 0.57608 | 0.57758 | 0.57883 | 0.57272 | 0.56944 | 0.56468 | 0.57050 | 0.58440 |
| 10 | +8c | 132 | 0.59531 | 0.59706 | 0.59849 | 0.60496 | 0.58200 | 0.58696 | 0.58927 | 0.58905 | 0.59773 |
| 10 | +9c | 89 | 0.54926 | 0.55217 | 0.55451 | 0.55201 | 0.54790 | 0.55492 | 0.56346 | 0.55357 | 0.55508 |
| 10 | +10c | 56 | 0.45116 | 0.46226 | 0.46927 | 0.44817 | 0.45525 | 0.46150 | 0.45990 | 0.46247 | 0.48818 |
| 10 | +11c | 48 | 0.65649 | 0.65265 | 0.65116 | 0.66863 | 0.63942 | 0.63261 | 0.63728 | 0.63519 | 0.65335 |
| 10 | +12c | 33 | 0.59898 | 0.60009 | 0.60114 | 0.59844 | 0.60953 | 0.60465 | 0.60635 | 0.60675 | 0.60309 |
| 10 | +13c | 20 | 0.55017 | 0.55520 | 0.55851 | 0.54545 | 0.57544 | 0.57319 | 0.56536 | 0.57764 | 0.56398 |
| 10 | +14c | 22 | 0.49735 | 0.50549 | 0.51077 | 0.49778 | 0.49997 | 0.52539 | 0.51344 | 0.51500 | 0.50542 |
| 10 | +15c | 13 | 0.72899 | 0.71869 | 0.71335 | 0.74335 | 0.70838 | 0.69245 | 0.68231 | 0.68686 | 0.70275 |
| 10 | +16c | 16 | 0.53895 | 0.54416 | 0.54770 | 0.53258 | 0.58573 | 0.57715 | 0.55332 | 0.56630 | 0.55570 |
| 10 | +17c | 14 | 0.50557 | 0.51360 | 0.51887 | 0.50814 | 0.50534 | 0.54429 | 0.53958 | 0.53138 | 0.53128 |
| 10 | +18c | 15 | 0.78766 | 0.77304 | 0.76527 | 0.78372 | 0.87145 | 0.77448 | 0.77740 | 0.77153 | 0.74975 |
| 10 | +19c | 12 | 0.62058 | 0.61982 | 0.61982 | 0.63360 | 0.57176 | 0.63250 | 0.65816 | 0.63878 | 0.59414 |
| 10 | +20c | 10 | 0.59354 | 0.59821 | 0.60114 | 0.58629 | 0.66366 | 0.61495 | 0.61441 | 0.61237 | 0.61268 |
| 10 | +21c | 5 | 0.47331 | 0.48432 | 0.49119 | 0.48009 | 0.44740 | 0.51515 | 0.52201 | 0.52124 | 0.51630 |
| 10 | +22c | 3 | 1.11346 | 1.04932 | 1.01634 | 1.18177 | 0.91821 | 0.72011 | 0.72764 | 0.72784 | 1.01553 |
| 10 | +23c | 7 | 0.45418 | 0.46587 | 0.47319 | 0.44748 | 0.50486 | 0.57599 | 0.60805 | 0.56291 | 0.47713 |
| 10 | +24c | 4 | 1.23556 | 1.16618 | 1.12969 | 1.31121 | 0.98499 | 0.80641 | 0.70498 | 0.81673 | 1.10732 |
| 10 | +25c | 4 | 0.52133 | 0.53011 | 0.53557 | 0.52393 | 0.53099 | 0.58826 | 0.59609 | 0.58877 | 0.54821 |
| 10 | +26c | 2 | 0.54771 | 0.55485 | 0.55933 | 0.55798 | 0.48887 | 0.53347 | 0.50854 | 0.52610 | 0.57101 |
| 10 | +27c | 4 | 0.91959 | 0.87083 | 0.84677 | 0.94319 | 0.99169 | 0.82896 | 0.88064 | 0.79141 | 0.84094 |
| 10 | +28c | 2 | 0.07309 | 0.10494 | 0.12504 | 0.05179 | 0.15738 | 0.40562 | 0.40505 | 0.44325 | 0.07718 |
| 10 | +29c | 2 | 0.56443 | 0.56499 | 0.56649 | 0.59062 | 0.37332 | 0.61593 | 0.69484 | 0.63765 | 0.49900 |
| 10 | +30c | 3 | 0.97890 | 0.94762 | 0.92995 | 1.02579 | 0.72440 | 0.77831 | 0.73458 | 0.68209 | 0.88181 |
| 10 | +32c | 4 | 0.92854 | 0.89836 | 0.88180 | 0.96029 | 0.82770 | 0.74291 | 0.69630 | 0.64992 | 0.85059 |
| 10 | +33c | 1 | 0.27444 | 0.29541 | 0.30851 | 0.24846 | 0.54915 | 0.55887 | 0.61508 | 0.50751 | 0.35243 |
| 10 | +34c | 4 | 1.09524 | 1.02431 | 0.98946 | 1.16003 | 0.95108 | 0.68606 | 0.66966 | 0.66214 | 1.02575 |
| 10 | +35c | 1 | 1.23787 | 1.19761 | 1.17349 | 1.30933 | 0.69212 | 0.65379 | 0.63558 | 0.63234 | 1.07805 |
| 10 | +36c | 1 | 0.34249 | 0.35942 | 0.37003 | 0.31471 | 0.70866 | 0.71262 | 0.75295 | 0.67363 | 0.41591 |
| 10 | +39c | 2 | 1.30184 | 1.18767 | 1.13165 | 1.40008 | 1.03746 | 0.55784 | 0.56529 | 0.51112 | 1.19258 |
| 10 | +43c | 1 | 2.40795 | 2.12995 | 1.99333 | 2.65926 | 1.38785 | 0.45890 | 0.35498 | 0.32470 | 2.14011 |
| 10 | +45c | 1 | 0.15082 | 0.17992 | 0.19791 | 0.12783 | 0.43799 | 0.82916 | 0.93433 | 0.67367 | 0.21230 |
| 10 | +59c | 1 | 0.01005 | 0.03434 | 0.05260 | 0.01005 | 0.09079 | 0.68850 | 1.09139 | 1.69497 | 0.01005 |
| 9 | -68c | 1 | 0.19845 | 0.22439 | 0.24048 | 0.17435 | 0.53897 | 0.30260 | 0.66331 | 0.43150 | 0.28034 |
| 9 | -67c | 1 | 0.22314 | 0.24743 | 0.26253 | 0.19845 | 0.58878 | 0.34476 | 0.75544 | 0.39657 | 0.30833 |
| 9 | -61c | 1 | 2.20727 | 1.98557 | 1.87247 | 2.40795 | 1.34124 | 0.53207 | 0.52459 | 0.33462 | 1.85470 |
| 9 | -58c | 1 | 0.41552 | 0.42855 | 0.43673 | 0.38566 | 0.91120 | 0.33115 | 0.62793 | 0.29971 | 0.48884 |
| 9 | -54c | 3 | 0.40100 | 0.41481 | 0.42348 | 0.37156 | 0.83686 | 0.24268 | 0.55325 | 0.26816 | 0.47629 |
| 9 | -52c | 1 | 0.54473 | 0.55161 | 0.55592 | 0.51083 | 1.08447 | 0.18763 | 0.35706 | 0.15641 | 0.59095 |
| 9 | -51c | 1 | 0.56212 | 0.56821 | 0.57202 | 0.52763 | 1.10355 | 0.38487 | 0.36279 | 0.23897 | 0.60406 |
| 9 | -49c | 2 | 0.53618 | 0.54345 | 0.54801 | 0.50256 | 1.03079 | 0.27128 | 0.39372 | 0.21078 | 0.58444 |
| 9 | -48c | 1 | 0.69315 | 0.69315 | 0.69315 | 0.65393 | 1.29127 | 0.54582 | 0.51634 | 0.30180 | 0.69973 |
| 9 | -47c | 1 | 1.30933 | 1.26090 | 1.23215 | 1.38629 | 0.79162 | 0.74312 | 0.83277 | 0.74314 | 1.10861 |
| 9 | -46c | 2 | 0.44167 | 0.45436 | 0.46227 | 0.41049 | 0.81922 | 0.38788 | 0.46035 | 0.37395 | 0.49660 |
| 9 | -45c | 1 | 1.30933 | 1.26090 | 1.23215 | 1.38629 | 0.81312 | 0.75594 | 0.85279 | 0.74923 | 1.10861 |
| 9 | -42c | 1 | 0.38566 | 0.40024 | 0.40938 | 0.35667 | 0.68469 | 0.50519 | 0.37037 | 0.37078 | 0.46370 |
| 9 | -41c | 3 | 0.52132 | 0.52958 | 0.53475 | 0.48787 | 0.90517 | 0.36651 | 0.46558 | 0.33064 | 0.56762 |
| 9 | -40c | 2 | 0.95359 | 0.93160 | 0.91862 | 0.97421 | 0.94681 | 0.65820 | 0.64228 | 0.61291 | 0.86963 |
| 9 | -38c | 2 | 0.70325 | 0.70275 | 0.70245 | 0.66364 | 1.16857 | 0.50879 | 0.54457 | 0.40337 | 0.70692 |
| 9 | -36c | 1 | 0.73397 | 0.73195 | 0.73069 | 0.69315 | 1.18515 | 0.74538 | 0.66936 | 0.55732 | 0.72869 |
| 9 | -35c | 2 | 0.27409 | 0.29555 | 0.30888 | 0.24797 | 0.44091 | 0.32672 | 0.38304 | 0.38742 | 0.34987 |
| 9 | -34c | 1 | 0.38566 | 0.40024 | 0.40938 | 0.35667 | 0.61023 | 0.32306 | 0.35639 | 0.38675 | 0.46370 |
| 9 | -33c | 1 | 0.77653 | 0.77230 | 0.76967 | 0.73397 | 1.20204 | 0.65517 | 0.71529 | 0.59258 | 0.75860 |
| 9 | -32c | 1 | 0.19845 | 0.22439 | 0.24048 | 0.17435 | 0.30346 | 0.39219 | 0.32644 | 0.40898 | 0.28034 |
| 9 | -31c | 3 | 0.35774 | 0.37423 | 0.38465 | 0.37340 | 0.22824 | 0.45932 | 0.45478 | 0.52237 | 0.38204 |
| 9 | -30c | 2 | 1.13554 | 1.05049 | 1.00784 | 1.21521 | 0.98379 | 0.80502 | 0.70557 | 0.73109 | 0.99306 |
| 9 | -28c | 2 | 0.53940 | 0.54731 | 0.55221 | 0.55342 | 0.43667 | 0.55154 | 0.53202 | 0.60884 | 0.53809 |
| 9 | -27c | 3 | 0.62879 | 0.63085 | 0.63220 | 0.61421 | 0.74160 | 0.62780 | 0.59461 | 0.64346 | 0.64047 |
| 9 | -26c | 6 | 0.83060 | 0.80459 | 0.79131 | 0.83847 | 0.91723 | 0.71247 | 0.67642 | 0.69564 | 0.78078 |
| 9 | -25c | 3 | 0.69616 | 0.69269 | 0.69078 | 0.74010 | 0.50199 | 0.68735 | 0.69565 | 0.64181 | 0.67430 |
| 9 | -24c | 6 | 0.74164 | 0.71758 | 0.70608 | 0.77161 | 0.69988 | 0.67302 | 0.65364 | 0.63663 | 0.70834 |
| 9 | -22c | 5 | 0.58954 | 0.59178 | 0.59342 | 0.59127 | 0.60371 | 0.59549 | 0.60761 | 0.60107 | 0.58941 |
| 9 | -21c | 6 | 0.80200 | 0.78740 | 0.77925 | 0.78664 | 0.93278 | 0.69374 | 0.67407 | 0.74614 | 0.75501 |
| 9 | -20c | 5 | 0.53136 | 0.53286 | 0.53448 | 0.50414 | 0.62646 | 0.59328 | 0.55615 | 0.58910 | 0.53543 |
| 9 | -19c | 5 | 0.72491 | 0.71891 | 0.71554 | 0.72622 | 0.73698 | 0.76833 | 0.73320 | 0.73487 | 0.70103 |
| 9 | -18c | 12 | 0.76926 | 0.75474 | 0.74691 | 0.80076 | 0.70136 | 0.70012 | 0.73510 | 0.71135 | 0.72016 |
| 9 | -17c | 4 | 0.51916 | 0.52578 | 0.53042 | 0.53268 | 0.46411 | 0.56380 | 0.60655 | 0.56340 | 0.51160 |
| 9 | -16c | 12 | 0.63038 | 0.62753 | 0.62644 | 0.64021 | 0.61952 | 0.61841 | 0.61794 | 0.62478 | 0.61909 |
| 9 | -15c | 23 | 0.75024 | 0.70945 | 0.69325 | 0.81230 | 0.73116 | 0.66166 | 0.64695 | 0.65824 | 0.80352 |
| 9 | -14c | 22 | 0.64283 | 0.62929 | 0.62383 | 0.65953 | 0.64572 | 0.60542 | 0.57780 | 0.61002 | 0.64542 |
| 9 | -13c | 22 | 0.66661 | 0.66207 | 0.65984 | 0.64540 | 0.72373 | 0.68917 | 0.67264 | 0.68968 | 0.65643 |
| 9 | -12c | 24 | 0.55408 | 0.55027 | 0.55003 | 0.56122 | 0.55437 | 0.53979 | 0.53637 | 0.54834 | 0.55589 |
| 9 | -11c | 39 | 0.49431 | 0.50236 | 0.50757 | 0.49299 | 0.49279 | 0.50350 | 0.50047 | 0.50014 | 0.51662 |
| 9 | -10c | 72 | 0.64210 | 0.63687 | 0.63456 | 0.64779 | 0.63910 | 0.63149 | 0.62789 | 0.63766 | 0.62722 |
| 9 | -9c | 85 | 0.59190 | 0.59104 | 0.59135 | 0.58931 | 0.60007 | 0.59068 | 0.58651 | 0.59348 | 0.58731 |
| 9 | -8c | 110 | 0.53388 | 0.53665 | 0.53908 | 0.53489 | 0.53491 | 0.53421 | 0.52862 | 0.52968 | 0.54229 |
| 9 | -7c | 194 | 0.54581 | 0.54824 | 0.55047 | 0.54423 | 0.54974 | 0.55008 | 0.54635 | 0.54779 | 0.56030 |
| 9 | -6c | 231 | 0.58665 | 0.58640 | 0.58703 | 0.58555 | 0.59205 | 0.58707 | 0.58132 | 0.58779 | 0.58588 |
| 9 | -5c | 528 | 0.59900 | 0.59779 | 0.59785 | 0.60430 | 0.59849 | 0.59941 | 0.59544 | 0.59763 | 0.59999 |
| 9 | -4c | 1,289 | 0.57385 | 0.57512 | 0.57659 | 0.57296 | 0.57601 | 0.57386 | 0.57307 | 0.57391 | 0.58359 |
| 9 | -3c | 3,710 | 0.59121 | 0.59191 | 0.59290 | 0.59290 | 0.59321 | 0.58976 | 0.58939 | 0.58980 | 0.59252 |
| 9 | -2c | 9,173 | 0.56104 | 0.56244 | 0.56404 | 0.56247 | 0.56176 | 0.56009 | 0.55903 | 0.55995 | 0.56617 |
| 9 | -1c | 25,011 | 0.54182 | 0.54406 | 0.54626 | 0.54299 | 0.54228 | 0.54090 | 0.53973 | 0.54075 | 0.54741 |
| 9 | +0c | 74,596 | 0.53004 | 0.53257 | 0.53503 | 0.53004 | 0.53043 | 0.52886 | 0.52703 | 0.52876 | 0.53617 |
| 9 | +1c | 24,887 | 0.54311 | 0.54513 | 0.54721 | 0.54474 | 0.54335 | 0.54223 | 0.54096 | 0.54214 | 0.54849 |
| 9 | +2c | 9,465 | 0.56093 | 0.56249 | 0.56417 | 0.56316 | 0.56106 | 0.55965 | 0.55863 | 0.55943 | 0.56618 |
| 9 | +3c | 3,696 | 0.57805 | 0.57939 | 0.58081 | 0.58150 | 0.57728 | 0.57621 | 0.57574 | 0.57610 | 0.58403 |
| 9 | +4c | 1,184 | 0.56832 | 0.57053 | 0.57242 | 0.56998 | 0.56775 | 0.56817 | 0.56645 | 0.56861 | 0.57707 |
| 9 | +5c | 488 | 0.58361 | 0.58443 | 0.58556 | 0.58528 | 0.58491 | 0.58223 | 0.58005 | 0.58264 | 0.58844 |
| 9 | +6c | 255 | 0.58646 | 0.58562 | 0.58596 | 0.59343 | 0.58336 | 0.57710 | 0.57277 | 0.57858 | 0.58570 |
| 9 | +7c | 148 | 0.56323 | 0.56593 | 0.56808 | 0.55943 | 0.57406 | 0.57047 | 0.57060 | 0.57042 | 0.56802 |
| 9 | +8c | 124 | 0.53474 | 0.53785 | 0.54038 | 0.54203 | 0.52363 | 0.53025 | 0.52726 | 0.52722 | 0.54663 |
| 9 | +9c | 78 | 0.69753 | 0.68206 | 0.67470 | 0.71084 | 0.69479 | 0.66747 | 0.64672 | 0.66705 | 0.67016 |
| 9 | +10c | 58 | 0.61322 | 0.61255 | 0.61280 | 0.61140 | 0.62932 | 0.61325 | 0.59907 | 0.61413 | 0.61753 |
| 9 | +11c | 41 | 0.57440 | 0.56904 | 0.56743 | 0.57666 | 0.58444 | 0.57112 | 0.55643 | 0.56311 | 0.56324 |
| 9 | +12c | 36 | 0.56840 | 0.57066 | 0.57257 | 0.56558 | 0.59168 | 0.59229 | 0.59832 | 0.58991 | 0.57446 |
| 9 | +13c | 25 | 0.47393 | 0.48153 | 0.48672 | 0.48510 | 0.44074 | 0.47120 | 0.46051 | 0.46331 | 0.50578 |
| 9 | +14c | 16 | 0.60501 | 0.60546 | 0.60613 | 0.62181 | 0.55514 | 0.55854 | 0.57669 | 0.56880 | 0.59801 |
| 9 | +15c | 16 | 0.68564 | 0.68131 | 0.67905 | 0.68964 | 0.68723 | 0.65598 | 0.62482 | 0.65097 | 0.67484 |
| 9 | +16c | 12 | 0.74253 | 0.72548 | 0.71687 | 0.75853 | 0.72964 | 0.71454 | 0.72311 | 0.71034 | 0.69916 |
| 9 | +17c | 10 | 0.94415 | 0.85943 | 0.82793 | 1.00187 | 0.93893 | 0.76189 | 0.78313 | 0.74724 | 0.99874 |
| 9 | +18c | 7 | 0.50491 | 0.51202 | 0.51684 | 0.48121 | 0.64002 | 0.58572 | 0.57711 | 0.58542 | 0.50653 |
| 9 | +19c | 7 | 0.56307 | 0.56965 | 0.57375 | 0.57277 | 0.51839 | 0.56436 | 0.52299 | 0.55638 | 0.57387 |
| 9 | +20c | 8 | 0.87345 | 0.84926 | 0.83589 | 0.90222 | 0.79628 | 0.80176 | 0.81426 | 0.80933 | 0.80675 |
| 9 | +21c | 4 | 0.48757 | 0.49750 | 0.50371 | 0.47449 | 0.58495 | 0.57026 | 0.58935 | 0.55290 | 0.53279 |
| 9 | +22c | 4 | 0.74516 | 0.72134 | 0.70958 | 0.78087 | 0.61538 | 0.61812 | 0.56698 | 0.64906 | 0.68085 |
| 9 | +23c | 7 | 0.77120 | 0.74845 | 0.73728 | 0.78692 | 0.78820 | 0.72025 | 0.71530 | 0.73731 | 0.73517 |
| 9 | +24c | 3 | 0.72409 | 0.69811 | 0.68526 | 0.75296 | 0.64579 | 0.60644 | 0.62953 | 0.58348 | 0.66548 |
| 9 | +25c | 2 | 0.48617 | 0.49576 | 0.50178 | 0.51923 | 0.26772 | 0.33237 | 0.33437 | 0.36933 | 0.54580 |
| 9 | +26c | 2 | 0.32351 | 0.34170 | 0.35307 | 0.29620 | 0.54414 | 0.56537 | 0.55019 | 0.51053 | 0.39637 |
| 9 | +27c | 4 | 1.48673 | 1.26778 | 1.18433 | 1.67621 | 1.14964 | 0.66749 | 0.59613 | 0.62351 | 1.65711 |
| 9 | +28c | 5 | 0.40523 | 0.41974 | 0.42890 | 0.39003 | 0.53568 | 0.59810 | 0.64455 | 0.61418 | 0.41369 |
| 9 | +30c | 2 | 1.07279 | 0.99888 | 0.96144 | 1.14533 | 0.80937 | 0.72192 | 0.70733 | 0.68533 | 0.93444 |
| 9 | +31c | 2 | 0.75457 | 0.74775 | 0.74375 | 0.80098 | 0.41679 | 0.51412 | 0.47266 | 0.47511 | 0.73211 |
| 9 | +34c | 4 | 1.00736 | 0.95682 | 0.93040 | 1.07415 | 0.62959 | 0.54236 | 0.58088 | 0.66885 | 0.88393 |
| 9 | +35c | 3 | 0.45654 | 0.46852 | 0.47597 | 0.45154 | 0.56455 | 0.60766 | 0.57953 | 0.59741 | 0.49988 |
| 9 | +37c | 1 | 0.69315 | 0.69315 | 0.69315 | 0.73397 | 0.31846 | 0.39972 | 0.49110 | 0.38071 | 0.69973 |
| 9 | +38c | 1 | 0.94161 | 0.92737 | 0.91861 | 0.89160 | 1.72613 | 1.34829 | 1.53982 | 1.26393 | 0.85729 |
| 9 | +39c | 3 | 1.33499 | 1.26804 | 1.23088 | 1.42350 | 0.71498 | 0.48202 | 0.49986 | 0.54460 | 1.16461 |
| 9 | +41c | 3 | 1.99082 | 1.80010 | 1.70500 | 2.18104 | 1.13447 | 0.46862 | 0.39959 | 0.44103 | 1.78075 |
| 9 | +46c | 1 | 2.52573 | 2.21192 | 2.06102 | 2.81341 | 1.40539 | 0.39411 | 0.24818 | 0.22740 | 2.30802 |
| 9 | +48c | 1 | 1.20397 | 1.16731 | 1.14526 | 1.27297 | 0.50174 | 0.36177 | 0.38991 | 0.35399 | 1.05425 |
| 9 | +50c | 1 | 1.27297 | 1.22879 | 1.20244 | 1.34707 | 0.51769 | 0.45161 | 0.57034 | 0.40751 | 1.10283 |
| 9 | +52c | 1 | 0.21072 | 0.23583 | 0.25143 | 0.18633 | 0.66798 | 0.91344 | 0.68255 | 0.77665 | 0.28569 |
| 9 | +53c | 1 | 1.27297 | 1.22879 | 1.20244 | 1.34707 | 0.48360 | 0.58146 | 0.79196 | 0.42496 | 1.10283 |
| 9 | +54c | 1 | 1.96611 | 1.80384 | 1.71729 | 2.12026 | 0.87537 | 0.37079 | 0.52222 | 0.33253 | 1.65404 |
| 9 | +55c | 1 | 0.32850 | 0.34623 | 0.35733 | 0.30111 | 0.99378 | 0.46191 | 0.49717 | 0.42085 | 0.40336 |
| 9 | +56c | 1 | 1.56065 | 1.47704 | 1.42920 | 1.66073 | 0.60439 | 0.68256 | 1.17524 | 0.44600 | 1.31410 |
| 9 | +61c | 1 | 2.65926 | 2.30247 | 2.13506 | 2.99573 | 1.17966 | 0.18373 | 0.19735 | 0.10531 | 2.53246 |
| 9 | +90c | 1 | 2.99573 | 2.52000 | 2.30987 | 3.50656 | 0.83602 | 0.24407 | 0.15810 | 0.11293 | 3.45259 |
| 8 | -89c | 1 | 4.60517 | 3.38851 | 2.97115 | 4.60517 | 2.54101 | 0.14270 | 0.18331 | 0.05515 | 4.60517 |
| 8 | -80c | 1 | 2.99573 | 2.52000 | 2.30987 | 3.50656 | 1.63944 | 0.40457 | 0.28020 | 0.10805 | 3.26424 |
| 8 | -66c | 1 | 0.34249 | 0.35942 | 0.37003 | 0.31471 | 0.84861 | 0.36491 | 0.67903 | 0.32040 | 0.42590 |
| 8 | -62c | 1 | 1.83258 | 1.69920 | 1.62633 | 1.96611 | 1.03734 | 0.15876 | 0.41125 | 0.25957 | 1.50456 |
| 8 | -59c | 1 | 1.56065 | 1.47704 | 1.42920 | 1.66073 | 0.86240 | 0.11547 | 0.23940 | 0.29337 | 1.28997 |
| 8 | -47c | 1 | 2.52573 | 2.21192 | 2.06102 | 2.81341 | 1.81409 | 0.22869 | 0.31617 | 0.34101 | 2.24412 |
| 8 | -44c | 1 | 1.10866 | 1.08122 | 1.06455 | 1.17118 | 0.65847 | 0.48267 | 0.63773 | 0.64276 | 0.97048 |
| 8 | -41c | 2 | 1.00124 | 0.95615 | 0.93169 | 1.04697 | 0.86003 | 0.73957 | 0.71687 | 0.71811 | 0.89172 |
| 8 | -38c | 1 | 0.44629 | 0.45779 | 0.46501 | 0.41552 | 0.74668 | 0.65217 | 0.73840 | 0.66109 | 0.51405 |
| 8 | -34c | 1 | 0.52763 | 0.53530 | 0.54010 | 0.49430 | 0.83790 | 0.46433 | 0.51957 | 0.47216 | 0.57794 |
| 8 | -33c | 2 | 0.98592 | 0.95754 | 0.94111 | 1.01248 | 0.95754 | 0.53905 | 0.65407 | 0.64992 | 0.89152 |
| 8 | -32c | 1 | 0.38566 | 0.40024 | 0.40938 | 0.35667 | 0.59123 | 0.55808 | 0.46194 | 0.47802 | 0.46370 |
| 8 | -31c | 1 | 1.46968 | 1.39998 | 1.35954 | 1.56065 | 1.12713 | 0.87074 | 0.97250 | 0.91876 | 1.22287 |
| 8 | -29c | 2 | 0.56273 | 0.56880 | 0.57260 | 0.59850 | 0.34478 | 0.45036 | 0.54202 | 0.50456 | 0.59235 |
| 8 | -28c | 2 | 1.23138 | 1.14024 | 1.09390 | 1.30865 | 1.13399 | 0.89498 | 0.82061 | 0.79042 | 1.09578 |
| 8 | -27c | 1 | 0.67334 | 0.67430 | 0.67489 | 0.63488 | 0.98105 | 0.75663 | 0.85042 | 0.79116 | 0.68556 |
| 8 | -26c | 3 | 1.33913 | 1.23272 | 1.18130 | 1.43710 | 1.23938 | 0.91803 | 0.87214 | 0.93621 | 1.23456 |
| 8 | -25c | 4 | 0.58198 | 0.58695 | 0.59009 | 0.58122 | 0.60019 | 0.56950 | 0.62598 | 0.57872 | 0.60188 |
| 8 | -24c | 6 | 0.71853 | 0.67826 | 0.66116 | 0.75890 | 0.69688 | 0.52297 | 0.55459 | 0.55710 | 0.71117 |
| 8 | -23c | 2 | 0.76209 | 0.75488 | 0.75064 | 0.71774 | 1.01551 | 0.85693 | 0.82274 | 0.81563 | 0.73839 |
| 8 | -22c | 12 | 0.72307 | 0.70081 | 0.69086 | 0.75610 | 0.65796 | 0.55710 | 0.57926 | 0.61927 | 0.70838 |
| 8 | -21c | 6 | 0.54033 | 0.54662 | 0.55064 | 0.53619 | 0.57374 | 0.46841 | 0.46680 | 0.52398 | 0.56785 |
| 8 | -20c | 5 | 0.61482 | 0.61781 | 0.61986 | 0.63357 | 0.52912 | 0.55618 | 0.55726 | 0.55972 | 0.58700 |
| 8 | -19c | 6 | 0.64159 | 0.63769 | 0.63589 | 0.66268 | 0.59311 | 0.56357 | 0.52967 | 0.57408 | 0.64509 |
| 8 | -18c | 8 | 0.52382 | 0.52501 | 0.52666 | 0.52258 | 0.54186 | 0.54988 | 0.53270 | 0.54370 | 0.54041 |
| 8 | -17c | 12 | 0.35759 | 0.37468 | 0.38532 | 0.34797 | 0.38308 | 0.43961 | 0.43501 | 0.39455 | 0.40825 |
| 8 | -16c | 9 | 0.58463 | 0.54774 | 0.53504 | 0.64957 | 0.52080 | 0.48972 | 0.40865 | 0.47166 | 0.65519 |
| 8 | -15c | 8 | 0.53640 | 0.54058 | 0.54352 | 0.54916 | 0.50838 | 0.53356 | 0.58549 | 0.54039 | 0.53095 |
| 8 | -14c | 18 | 0.52616 | 0.53183 | 0.53581 | 0.52512 | 0.54050 | 0.55840 | 0.58543 | 0.56405 | 0.52175 |
| 8 | -13c | 17 | 0.55567 | 0.56112 | 0.56462 | 0.55464 | 0.56689 | 0.56137 | 0.55838 | 0.56286 | 0.57103 |
| 8 | -12c | 28 | 0.45746 | 0.46691 | 0.47319 | 0.44018 | 0.49643 | 0.50262 | 0.49329 | 0.49017 | 0.47913 |
| 8 | -11c | 42 | 0.47671 | 0.47733 | 0.47990 | 0.47675 | 0.47885 | 0.48053 | 0.46447 | 0.47368 | 0.49936 |
| 8 | -10c | 53 | 0.49684 | 0.50263 | 0.50676 | 0.50023 | 0.49019 | 0.50314 | 0.48949 | 0.48955 | 0.51758 |
| 8 | -9c | 91 | 0.63755 | 0.62159 | 0.61643 | 0.64179 | 0.63182 | 0.61317 | 0.59579 | 0.61649 | 0.62662 |
| 8 | -8c | 127 | 0.59776 | 0.59179 | 0.59036 | 0.60249 | 0.59794 | 0.59334 | 0.57853 | 0.59143 | 0.60595 |
| 8 | -7c | 174 | 0.66558 | 0.65675 | 0.65293 | 0.67631 | 0.66502 | 0.65891 | 0.65802 | 0.66079 | 0.65809 |
| 8 | -6c | 266 | 0.57719 | 0.57468 | 0.57468 | 0.58449 | 0.57550 | 0.57558 | 0.57006 | 0.57392 | 0.58135 |
| 8 | -5c | 490 | 0.58565 | 0.58389 | 0.58394 | 0.59224 | 0.58354 | 0.58584 | 0.58338 | 0.58598 | 0.59183 |
| 8 | -4c | 1,186 | 0.54540 | 0.54867 | 0.55131 | 0.54545 | 0.54540 | 0.54398 | 0.54222 | 0.54325 | 0.55646 |
| 8 | -3c | 3,515 | 0.56391 | 0.56525 | 0.56683 | 0.56560 | 0.56476 | 0.56312 | 0.56164 | 0.56254 | 0.56970 |
| 8 | -2c | 9,306 | 0.53353 | 0.53605 | 0.53845 | 0.53605 | 0.53403 | 0.53204 | 0.53052 | 0.53135 | 0.53921 |
| 8 | -1c | 25,702 | 0.51244 | 0.51518 | 0.51795 | 0.51441 | 0.51238 | 0.51083 | 0.50914 | 0.51051 | 0.51949 |
| 8 | +0c | 76,509 | 0.50384 | 0.50656 | 0.50941 | 0.50384 | 0.50417 | 0.50230 | 0.49992 | 0.50193 | 0.51020 |
| 8 | +1c | 25,485 | 0.51121 | 0.51393 | 0.51670 | 0.51212 | 0.51121 | 0.50963 | 0.50771 | 0.50915 | 0.51845 |
| 8 | +2c | 9,292 | 0.53348 | 0.53580 | 0.53815 | 0.53489 | 0.53383 | 0.53126 | 0.52954 | 0.53104 | 0.54007 |
| 8 | +3c | 3,547 | 0.56726 | 0.56847 | 0.57000 | 0.56865 | 0.56787 | 0.56653 | 0.56505 | 0.56644 | 0.57321 |
| 8 | +4c | 1,142 | 0.56109 | 0.56329 | 0.56531 | 0.56199 | 0.56120 | 0.55999 | 0.55837 | 0.55966 | 0.56971 |
| 8 | +5c | 474 | 0.58540 | 0.58394 | 0.58430 | 0.59035 | 0.58446 | 0.57801 | 0.57944 | 0.57979 | 0.59301 |
| 8 | +6c | 290 | 0.54741 | 0.54938 | 0.55141 | 0.55096 | 0.54436 | 0.55038 | 0.54331 | 0.54797 | 0.55935 |
| 8 | +7c | 154 | 0.52644 | 0.53047 | 0.53355 | 0.52297 | 0.53320 | 0.53038 | 0.51976 | 0.52858 | 0.54082 |
| 8 | +8c | 107 | 0.48537 | 0.49067 | 0.49485 | 0.49010 | 0.47921 | 0.48335 | 0.48425 | 0.47982 | 0.50409 |
| 8 | +9c | 96 | 0.60465 | 0.60238 | 0.60184 | 0.60679 | 0.60886 | 0.61067 | 0.60167 | 0.60771 | 0.60587 |
| 8 | +10c | 63 | 0.58326 | 0.58366 | 0.58450 | 0.58267 | 0.59491 | 0.58433 | 0.56715 | 0.58320 | 0.58732 |
| 8 | +11c | 43 | 0.59377 | 0.58199 | 0.57866 | 0.61622 | 0.59576 | 0.57695 | 0.55729 | 0.56982 | 0.62432 |
| 8 | +12c | 32 | 0.68776 | 0.68208 | 0.67914 | 0.69617 | 0.68155 | 0.67766 | 0.66907 | 0.67863 | 0.66626 |
| 8 | +13c | 35 | 0.51603 | 0.52315 | 0.52773 | 0.51166 | 0.53622 | 0.53374 | 0.53766 | 0.53777 | 0.53145 |
| 8 | +14c | 23 | 0.65715 | 0.64786 | 0.64354 | 0.66914 | 0.64437 | 0.63469 | 0.63269 | 0.63712 | 0.63588 |
| 8 | +15c | 19 | 0.67980 | 0.65392 | 0.64398 | 0.70586 | 0.70881 | 0.64562 | 0.64955 | 0.63526 | 0.75778 |
| 8 | +16c | 14 | 0.95227 | 0.90224 | 0.87873 | 1.01470 | 0.85385 | 0.80506 | 0.76036 | 0.79688 | 0.95277 |
| 8 | +17c | 13 | 0.61362 | 0.61418 | 0.61481 | 0.61255 | 0.64466 | 0.61373 | 0.60742 | 0.59762 | 0.61923 |
| 8 | +18c | 16 | 0.60644 | 0.60408 | 0.60354 | 0.62811 | 0.53727 | 0.57041 | 0.58139 | 0.60073 | 0.60323 |
| 8 | +19c | 10 | 0.83989 | 0.81852 | 0.80686 | 0.84142 | 0.88932 | 0.79192 | 0.76822 | 0.82235 | 0.76462 |
| 8 | +20c | 2 | 1.23255 | 1.12288 | 1.06958 | 1.33250 | 1.04366 | 1.00184 | 1.07381 | 0.91922 | 1.11742 |
| 8 | +21c | 3 | 0.40867 | 0.42226 | 0.43078 | 0.41943 | 0.36114 | 0.46464 | 0.45971 | 0.43981 | 0.47709 |
| 8 | +22c | 6 | 1.02625 | 0.99723 | 0.98028 | 1.05098 | 0.96591 | 0.87879 | 0.97595 | 0.93462 | 0.92718 |
| 8 | +23c | 2 | 0.75706 | 0.75372 | 0.75165 | 0.75525 | 0.82930 | 0.72467 | 0.76631 | 0.77101 | 0.73748 |
| 8 | +24c | 1 | 0.82098 | 0.81430 | 0.81015 | 0.77653 | 1.23477 | 0.90350 | 0.98369 | 1.01459 | 0.77523 |
| 8 | +25c | 7 | 0.64878 | 0.64458 | 0.64262 | 0.67781 | 0.50059 | 0.56582 | 0.59355 | 0.55188 | 0.64590 |
| 8 | +26c | 3 | 0.54690 | 0.55486 | 0.55977 | 0.51172 | 0.86582 | 0.67265 | 0.63634 | 0.66071 | 0.55259 |
| 8 | +27c | 5 | 1.06588 | 1.03151 | 1.01180 | 1.04956 | 1.36650 | 1.06364 | 1.18836 | 1.08677 | 0.95078 |
| 8 | +28c | 1 | 0.12783 | 0.15835 | 0.17720 | 0.10536 | 0.25308 | 0.35004 | 0.38492 | 0.35926 | 0.17974 |
| 8 | +29c | 2 | 0.54622 | 0.55352 | 0.55809 | 0.55737 | 0.47506 | 0.56097 | 0.53726 | 0.54345 | 0.56590 |
| 8 | +30c | 1 | 0.59784 | 0.60230 | 0.60509 | 0.63488 | 0.30879 | 0.47431 | 0.49135 | 0.43692 | 0.63065 |
| 8 | +31c | 1 | 0.34249 | 0.35942 | 0.37003 | 0.31471 | 0.64044 | 0.45597 | 0.37323 | 0.43871 | 0.41591 |
| 8 | +33c | 1 | 0.75502 | 0.75192 | 0.74999 | 0.71335 | 1.31978 | 0.89769 | 0.83291 | 0.77136 | 0.72981 |
| 8 | +34c | 1 | 2.40795 | 2.12995 | 1.99333 | 2.65926 | 1.58494 | 1.04368 | 1.09091 | 1.02687 | 2.14011 |
| 8 | +35c | 1 | 0.09431 | 0.12652 | 0.14646 | 0.07257 | 0.23501 | 0.64690 | 0.51472 | 0.59439 | 0.12516 |
| 8 | +36c | 2 | 0.83037 | 0.80084 | 0.78517 | 0.87263 | 0.54635 | 0.49872 | 0.64531 | 0.58634 | 0.69201 |
| 8 | +37c | 1 | 0.11653 | 0.14769 | 0.16693 | 0.09431 | 0.29399 | 0.46628 | 0.61674 | 0.39855 | 0.16245 |
| 8 | +40c | 2 | 2.17140 | 1.90700 | 1.78627 | 2.46684 | 1.28419 | 0.67087 | 0.69477 | 0.62489 | 2.30416 |
| 8 | +41c | 1 | 1.20397 | 1.16731 | 1.14526 | 1.27297 | 0.58601 | 0.39888 | 0.54185 | 0.53153 | 1.05425 |
| 8 | +42c | 2 | 1.10364 | 1.06005 | 1.03543 | 1.13846 | 1.04763 | 0.79470 | 0.88830 | 0.77148 | 0.98512 |
| 8 | +43c | 1 | 1.66073 | 1.56023 | 1.50366 | 1.77196 | 0.85848 | 1.15934 | 1.20011 | 0.97099 | 1.39228 |
| 8 | +44c | 1 | 0.75502 | 0.75192 | 0.74999 | 0.71335 | 1.56193 | 0.71393 | 0.71975 | 0.54876 | 0.72981 |
| 8 | +49c | 1 | 2.30258 | 2.05490 | 1.93077 | 2.52573 | 1.18703 | 0.31193 | 0.52712 | 0.28574 | 2.00707 |
| 8 | +53c | 1 | 0.59784 | 0.60230 | 0.60509 | 0.56212 | 1.49237 | 0.36413 | 0.56630 | 0.35556 | 0.61839 |
| 8 | +59c | 1 | 2.30258 | 2.05490 | 1.93077 | 2.52573 | 0.99558 | 0.35300 | 0.74794 | 0.26448 | 2.00707 |
| 8 | +66c | 1 | 0.30111 | 0.32044 | 0.33254 | 0.27444 | 1.13730 | 0.11215 | 0.38910 | 0.14227 | 0.37808 |
| 8 | +67c | 1 | 2.99573 | 2.52000 | 2.30987 | 3.50656 | 1.26266 | 0.32359 | 0.34616 | 0.15337 | 3.45259 |
| 8 | +72c | 1 | 0.21072 | 0.23583 | 0.25143 | 0.18633 | 0.99228 | 0.07194 | 0.28048 | 0.10803 | 0.28569 |
| 8 | +73c | 1 | 0.07257 | 0.10537 | 0.12583 | 0.05129 | 0.48815 | 0.12057 | 0.80006 | 0.30192 | 0.08280 |
| 7 | -55c | 1 | 1.89712 | 1.75014 | 1.67076 | 2.04022 | 1.18149 | 0.30555 | 0.41122 | 0.32678 | 1.55947 |
| 7 | -47c | 1 | 1.89712 | 1.75014 | 1.67076 | 2.04022 | 1.29192 | 0.68068 | 0.57108 | 0.59367 | 1.55947 |
| 7 | -45c | 1 | 1.66073 | 1.56023 | 1.50366 | 1.77196 | 1.11680 | 0.57823 | 0.54738 | 0.54145 | 1.36622 |
| 7 | -42c | 1 | 0.91629 | 0.90376 | 0.89604 | 0.96758 | 0.52253 | 0.41233 | 0.52945 | 0.61408 | 0.84014 |
| 7 | -41c | 1 | 1.38629 | 1.32817 | 1.29403 | 1.46968 | 0.92756 | 0.49084 | 0.55346 | 0.59395 | 1.16289 |
| 7 | -39c | 1 | 0.99425 | 0.97623 | 0.96518 | 1.04982 | 0.61116 | 0.64297 | 0.65987 | 0.70624 | 0.89293 |
| 7 | -38c | 1 | 1.89712 | 1.75014 | 1.67076 | 2.04022 | 1.42226 | 0.86519 | 0.94845 | 0.86829 | 1.55947 |
| 7 | -37c | 1 | 1.42712 | 1.36347 | 1.32631 | 1.51413 | 1.01234 | 0.70768 | 0.77971 | 0.70133 | 1.19209 |
| 7 | -36c | 2 | 0.88598 | 0.87357 | 0.86601 | 0.93670 | 0.55213 | 0.59175 | 0.63890 | 0.53785 | 0.81840 |
| 7 | -31c | 3 | 0.24479 | 0.26528 | 0.27890 | 0.22171 | 0.38225 | 0.38868 | 0.40265 | 0.51416 | 0.23522 |
| 7 | -29c | 5 | 0.31497 | 0.33083 | 0.34160 | 0.32396 | 0.24708 | 0.36502 | 0.37229 | 0.38327 | 0.31334 |
| 7 | -28c | 1 | 0.40048 | 0.41428 | 0.42294 | 0.37106 | 0.57857 | 0.58403 | 0.51394 | 0.52479 | 0.47627 |
| 7 | -27c | 4 | 0.50329 | 0.51111 | 0.51647 | 0.49569 | 0.56907 | 0.63466 | 0.63313 | 0.58560 | 0.50641 |
| 7 | -26c | 2 | 0.87263 | 0.83285 | 0.81224 | 0.92132 | 0.74287 | 0.64254 | 0.71694 | 0.69101 | 0.70934 |
| 7 | -25c | 4 | 0.43251 | 0.44567 | 0.45390 | 0.41774 | 0.49242 | 0.49221 | 0.43727 | 0.48548 | 0.42908 |
| 7 | -24c | 2 | 0.58883 | 0.59355 | 0.59654 | 0.59800 | 0.53620 | 0.62945 | 0.61312 | 0.62349 | 0.60688 |
| 7 | -23c | 3 | 0.21703 | 0.24179 | 0.25716 | 0.19243 | 0.28556 | 0.35126 | 0.29758 | 0.31778 | 0.29550 |
| 7 | -22c | 1 | 0.08338 | 0.11596 | 0.13619 | 0.06188 | 0.11287 | 0.21678 | 0.22939 | 0.22253 | 0.11207 |
| 7 | -21c | 7 | 0.61263 | 0.61051 | 0.60989 | 0.61012 | 0.65966 | 0.60121 | 0.59847 | 0.62797 | 0.60466 |
| 7 | -20c | 7 | 0.26103 | 0.28314 | 0.29694 | 0.26151 | 0.22575 | 0.29271 | 0.25227 | 0.29104 | 0.30924 |
| 7 | -19c | 7 | 0.67928 | 0.67700 | 0.67602 | 0.69461 | 0.64871 | 0.64603 | 0.67010 | 0.66480 | 0.64564 |
| 7 | -18c | 12 | 0.71593 | 0.70705 | 0.70235 | 0.70524 | 0.78170 | 0.73872 | 0.73289 | 0.75080 | 0.67566 |
| 7 | -17c | 12 | 0.47456 | 0.48455 | 0.49084 | 0.47135 | 0.47943 | 0.51041 | 0.53968 | 0.51935 | 0.49393 |
| 7 | -16c | 13 | 0.50124 | 0.50726 | 0.51147 | 0.51641 | 0.46092 | 0.47857 | 0.48794 | 0.47118 | 0.51618 |
| 7 | -15c | 21 | 0.57121 | 0.56957 | 0.56972 | 0.57881 | 0.57293 | 0.56188 | 0.55618 | 0.56218 | 0.57894 |
| 7 | -14c | 15 | 0.49192 | 0.49960 | 0.50468 | 0.49291 | 0.49443 | 0.50489 | 0.51027 | 0.49901 | 0.51159 |
| 7 | -13c | 20 | 0.60099 | 0.60033 | 0.60063 | 0.58746 | 0.65576 | 0.62610 | 0.65101 | 0.64054 | 0.59123 |
| 7 | -12c | 30 | 0.54704 | 0.54956 | 0.55182 | 0.56272 | 0.50784 | 0.52770 | 0.50442 | 0.52065 | 0.55850 |
| 7 | -11c | 33 | 0.53579 | 0.54273 | 0.54713 | 0.53883 | 0.53087 | 0.54120 | 0.54358 | 0.53950 | 0.54860 |
| 7 | -10c | 54 | 0.76145 | 0.74001 | 0.72954 | 0.77765 | 0.76070 | 0.72922 | 0.74903 | 0.73374 | 0.72301 |
| 7 | -9c | 72 | 0.48621 | 0.49336 | 0.49822 | 0.48445 | 0.48440 | 0.49887 | 0.49628 | 0.49653 | 0.50160 |
| 7 | -8c | 117 | 0.58250 | 0.58151 | 0.58176 | 0.57399 | 0.59609 | 0.58639 | 0.58609 | 0.59002 | 0.58037 |
| 7 | -7c | 180 | 0.57435 | 0.57535 | 0.57654 | 0.57053 | 0.58533 | 0.57655 | 0.57574 | 0.57941 | 0.57052 |
| 7 | -6c | 270 | 0.51198 | 0.51820 | 0.52243 | 0.51383 | 0.50890 | 0.51442 | 0.51258 | 0.51389 | 0.52788 |
| 7 | -5c | 413 | 0.55996 | 0.56083 | 0.56224 | 0.56566 | 0.55816 | 0.55628 | 0.55840 | 0.55663 | 0.56032 |
| 7 | -4c | 1,093 | 0.55246 | 0.55535 | 0.55771 | 0.55343 | 0.55386 | 0.55063 | 0.54779 | 0.55016 | 0.55763 |
| 7 | -3c | 3,230 | 0.55060 | 0.55301 | 0.55520 | 0.55209 | 0.55175 | 0.54865 | 0.54637 | 0.54767 | 0.55497 |
| 7 | -2c | 8,752 | 0.51175 | 0.51436 | 0.51707 | 0.51363 | 0.51201 | 0.51120 | 0.51031 | 0.51096 | 0.51960 |
| 7 | -1c | 23,901 | 0.49377 | 0.49660 | 0.49963 | 0.49561 | 0.49451 | 0.49232 | 0.49092 | 0.49210 | 0.49827 |
| 7 | +0c | 73,727 | 0.48106 | 0.48361 | 0.48675 | 0.48106 | 0.48153 | 0.47885 | 0.47603 | 0.47917 | 0.48719 |
| 7 | +1c | 24,388 | 0.48944 | 0.49240 | 0.49557 | 0.49157 | 0.48961 | 0.48814 | 0.48658 | 0.48793 | 0.49633 |
| 7 | +2c | 8,799 | 0.52092 | 0.52348 | 0.52603 | 0.52314 | 0.52143 | 0.51827 | 0.51640 | 0.51801 | 0.52594 |
| 7 | +3c | 3,352 | 0.56329 | 0.56396 | 0.56526 | 0.56688 | 0.56356 | 0.56174 | 0.55909 | 0.56087 | 0.56645 |
| 7 | +4c | 1,116 | 0.57509 | 0.57608 | 0.57733 | 0.57630 | 0.57704 | 0.57602 | 0.57214 | 0.57609 | 0.57593 |
| 7 | +5c | 447 | 0.55024 | 0.55092 | 0.55238 | 0.55561 | 0.54796 | 0.54740 | 0.54461 | 0.54657 | 0.55995 |
| 7 | +6c | 251 | 0.54567 | 0.54822 | 0.55070 | 0.54754 | 0.54946 | 0.54786 | 0.54779 | 0.54887 | 0.56179 |
| 7 | +7c | 181 | 0.58807 | 0.58451 | 0.58384 | 0.59605 | 0.58261 | 0.58700 | 0.58440 | 0.58478 | 0.59518 |
| 7 | +8c | 120 | 0.53209 | 0.53139 | 0.53278 | 0.54320 | 0.53052 | 0.53310 | 0.52514 | 0.52947 | 0.54434 |
| 7 | +9c | 67 | 0.53252 | 0.53647 | 0.53946 | 0.54011 | 0.52181 | 0.52717 | 0.51815 | 0.52367 | 0.54340 |
| 7 | +10c | 61 | 0.65038 | 0.64043 | 0.63623 | 0.66935 | 0.63024 | 0.62999 | 0.63466 | 0.62934 | 0.63792 |
| 7 | +11c | 48 | 0.58190 | 0.58358 | 0.58509 | 0.60105 | 0.53703 | 0.55282 | 0.55248 | 0.55561 | 0.58634 |
| 7 | +12c | 36 | 0.61169 | 0.60290 | 0.59946 | 0.63214 | 0.57719 | 0.57741 | 0.55453 | 0.57760 | 0.60703 |
| 7 | +13c | 31 | 0.52780 | 0.53090 | 0.53368 | 0.53192 | 0.53106 | 0.53379 | 0.52983 | 0.52706 | 0.54628 |
| 7 | +14c | 14 | 0.82500 | 0.79666 | 0.78320 | 0.86580 | 0.74853 | 0.70322 | 0.68783 | 0.72355 | 0.81112 |
| 7 | +15c | 16 | 0.47481 | 0.48361 | 0.48932 | 0.47019 | 0.50300 | 0.51523 | 0.50919 | 0.50431 | 0.49998 |
| 7 | +16c | 11 | 0.29586 | 0.31620 | 0.32889 | 0.28653 | 0.32401 | 0.36977 | 0.37075 | 0.35289 | 0.34248 |
| 7 | +17c | 7 | 0.58484 | 0.56296 | 0.55455 | 0.59824 | 0.61874 | 0.52992 | 0.51643 | 0.51406 | 0.60538 |
| 7 | +18c | 12 | 0.82750 | 0.78994 | 0.77241 | 0.84849 | 0.87811 | 0.75777 | 0.72819 | 0.73820 | 0.80881 |
| 7 | +19c | 8 | 0.60878 | 0.60696 | 0.60648 | 0.62868 | 0.53915 | 0.59364 | 0.60175 | 0.60682 | 0.59610 |
| 7 | +20c | 2 | 0.20293 | 0.22450 | 0.23891 | 0.18336 | 0.30553 | 0.29476 | 0.22556 | 0.32484 | 0.23168 |
| 7 | +21c | 8 | 0.48307 | 0.47952 | 0.47962 | 0.49003 | 0.48874 | 0.49661 | 0.46745 | 0.46985 | 0.49010 |
| 7 | +22c | 4 | 1.22352 | 1.11633 | 1.06702 | 1.35905 | 0.89329 | 0.68262 | 0.67746 | 0.74795 | 1.22962 |
| 7 | +23c | 4 | 0.68751 | 0.68415 | 0.68234 | 0.66095 | 0.92268 | 0.79969 | 0.80847 | 0.77397 | 0.67336 |
| 7 | +24c | 9 | 0.68229 | 0.64917 | 0.63611 | 0.72664 | 0.63505 | 0.54689 | 0.56234 | 0.54560 | 0.73236 |
| 7 | +25c | 1 | 0.49430 | 0.50351 | 0.50928 | 0.52763 | 0.27278 | 0.32619 | 0.30281 | 0.33631 | 0.55218 |
| 7 | +26c | 1 | 0.01005 | 0.03434 | 0.05260 | 0.01005 | 0.03622 | 0.08680 | 0.19206 | 0.17835 | 0.01005 |
| 7 | +27c | 3 | 0.53343 | 0.54083 | 0.54547 | 0.56812 | 0.28601 | 0.36669 | 0.33053 | 0.36465 | 0.58231 |
| 7 | +28c | 7 | 0.70293 | 0.69506 | 0.69109 | 0.71048 | 0.73630 | 0.72818 | 0.79856 | 0.77289 | 0.66541 |
| 7 | +29c | 3 | 0.31458 | 0.33225 | 0.34367 | 0.32923 | 0.17725 | 0.32789 | 0.30547 | 0.29234 | 0.35056 |
| 7 | +30c | 1 | 0.73397 | 0.73195 | 0.73069 | 0.69315 | 1.22499 | 0.93388 | 1.05171 | 0.97656 | 0.71519 |
| 7 | +31c | 1 | 0.49430 | 0.50351 | 0.50928 | 0.52763 | 0.23410 | 0.30930 | 0.28295 | 0.28732 | 0.55218 |
| 7 | +34c | 1 | 0.32850 | 0.34623 | 0.35733 | 0.30111 | 0.65919 | 0.51330 | 0.54765 | 0.39989 | 0.40336 |
| 7 | +37c | 1 | 0.91629 | 0.90376 | 0.89604 | 0.96758 | 0.44985 | 0.68884 | 0.81514 | 0.68125 | 0.85546 |
| 7 | +39c | 1 | 2.12026 | 1.92106 | 1.81778 | 2.30259 | 1.26037 | 0.89708 | 0.72501 | 0.95831 | 1.80478 |
| 7 | +42c | 1 | 0.84397 | 0.83595 | 0.83099 | 0.89160 | 0.35922 | 0.85295 | 0.76176 | 0.77739 | 0.80553 |
| 7 | +46c | 1 | 0.49430 | 0.50351 | 0.50928 | 0.46204 | 1.15905 | 0.53546 | 0.54536 | 0.45404 | 0.54085 |
| 7 | +47c | 1 | 0.89160 | 0.88067 | 0.87392 | 0.94161 | 0.34209 | 0.79217 | 0.75863 | 0.75646 | 0.83844 |
| 7 | +58c | 1 | 0.41552 | 0.42855 | 0.43673 | 0.38566 | 1.24661 | 0.18440 | 0.40236 | 0.17875 | 0.47820 |
| 7 | +61c | 1 | 0.34249 | 0.35942 | 0.37003 | 0.31471 | 1.14312 | 0.14322 | 0.29343 | 0.16798 | 0.41591 |
| 7 | +62c | 1 | 0.35667 | 0.37281 | 0.38293 | 0.32850 | 1.19640 | 0.16499 | 0.43493 | 0.20545 | 0.42841 |
| 7 | +64c | 1 | 0.31471 | 0.33324 | 0.34484 | 0.28768 | 1.13440 | 0.13093 | 0.36835 | 0.18535 | 0.39076 |
| 6 | -79c | 1 | 2.20727 | 1.98557 | 1.87247 | 2.40795 | 1.09893 | 0.75930 | 0.44403 | 0.40885 | 1.85470 |
| 6 | -56c | 1 | 0.32850 | 0.34623 | 0.35733 | 0.30111 | 0.71071 | 1.16316 | 0.94852 | 0.89460 | 0.41323 |
| 6 | -53c | 2 | 0.42385 | 0.43651 | 0.44445 | 0.39373 | 0.86577 | 0.93332 | 0.87810 | 0.72987 | 0.49520 |
| 6 | -47c | 1 | 0.67334 | 0.67430 | 0.67489 | 0.71335 | 0.31726 | 0.31846 | 0.38049 | 0.31426 | 0.67262 |
| 6 | -42c | 3 | 0.32557 | 0.34463 | 0.35646 | 0.29781 | 0.57572 | 0.76181 | 0.74614 | 0.65377 | 0.39266 |
| 6 | -40c | 1 | 0.63488 | 0.63764 | 0.63936 | 0.67334 | 0.33183 | 0.30631 | 0.37462 | 0.35638 | 0.64519 |
| 6 | -39c | 1 | 0.77653 | 0.77230 | 0.76967 | 0.82098 | 0.43716 | 0.45215 | 0.51777 | 0.54023 | 0.74468 |
| 6 | -38c | 2 | 0.65467 | 0.65648 | 0.65761 | 0.69395 | 0.35585 | 0.47618 | 0.46534 | 0.42394 | 0.65916 |
| 6 | -37c | 1 | 0.79851 | 0.79308 | 0.78972 | 0.84397 | 0.46877 | 0.54900 | 0.64000 | 0.64178 | 0.75982 |
| 6 | -32c | 2 | 0.49551 | 0.50470 | 0.51047 | 0.52893 | 0.28488 | 0.37040 | 0.34948 | 0.38541 | 0.54115 |
| 6 | -31c | 1 | 0.11653 | 0.14769 | 0.16693 | 0.09431 | 0.17909 | 0.32815 | 0.40962 | 0.36990 | 0.17020 |
| 6 | -30c | 3 | 0.27701 | 0.29925 | 0.31299 | 0.27454 | 0.23876 | 0.39237 | 0.41593 | 0.40310 | 0.31931 |
| 6 | -29c | 3 | 0.30945 | 0.32904 | 0.34135 | 0.28170 | 0.46064 | 0.47030 | 0.51387 | 0.53724 | 0.33963 |
| 6 | -28c | 1 | 0.01005 | 0.03434 | 0.05260 | 0.01005 | 0.02565 | 0.20085 | 0.19597 | 0.23635 | 0.01005 |
| 6 | -27c | 4 | 0.24184 | 0.26396 | 0.27803 | 0.23251 | 0.24813 | 0.43581 | 0.47938 | 0.39002 | 0.29214 |
| 6 | -26c | 4 | 0.81896 | 0.79034 | 0.77577 | 0.85439 | 0.73553 | 0.63222 | 0.65365 | 0.69124 | 0.72674 |
| 6 | -25c | 8 | 0.41404 | 0.42847 | 0.43743 | 0.39130 | 0.53365 | 0.53941 | 0.52915 | 0.53144 | 0.45009 |
| 6 | -24c | 3 | 0.14676 | 0.17303 | 0.18999 | 0.14531 | 0.12692 | 0.19760 | 0.21709 | 0.21737 | 0.18424 |
| 6 | -23c | 7 | 0.30954 | 0.32865 | 0.34071 | 0.30910 | 0.28788 | 0.35739 | 0.34266 | 0.34071 | 0.35358 |
| 6 | -22c | 6 | 0.62246 | 0.62503 | 0.62670 | 0.62277 | 0.64163 | 0.61248 | 0.58455 | 0.60287 | 0.63668 |
| 6 | -21c | 7 | 0.53397 | 0.54087 | 0.54525 | 0.51772 | 0.61109 | 0.58564 | 0.55533 | 0.58452 | 0.56518 |
| 6 | -20c | 12 | 0.47405 | 0.48210 | 0.48755 | 0.46438 | 0.50198 | 0.51437 | 0.52148 | 0.52538 | 0.48569 |
| 6 | -19c | 8 | 0.57563 | 0.57195 | 0.57100 | 0.57297 | 0.62674 | 0.61039 | 0.60786 | 0.60609 | 0.57097 |
| 6 | -18c | 7 | 1.45488 | 1.27229 | 1.19641 | 1.61006 | 1.33089 | 1.03315 | 0.95340 | 1.00536 | 1.47735 |
| 6 | -17c | 12 | 0.69819 | 0.67231 | 0.66094 | 0.74343 | 0.61591 | 0.53790 | 0.52034 | 0.54488 | 0.68323 |
| 6 | -16c | 13 | 0.73593 | 0.72413 | 0.71779 | 0.76943 | 0.66039 | 0.61053 | 0.62556 | 0.63895 | 0.69736 |
| 6 | -15c | 28 | 0.47421 | 0.48470 | 0.49130 | 0.46895 | 0.48199 | 0.48936 | 0.49407 | 0.50001 | 0.49622 |
| 6 | -14c | 23 | 0.46308 | 0.47357 | 0.48020 | 0.47270 | 0.42559 | 0.45581 | 0.44216 | 0.44565 | 0.50280 |
| 6 | -13c | 22 | 0.76817 | 0.74850 | 0.73836 | 0.80576 | 0.72502 | 0.67762 | 0.67502 | 0.69476 | 0.71493 |
| 6 | -12c | 30 | 0.63745 | 0.62463 | 0.61917 | 0.66473 | 0.60225 | 0.59550 | 0.57987 | 0.59710 | 0.61611 |
| 6 | -11c | 34 | 0.58725 | 0.58939 | 0.59110 | 0.58201 | 0.60829 | 0.59703 | 0.60375 | 0.60380 | 0.57456 |
| 6 | -10c | 43 | 0.53059 | 0.53148 | 0.53348 | 0.54202 | 0.52034 | 0.52612 | 0.52060 | 0.52065 | 0.54964 |
| 6 | -9c | 73 | 0.51581 | 0.51914 | 0.52209 | 0.51620 | 0.50906 | 0.51873 | 0.50411 | 0.51587 | 0.53101 |
| 6 | -8c | 111 | 0.48272 | 0.48907 | 0.49364 | 0.48479 | 0.47888 | 0.48828 | 0.48565 | 0.48876 | 0.49823 |
| 6 | -7c | 153 | 0.56248 | 0.56325 | 0.56455 | 0.55881 | 0.57010 | 0.56339 | 0.55184 | 0.56329 | 0.56857 |
| 6 | -6c | 256 | 0.58974 | 0.58816 | 0.58814 | 0.59794 | 0.59062 | 0.58582 | 0.58369 | 0.58653 | 0.58536 |
| 6 | -5c | 422 | 0.54652 | 0.54863 | 0.55066 | 0.54919 | 0.54821 | 0.54468 | 0.54245 | 0.54416 | 0.54749 |
| 6 | -4c | 1,067 | 0.53458 | 0.53746 | 0.54000 | 0.53803 | 0.53326 | 0.53250 | 0.52892 | 0.53228 | 0.54086 |
| 6 | -3c | 3,274 | 0.55321 | 0.55403 | 0.55551 | 0.55452 | 0.55517 | 0.55212 | 0.55066 | 0.55229 | 0.55748 |
| 6 | -2c | 8,691 | 0.51438 | 0.51593 | 0.51812 | 0.51632 | 0.51615 | 0.51282 | 0.51130 | 0.51271 | 0.51573 |
| 6 | -1c | 24,564 | 0.46391 | 0.46749 | 0.47122 | 0.46498 | 0.46467 | 0.46255 | 0.46114 | 0.46249 | 0.46854 |
| 6 | +0c | 78,387 | 0.44141 | 0.44547 | 0.44972 | 0.44141 | 0.44192 | 0.43965 | 0.43773 | 0.43989 | 0.44638 |
| 6 | +1c | 24,658 | 0.46613 | 0.46937 | 0.47294 | 0.46801 | 0.46648 | 0.46489 | 0.46312 | 0.46470 | 0.47110 |
| 6 | +2c | 8,823 | 0.50072 | 0.50348 | 0.50638 | 0.50229 | 0.50122 | 0.49954 | 0.49854 | 0.49987 | 0.50625 |
| 6 | +3c | 3,310 | 0.54260 | 0.54405 | 0.54592 | 0.54350 | 0.54324 | 0.54168 | 0.53959 | 0.54112 | 0.54910 |
| 6 | +4c | 1,067 | 0.56472 | 0.56584 | 0.56731 | 0.56806 | 0.56448 | 0.56171 | 0.56063 | 0.56139 | 0.56841 |
| 6 | +5c | 456 | 0.56970 | 0.56679 | 0.56674 | 0.57777 | 0.56699 | 0.55973 | 0.55520 | 0.56096 | 0.57838 |
| 6 | +6c | 273 | 0.55791 | 0.55816 | 0.55940 | 0.56192 | 0.55935 | 0.55463 | 0.54811 | 0.55176 | 0.56499 |
| 6 | +7c | 142 | 0.58884 | 0.57544 | 0.57187 | 0.60112 | 0.58254 | 0.56260 | 0.55773 | 0.56765 | 0.59921 |
| 6 | +8c | 127 | 0.52340 | 0.52410 | 0.52646 | 0.52981 | 0.52043 | 0.51810 | 0.51306 | 0.51733 | 0.54167 |
| 6 | +9c | 82 | 0.51820 | 0.52528 | 0.52988 | 0.51998 | 0.51603 | 0.51987 | 0.51194 | 0.51615 | 0.53411 |
| 6 | +10c | 63 | 0.51769 | 0.52312 | 0.52685 | 0.51729 | 0.52197 | 0.52689 | 0.51301 | 0.52066 | 0.52962 |
| 6 | +11c | 41 | 0.60213 | 0.60184 | 0.60218 | 0.59930 | 0.62611 | 0.62915 | 0.62070 | 0.63168 | 0.58815 |
| 6 | +12c | 26 | 0.52282 | 0.52843 | 0.53226 | 0.51983 | 0.53779 | 0.55010 | 0.56855 | 0.54149 | 0.53477 |
| 6 | +13c | 18 | 0.49600 | 0.50141 | 0.50539 | 0.51825 | 0.42252 | 0.47204 | 0.45567 | 0.46033 | 0.50200 |
| 6 | +14c | 17 | 0.51058 | 0.51564 | 0.51934 | 0.51896 | 0.47546 | 0.48741 | 0.48755 | 0.49026 | 0.50748 |
| 6 | +15c | 16 | 0.76097 | 0.69319 | 0.67229 | 0.76004 | 0.70997 | 0.63572 | 0.59444 | 0.64975 | 0.78570 |
| 6 | +16c | 7 | 0.96021 | 0.91138 | 0.88762 | 1.00851 | 0.88980 | 0.77013 | 0.71593 | 0.76393 | 0.92031 |
| 6 | +17c | 7 | 0.65187 | 0.65032 | 0.64964 | 0.64906 | 0.68447 | 0.61733 | 0.66139 | 0.63101 | 0.63319 |
| 6 | +18c | 8 | 0.72776 | 0.72222 | 0.71907 | 0.73780 | 0.71388 | 0.67300 | 0.69116 | 0.68241 | 0.71143 |
| 6 | +19c | 8 | 0.66944 | 0.66491 | 0.66255 | 0.68373 | 0.63181 | 0.62950 | 0.61699 | 0.64991 | 0.64535 |
| 6 | +20c | 10 | 0.44022 | 0.45318 | 0.46128 | 0.43395 | 0.47040 | 0.47937 | 0.45816 | 0.48748 | 0.44681 |
| 6 | +21c | 9 | 0.53714 | 0.54026 | 0.54275 | 0.54492 | 0.52179 | 0.53081 | 0.53983 | 0.54178 | 0.54272 |
| 6 | +22c | 2 | 0.42199 | 0.43654 | 0.44553 | 0.43090 | 0.35231 | 0.39748 | 0.33599 | 0.37952 | 0.45601 |
| 6 | +23c | 5 | 0.46164 | 0.47365 | 0.48111 | 0.47624 | 0.36258 | 0.42031 | 0.40290 | 0.41396 | 0.49503 |
| 6 | +24c | 2 | 0.99645 | 0.97476 | 0.96174 | 1.01134 | 0.99879 | 0.91045 | 0.99695 | 0.99414 | 0.90393 |
| 6 | +25c | 4 | 0.40992 | 0.42421 | 0.43312 | 0.41634 | 0.37157 | 0.43533 | 0.43807 | 0.41410 | 0.45905 |
| 6 | +26c | 1 | 1.66073 | 1.56023 | 1.50366 | 1.77196 | 1.16773 | 1.28479 | 1.44490 | 1.27711 | 1.39228 |
| 6 | +27c | 3 | 0.32157 | 0.33989 | 0.35135 | 0.33273 | 0.25842 | 0.27263 | 0.22884 | 0.26088 | 0.40082 |
| 6 | +29c | 4 | 0.41832 | 0.43219 | 0.44084 | 0.40252 | 0.58894 | 0.49249 | 0.45971 | 0.48452 | 0.45128 |
| 6 | +30c | 2 | 0.70325 | 0.70275 | 0.70245 | 0.66364 | 1.17762 | 0.87836 | 0.96755 | 0.96552 | 0.69370 |
| 6 | +32c | 1 | 0.40048 | 0.41428 | 0.42294 | 0.43078 | 0.17376 | 0.20571 | 0.15465 | 0.15129 | 0.47627 |
| 6 | +33c | 4 | 0.87095 | 0.83551 | 0.81695 | 0.91664 | 0.62723 | 0.72521 | 0.78912 | 0.76240 | 0.75946 |
| 6 | +34c | 1 | 0.16252 | 0.19086 | 0.20838 | 0.13926 | 0.36118 | 0.44829 | 0.30215 | 0.37326 | 0.22776 |
| 6 | +35c | 1 | 2.04022 | 1.86066 | 1.76619 | 2.20727 | 1.28130 | 1.10251 | 1.17121 | 1.12998 | 1.72457 |
| 6 | +36c | 3 | 0.32123 | 0.34117 | 0.35349 | 0.29317 | 0.64753 | 0.59939 | 0.53016 | 0.54251 | 0.36004 |
| 6 | +38c | 2 | 0.57601 | 0.58147 | 0.58491 | 0.58915 | 0.49070 | 0.56139 | 0.63099 | 0.57862 | 0.58421 |
| 6 | +48c | 1 | 1.30933 | 1.26090 | 1.23215 | 1.38629 | 0.56038 | 1.06915 | 0.98690 | 1.15516 | 1.12869 |
| 5 | -65c | 1 | 0.27444 | 0.29541 | 0.30851 | 0.24846 | 0.68379 | 0.93258 | 1.08808 | 0.91729 | 0.36181 |
| 5 | -63c | 1 | 0.31471 | 0.33324 | 0.34484 | 0.28768 | 0.75216 | 0.98490 | 0.96663 | 0.83991 | 0.40051 |
| 5 | -49c | 1 | 1.38629 | 1.32817 | 1.29403 | 1.46968 | 0.83867 | 0.54593 | 0.47064 | 0.51150 | 1.16289 |
| 5 | -48c | 1 | 0.10536 | 0.13709 | 0.15670 | 0.08338 | 0.22027 | 0.81380 | 0.88182 | 0.79250 | 0.15192 |
| 5 | -46c | 1 | 1.38629 | 1.32817 | 1.29403 | 1.46968 | 0.87228 | 0.65023 | 0.49224 | 0.52973 | 1.16289 |
| 5 | -44c | 1 | 1.60944 | 1.51780 | 1.46578 | 1.71480 | 1.08763 | 0.59023 | 0.50623 | 0.59024 | 1.32679 |
| 5 | -43c | 1 | 0.31471 | 0.33324 | 0.34484 | 0.28768 | 0.56508 | 0.73334 | 0.78063 | 0.75119 | 0.40051 |
| 5 | -40c | 2 | 2.23270 | 1.95999 | 1.83470 | 2.53360 | 1.66446 | 0.74824 | 0.65084 | 0.78369 | 2.24355 |
| 5 | -38c | 2 | 0.87493 | 0.84849 | 0.83380 | 0.91008 | 0.73737 | 0.73836 | 0.83189 | 0.69689 | 0.78869 |
| 5 | -37c | 1 | 1.46968 | 1.39998 | 1.35954 | 1.56065 | 1.05363 | 0.83875 | 0.70900 | 0.72644 | 1.22287 |
| 5 | -36c | 2 | 0.33824 | 0.35620 | 0.36738 | 0.34259 | 0.28859 | 0.43005 | 0.38645 | 0.42748 | 0.40428 |
| 5 | -35c | 2 | 0.80974 | 0.78745 | 0.77526 | 0.84641 | 0.64759 | 0.64311 | 0.53604 | 0.57241 | 0.70367 |
| 5 | -34c | 1 | 1.56065 | 1.47704 | 1.42920 | 1.66073 | 1.17690 | 0.89872 | 0.64424 | 0.83236 | 1.28997 |
| 5 | -31c | 1 | 0.12783 | 0.15835 | 0.17720 | 0.10536 | 0.19440 | 0.40110 | 0.36722 | 0.38306 | 0.18762 |
| 5 | -30c | 2 | 1.04414 | 1.00617 | 0.98465 | 1.07751 | 1.01065 | 0.80207 | 0.69714 | 0.75513 | 0.93395 |
| 5 | -29c | 3 | 0.75705 | 0.74112 | 0.73258 | 0.79574 | 0.59040 | 0.54277 | 0.54031 | 0.55463 | 0.68477 |
| 5 | -28c | 1 | 0.08338 | 0.11596 | 0.13619 | 0.06188 | 0.12538 | 0.33843 | 0.36990 | 0.33629 | 0.11207 |
| 5 | -27c | 4 | 0.87995 | 0.82352 | 0.79742 | 0.94381 | 0.75254 | 0.69045 | 0.69297 | 0.68545 | 0.81266 |
| 5 | -26c | 2 | 0.40226 | 0.41610 | 0.42477 | 0.43268 | 0.25087 | 0.30257 | 0.32582 | 0.30910 | 0.46578 |
| 5 | -25c | 4 | 0.75427 | 0.74967 | 0.74691 | 0.70998 | 1.03475 | 0.88066 | 0.80832 | 0.89351 | 0.70140 |
| 5 | -24c | 7 | 0.38089 | 0.39550 | 0.40482 | 0.35812 | 0.48828 | 0.50293 | 0.49131 | 0.49255 | 0.40230 |
| 5 | -23c | 5 | 0.33519 | 0.35123 | 0.36185 | 0.35143 | 0.25199 | 0.29479 | 0.29259 | 0.30428 | 0.36869 |
| 5 | -22c | 5 | 0.41999 | 0.43412 | 0.44288 | 0.43348 | 0.33120 | 0.40735 | 0.38928 | 0.39620 | 0.45636 |
| 5 | -21c | 6 | 1.02137 | 0.97640 | 0.95208 | 1.07862 | 0.90187 | 0.82998 | 0.78099 | 0.79566 | 0.89921 |
| 5 | -20c | 6 | 0.99613 | 0.96019 | 0.94004 | 1.00958 | 1.01854 | 0.92569 | 0.89525 | 0.92322 | 0.87994 |
| 5 | -19c | 15 | 0.75167 | 0.73130 | 0.72106 | 0.77438 | 0.73805 | 0.67021 | 0.67252 | 0.69818 | 0.70155 |
| 5 | -18c | 8 | 1.15703 | 1.04378 | 0.99973 | 1.25355 | 1.08421 | 0.89415 | 0.78704 | 0.88682 | 1.22012 |
| 5 | -17c | 18 | 0.66903 | 0.65781 | 0.65237 | 0.68059 | 0.67734 | 0.62441 | 0.61641 | 0.64025 | 0.65109 |
| 5 | -16c | 17 | 0.67141 | 0.66351 | 0.65974 | 0.68609 | 0.65632 | 0.63979 | 0.65495 | 0.63994 | 0.64587 |
| 5 | -15c | 23 | 0.50839 | 0.51051 | 0.51290 | 0.51071 | 0.50867 | 0.52616 | 0.54637 | 0.53753 | 0.51517 |
| 5 | -14c | 28 | 0.61139 | 0.60963 | 0.60922 | 0.62328 | 0.59214 | 0.59972 | 0.61264 | 0.60548 | 0.60410 |
| 5 | -13c | 18 | 0.67248 | 0.65936 | 0.65315 | 0.69903 | 0.65016 | 0.60269 | 0.58354 | 0.61076 | 0.64692 |
| 5 | -12c | 42 | 0.53622 | 0.53720 | 0.53878 | 0.53718 | 0.54058 | 0.53221 | 0.52547 | 0.53600 | 0.53480 |
| 5 | -11c | 46 | 0.64204 | 0.63725 | 0.63516 | 0.64853 | 0.63489 | 0.62888 | 0.62702 | 0.63095 | 0.62300 |
| 5 | -10c | 58 | 0.57730 | 0.57529 | 0.57516 | 0.58602 | 0.57200 | 0.56892 | 0.56805 | 0.57379 | 0.56749 |
| 5 | -9c | 104 | 0.59189 | 0.58737 | 0.58647 | 0.59335 | 0.60685 | 0.59850 | 0.59680 | 0.59526 | 0.60656 |
| 5 | -8c | 133 | 0.51182 | 0.51525 | 0.51824 | 0.51482 | 0.51232 | 0.51566 | 0.52216 | 0.51485 | 0.52110 |
| 5 | -7c | 182 | 0.55586 | 0.55698 | 0.55858 | 0.55401 | 0.56465 | 0.55951 | 0.56364 | 0.56261 | 0.55224 |
| 5 | -6c | 287 | 0.55868 | 0.55829 | 0.55925 | 0.56099 | 0.56054 | 0.55621 | 0.55219 | 0.55609 | 0.56524 |
| 5 | -5c | 504 | 0.51147 | 0.51585 | 0.51925 | 0.51016 | 0.51243 | 0.51016 | 0.50786 | 0.50933 | 0.52167 |
| 5 | -4c | 1,121 | 0.52723 | 0.52871 | 0.53098 | 0.53133 | 0.52584 | 0.52569 | 0.52302 | 0.52498 | 0.54007 |
| 5 | -3c | 3,285 | 0.52837 | 0.53032 | 0.53253 | 0.53003 | 0.52901 | 0.52717 | 0.52611 | 0.52672 | 0.53262 |
| 5 | -2c | 8,908 | 0.47800 | 0.48169 | 0.48527 | 0.48006 | 0.47774 | 0.47655 | 0.47498 | 0.47626 | 0.48517 |
| 5 | -1c | 25,925 | 0.43276 | 0.43776 | 0.44250 | 0.43498 | 0.43279 | 0.43084 | 0.42901 | 0.43059 | 0.43943 |
| 5 | +0c | 86,211 | 0.40300 | 0.40857 | 0.41398 | 0.40300 | 0.40320 | 0.40117 | 0.39912 | 0.40101 | 0.40893 |
| 5 | +1c | 26,484 | 0.43183 | 0.43680 | 0.44154 | 0.43427 | 0.43150 | 0.42953 | 0.42796 | 0.42924 | 0.43899 |
| 5 | +2c | 8,927 | 0.47598 | 0.47958 | 0.48315 | 0.47852 | 0.47542 | 0.47378 | 0.47197 | 0.47309 | 0.48429 |
| 5 | +3c | 3,235 | 0.53625 | 0.53737 | 0.53921 | 0.54098 | 0.53514 | 0.53308 | 0.53074 | 0.53283 | 0.54055 |
| 5 | +4c | 1,088 | 0.52603 | 0.52908 | 0.53182 | 0.53091 | 0.52279 | 0.52458 | 0.52214 | 0.52445 | 0.53787 |
| 5 | +5c | 479 | 0.55776 | 0.55784 | 0.55887 | 0.56234 | 0.55644 | 0.55391 | 0.54657 | 0.55156 | 0.56052 |
| 5 | +6c | 323 | 0.51400 | 0.51630 | 0.51886 | 0.51980 | 0.50884 | 0.50746 | 0.49991 | 0.50480 | 0.52647 |
| 5 | +7c | 172 | 0.50170 | 0.50620 | 0.50978 | 0.50245 | 0.50171 | 0.50710 | 0.49955 | 0.50346 | 0.51192 |
| 5 | +8c | 118 | 0.56054 | 0.55945 | 0.56014 | 0.57404 | 0.54408 | 0.54507 | 0.53632 | 0.54511 | 0.56406 |
| 5 | +9c | 86 | 0.55423 | 0.54797 | 0.54718 | 0.56693 | 0.54377 | 0.53910 | 0.53311 | 0.54697 | 0.56296 |
| 5 | +10c | 70 | 0.64861 | 0.64621 | 0.64528 | 0.65623 | 0.64407 | 0.64700 | 0.64818 | 0.65049 | 0.63822 |
| 5 | +11c | 52 | 0.53001 | 0.53232 | 0.53467 | 0.53116 | 0.53748 | 0.54229 | 0.54035 | 0.53702 | 0.53306 |
| 5 | +12c | 37 | 0.59871 | 0.58777 | 0.58416 | 0.61717 | 0.58189 | 0.57432 | 0.56273 | 0.57588 | 0.61706 |
| 5 | +13c | 32 | 0.53049 | 0.53546 | 0.53890 | 0.53943 | 0.51250 | 0.52307 | 0.52930 | 0.52673 | 0.53634 |
| 5 | +14c | 24 | 0.48406 | 0.49269 | 0.49822 | 0.48103 | 0.50440 | 0.50588 | 0.51085 | 0.51305 | 0.49401 |
| 5 | +15c | 21 | 0.47253 | 0.48319 | 0.48987 | 0.47410 | 0.47261 | 0.49019 | 0.48237 | 0.48687 | 0.50683 |
| 5 | +16c | 9 | 0.48795 | 0.49711 | 0.50290 | 0.48705 | 0.50532 | 0.49247 | 0.49516 | 0.47892 | 0.52423 |
| 5 | +17c | 14 | 0.39013 | 0.40421 | 0.41318 | 0.38988 | 0.39931 | 0.40917 | 0.38057 | 0.39812 | 0.43645 |
| 5 | +18c | 9 | 0.45444 | 0.46413 | 0.47036 | 0.44194 | 0.52306 | 0.49405 | 0.54610 | 0.46962 | 0.48842 |
| 5 | +19c | 7 | 0.63188 | 0.63181 | 0.63200 | 0.64255 | 0.60871 | 0.63260 | 0.64368 | 0.65863 | 0.62082 |
| 5 | +20c | 8 | 0.57224 | 0.57576 | 0.57813 | 0.55859 | 0.67721 | 0.60980 | 0.60897 | 0.63855 | 0.56763 |
| 5 | +21c | 6 | 0.55319 | 0.55127 | 0.55131 | 0.54753 | 0.62364 | 0.57538 | 0.59200 | 0.59871 | 0.50535 |
| 5 | +22c | 8 | 0.91673 | 0.86673 | 0.84422 | 0.98584 | 0.73903 | 0.68107 | 0.71393 | 0.74230 | 0.91031 |
| 5 | +23c | 3 | 0.64421 | 0.64559 | 0.64651 | 0.62682 | 0.82323 | 0.73188 | 0.69073 | 0.73369 | 0.65017 |
| 5 | +24c | 6 | 0.80770 | 0.79625 | 0.78956 | 0.82706 | 0.74571 | 0.75357 | 0.80762 | 0.76869 | 0.76880 |
| 5 | +25c | 7 | 0.38712 | 0.40088 | 0.40993 | 0.40907 | 0.23989 | 0.36731 | 0.37711 | 0.30893 | 0.42038 |
| 5 | +27c | 3 | 0.52941 | 0.53749 | 0.54254 | 0.51926 | 0.65672 | 0.59544 | 0.56317 | 0.58122 | 0.56164 |
| 5 | +28c | 1 | 0.51083 | 0.51927 | 0.52456 | 0.54473 | 0.26265 | 0.31534 | 0.30898 | 0.30042 | 0.56502 |
| 5 | +29c | 1 | 1.17118 | 1.13784 | 1.11772 | 1.10866 | 1.85368 | 1.53815 | 1.51448 | 1.61943 | 1.01311 |
| 5 | +30c | 1 | 0.35667 | 0.37281 | 0.38293 | 0.32850 | 0.65227 | 0.47145 | 0.39296 | 0.43851 | 0.42841 |
| 5 | +31c | 2 | 0.44935 | 0.46086 | 0.46808 | 0.41839 | 0.81333 | 0.51113 | 0.50441 | 0.57839 | 0.50359 |
| 5 | +32c | 1 | 1.38629 | 1.32817 | 1.29403 | 1.46968 | 0.84089 | 1.01007 | 1.04920 | 1.07622 | 1.18411 |
| 5 | +33c | 2 | 0.67740 | 0.67761 | 0.67778 | 0.68482 | 0.69490 | 0.65960 | 0.59547 | 0.60124 | 0.67953 |
| 5 | +35c | 2 | 0.51712 | 0.52613 | 0.53175 | 0.52735 | 0.45189 | 0.56824 | 0.50780 | 0.45451 | 0.54345 |
| 5 | +36c | 4 | 0.55032 | 0.55714 | 0.56151 | 0.56111 | 0.47418 | 0.59994 | 0.56985 | 0.54743 | 0.53972 |
| 5 | +37c | 1 | 0.89160 | 0.88067 | 0.87392 | 0.94161 | 0.43272 | 0.83326 | 0.85008 | 0.68017 | 0.83844 |
| 5 | +38c | 1 | 0.52763 | 0.53530 | 0.54010 | 0.49430 | 1.06475 | 0.55371 | 0.57482 | 0.65379 | 0.56631 |
| 5 | +39c | 1 | 0.16252 | 0.19086 | 0.20838 | 0.13926 | 0.40782 | 0.47810 | 0.35556 | 0.36753 | 0.22776 |
| 5 | +43c | 1 | 0.26136 | 0.28316 | 0.29676 | 0.23572 | 0.65985 | 0.34802 | 0.33020 | 0.32715 | 0.33943 |
| 5 | +45c | 1 | 1.27297 | 1.22879 | 1.20244 | 1.34707 | 0.57516 | 1.11021 | 1.17963 | 1.18533 | 1.10283 |
| 5 | +47c | 1 | 0.26136 | 0.28316 | 0.29676 | 0.23572 | 0.71728 | 0.30711 | 0.28902 | 0.30350 | 0.33943 |
| 5 | +49c | 2 | 1.47826 | 1.40590 | 1.36422 | 1.57096 | 0.64805 | 1.04895 | 1.17633 | 1.14398 | 1.25379 |
| 5 | +58c | 1 | 0.19845 | 0.22439 | 0.24048 | 0.17435 | 0.72808 | 0.16592 | 0.23145 | 0.18556 | 0.27170 |
| 5 | +59c | 1 | 1.96611 | 1.80384 | 1.71729 | 2.12026 | 0.78912 | 1.53513 | 1.38594 | 1.18692 | 1.65404 |
| 5 | +78c | 1 | 2.65926 | 2.30247 | 2.13506 | 2.99573 | 0.86201 | 1.15962 | 0.43765 | 0.94956 | 2.53246 |
| 4 | -84c | 1 | 0.09431 | 0.12652 | 0.14646 | 0.07257 | 0.36840 | 0.16596 | 0.59992 | 0.20271 | 0.13262 |
| 4 | -78c | 1 | 0.22314 | 0.24743 | 0.26253 | 0.19845 | 0.68429 | 0.90279 | 0.95683 | 1.12815 | 0.30833 |
| 4 | -73c | 2 | 0.16844 | 0.19638 | 0.21367 | 0.14504 | 0.50002 | 0.40041 | 0.71657 | 0.45591 | 0.24361 |
| 4 | -71c | 1 | 0.12783 | 0.15835 | 0.17720 | 0.10536 | 0.38295 | 0.36792 | 0.80851 | 0.39621 | 0.18762 |
| 4 | -68c | 1 | 0.15082 | 0.17992 | 0.19791 | 0.12783 | 0.42048 | 0.53073 | 0.86261 | 0.63049 | 0.22045 |
| 4 | -67c | 1 | 0.32850 | 0.34623 | 0.35733 | 0.30111 | 0.82289 | 0.66221 | 0.79919 | 0.78249 | 0.41323 |
| 4 | -64c | 1 | 2.30259 | 2.05490 | 1.93077 | 2.52573 | 1.37969 | 0.27427 | 0.39653 | 0.20344 | 1.95940 |
| 4 | -60c | 1 | 2.52573 | 2.21192 | 2.06102 | 2.81341 | 1.61345 | 0.23833 | 0.33033 | 0.35921 | 2.24412 |
| 4 | -55c | 1 | 0.02020 | 0.04876 | 0.06843 | 0.01005 | 0.06844 | 0.73000 | 0.78030 | 0.91327 | 0.01005 |
| 4 | -53c | 1 | 0.43078 | 0.44305 | 0.45075 | 0.40048 | 0.87634 | 0.88115 | 0.92328 | 0.99257 | 0.50143 |
| 4 | -52c | 2 | 1.80596 | 1.67731 | 1.60685 | 1.93640 | 1.15125 | 0.53161 | 0.49252 | 0.38927 | 1.48405 |
| 4 | -51c | 1 | 0.82098 | 0.81430 | 0.81015 | 0.86750 | 0.39166 | 0.57111 | 0.74761 | 0.68135 | 0.77523 |
| 4 | -46c | 1 | 0.65393 | 0.65580 | 0.65696 | 0.69315 | 0.31262 | 0.40733 | 0.50520 | 0.33526 | 0.65882 |
| 4 | -44c | 2 | 0.81126 | 0.78587 | 0.77226 | 0.85078 | 0.58893 | 0.53652 | 0.45109 | 0.52792 | 0.67789 |
| 4 | -39c | 1 | 1.60944 | 1.51780 | 1.46578 | 1.71480 | 1.15542 | 0.70105 | 0.72291 | 0.69391 | 1.32679 |
| 4 | -38c | 1 | 0.37106 | 0.38642 | 0.39605 | 0.34249 | 0.61690 | 0.71285 | 0.57165 | 0.73172 | 0.45112 |
| 4 | -34c | 2 | 0.37201 | 0.38908 | 0.39963 | 0.37964 | 0.27970 | 0.36786 | 0.35765 | 0.42295 | 0.40967 |
| 4 | -33c | 1 | 2.12026 | 1.92106 | 1.81778 | 2.30259 | 1.70322 | 1.12074 | 1.40348 | 1.05309 | 1.76566 |
| 4 | -32c | 1 | 0.09431 | 0.12652 | 0.14646 | 0.07257 | 0.14988 | 0.33057 | 0.31964 | 0.35073 | 0.13262 |
| 4 | -31c | 1 | 1.04982 | 1.02744 | 1.01377 | 1.10866 | 0.73956 | 0.66906 | 0.55037 | 0.66816 | 0.93055 |
| 4 | -29c | 2 | 0.20504 | 0.23054 | 0.24637 | 0.18077 | 0.29550 | 0.49216 | 0.54341 | 0.48797 | 0.28657 |
| 4 | -28c | 1 | 0.10536 | 0.13709 | 0.15670 | 0.08338 | 0.15416 | 0.39020 | 0.35238 | 0.35766 | 0.15192 |
| 4 | -27c | 5 | 0.75739 | 0.74265 | 0.73490 | 0.75374 | 0.88295 | 0.78262 | 0.78947 | 0.77028 | 0.71062 |
| 4 | -26c | 2 | 0.40822 | 0.42333 | 0.43267 | 0.41620 | 0.32775 | 0.45586 | 0.43019 | 0.40637 | 0.44743 |
| 4 | -25c | 3 | 1.53032 | 1.39174 | 1.32470 | 1.68356 | 1.27209 | 0.94576 | 0.94364 | 0.94070 | 1.39721 |
| 4 | -24c | 5 | 0.82113 | 0.77011 | 0.74728 | 0.86960 | 0.78655 | 0.68279 | 0.70618 | 0.66543 | 0.81741 |
| 4 | -23c | 6 | 0.78035 | 0.76777 | 0.76065 | 0.79490 | 0.74397 | 0.72547 | 0.69891 | 0.72397 | 0.73014 |
| 4 | -22c | 3 | 0.30070 | 0.32038 | 0.33265 | 0.31232 | 0.23500 | 0.26720 | 0.22619 | 0.26468 | 0.37348 |
| 4 | -21c | 5 | 0.46952 | 0.47870 | 0.48475 | 0.43610 | 0.61172 | 0.59236 | 0.53119 | 0.59578 | 0.48633 |
| 4 | -20c | 3 | 1.20323 | 1.12575 | 1.08607 | 1.28509 | 1.08410 | 0.91057 | 0.93291 | 0.94127 | 1.05688 |
| 4 | -19c | 16 | 0.81273 | 0.78523 | 0.77137 | 0.83951 | 0.79007 | 0.71249 | 0.72871 | 0.70260 | 0.76467 |
| 4 | -18c | 14 | 0.89057 | 0.83458 | 0.81121 | 0.96335 | 0.88589 | 0.77250 | 0.77368 | 0.77448 | 0.94274 |
| 4 | -17c | 12 | 0.47313 | 0.48297 | 0.48922 | 0.49245 | 0.40738 | 0.43285 | 0.45901 | 0.44597 | 0.48348 |
| 4 | -16c | 10 | 0.46982 | 0.47990 | 0.48629 | 0.48624 | 0.40277 | 0.45251 | 0.47514 | 0.44654 | 0.48312 |
| 4 | -15c | 17 | 0.56054 | 0.56234 | 0.56401 | 0.55071 | 0.59529 | 0.59294 | 0.58486 | 0.59012 | 0.55962 |
| 4 | -14c | 20 | 1.01778 | 0.90702 | 0.86692 | 1.06681 | 0.94357 | 0.82522 | 0.77404 | 0.80327 | 1.03714 |
| 4 | -13c | 24 | 0.68216 | 0.63979 | 0.62763 | 0.69167 | 0.62203 | 0.60296 | 0.57804 | 0.59886 | 0.69589 |
| 4 | -12c | 47 | 0.66379 | 0.63178 | 0.62041 | 0.70566 | 0.63686 | 0.59359 | 0.57526 | 0.59901 | 0.71654 |
| 4 | -11c | 43 | 0.54260 | 0.54695 | 0.55000 | 0.54245 | 0.54643 | 0.54705 | 0.54383 | 0.55138 | 0.55055 |
| 4 | -10c | 67 | 0.57975 | 0.57230 | 0.57016 | 0.59020 | 0.57418 | 0.56224 | 0.54584 | 0.55729 | 0.58438 |
| 4 | -9c | 65 | 0.62917 | 0.62581 | 0.62444 | 0.62364 | 0.64826 | 0.63733 | 0.63349 | 0.63530 | 0.60980 |
| 4 | -8c | 107 | 0.54446 | 0.54494 | 0.54641 | 0.54382 | 0.55368 | 0.54572 | 0.55198 | 0.54751 | 0.54118 |
| 4 | -7c | 194 | 0.48958 | 0.49390 | 0.49760 | 0.49262 | 0.48507 | 0.49280 | 0.49371 | 0.49312 | 0.49851 |
| 4 | -6c | 277 | 0.50587 | 0.50864 | 0.51158 | 0.51173 | 0.49908 | 0.50186 | 0.49876 | 0.50218 | 0.52534 |
| 4 | -5c | 490 | 0.54342 | 0.54068 | 0.54106 | 0.54780 | 0.54307 | 0.54199 | 0.53773 | 0.54053 | 0.54933 |
| 4 | -4c | 1,134 | 0.54319 | 0.54413 | 0.54583 | 0.54603 | 0.54307 | 0.54323 | 0.54167 | 0.54339 | 0.54705 |
| 4 | -3c | 3,090 | 0.52669 | 0.52795 | 0.52990 | 0.52964 | 0.52760 | 0.52620 | 0.52403 | 0.52594 | 0.52992 |
| 4 | -2c | 8,528 | 0.47606 | 0.47775 | 0.48049 | 0.47819 | 0.47706 | 0.47456 | 0.47263 | 0.47444 | 0.47782 |
| 4 | -1c | 25,585 | 0.41209 | 0.41694 | 0.42191 | 0.41511 | 0.41267 | 0.41057 | 0.40930 | 0.41055 | 0.41753 |
| 4 | +0c | 86,785 | 0.36763 | 0.37424 | 0.38052 | 0.36763 | 0.36865 | 0.36639 | 0.36505 | 0.36638 | 0.37273 |
| 4 | +1c | 25,698 | 0.41298 | 0.41762 | 0.42249 | 0.41556 | 0.41355 | 0.41168 | 0.41029 | 0.41152 | 0.41853 |
| 4 | +2c | 8,583 | 0.46124 | 0.46402 | 0.46742 | 0.46514 | 0.46126 | 0.45904 | 0.45719 | 0.45874 | 0.46701 |
| 4 | +3c | 3,130 | 0.53452 | 0.53533 | 0.53701 | 0.53890 | 0.53448 | 0.53274 | 0.53087 | 0.53244 | 0.53700 |
| 4 | +4c | 1,085 | 0.54264 | 0.54348 | 0.54509 | 0.54358 | 0.54405 | 0.54151 | 0.53958 | 0.54165 | 0.54808 |
| 4 | +5c | 494 | 0.57410 | 0.56929 | 0.56837 | 0.58365 | 0.57098 | 0.56809 | 0.56199 | 0.56827 | 0.58035 |
| 4 | +6c | 291 | 0.54302 | 0.53713 | 0.53641 | 0.55116 | 0.54280 | 0.53699 | 0.53296 | 0.53870 | 0.56376 |
| 4 | +7c | 177 | 0.55082 | 0.55089 | 0.55209 | 0.55560 | 0.55221 | 0.55182 | 0.55026 | 0.55151 | 0.55760 |
| 4 | +8c | 139 | 0.56982 | 0.56450 | 0.56368 | 0.57429 | 0.56967 | 0.56612 | 0.56401 | 0.56580 | 0.57897 |
| 4 | +9c | 87 | 0.55229 | 0.53973 | 0.53720 | 0.57211 | 0.53316 | 0.51711 | 0.52132 | 0.52491 | 0.58336 |
| 4 | +10c | 62 | 0.62738 | 0.60925 | 0.60411 | 0.63516 | 0.60116 | 0.59547 | 0.59285 | 0.59549 | 0.63061 |
| 4 | +11c | 55 | 0.58526 | 0.58005 | 0.57856 | 0.58229 | 0.61016 | 0.59439 | 0.57838 | 0.58948 | 0.58018 |
| 4 | +12c | 30 | 0.70770 | 0.69610 | 0.69023 | 0.70949 | 0.73024 | 0.70431 | 0.69896 | 0.70389 | 0.66860 |
| 4 | +13c | 22 | 0.85559 | 0.80501 | 0.78199 | 0.89688 | 0.83887 | 0.75605 | 0.71970 | 0.75212 | 0.84239 |
| 4 | +14c | 28 | 0.66397 | 0.62890 | 0.61612 | 0.72675 | 0.61366 | 0.58733 | 0.56267 | 0.58683 | 0.73397 |
| 4 | +15c | 13 | 0.82617 | 0.78703 | 0.76861 | 0.87197 | 0.76486 | 0.72104 | 0.71965 | 0.71920 | 0.79582 |
| 4 | +16c | 24 | 0.92224 | 0.84765 | 0.81856 | 0.94019 | 0.86186 | 0.77849 | 0.77012 | 0.78557 | 0.86976 |
| 4 | +17c | 17 | 0.63189 | 0.63179 | 0.63196 | 0.62568 | 0.68510 | 0.64903 | 0.64931 | 0.65205 | 0.62475 |
| 4 | +18c | 18 | 0.51456 | 0.52069 | 0.52487 | 0.51762 | 0.51486 | 0.53528 | 0.53741 | 0.54144 | 0.51614 |
| 4 | +19c | 6 | 0.24606 | 0.26512 | 0.27806 | 0.24465 | 0.27392 | 0.30278 | 0.34396 | 0.28968 | 0.28975 |
| 4 | +20c | 7 | 0.61717 | 0.60492 | 0.59978 | 0.62417 | 0.65970 | 0.57028 | 0.55963 | 0.54812 | 0.61786 |
| 4 | +21c | 4 | 0.48764 | 0.49779 | 0.50414 | 0.48766 | 0.51732 | 0.50713 | 0.48082 | 0.49533 | 0.53069 |
| 4 | +22c | 4 | 0.38991 | 0.40552 | 0.41523 | 0.39675 | 0.33455 | 0.40492 | 0.39004 | 0.36363 | 0.43233 |
| 4 | +23c | 7 | 0.52156 | 0.52873 | 0.53329 | 0.52389 | 0.53515 | 0.53374 | 0.50365 | 0.51230 | 0.54817 |
| 4 | +24c | 1 | 0.24846 | 0.27109 | 0.28519 | 0.22314 | 0.41329 | 0.40201 | 0.28995 | 0.36255 | 0.32628 |
| 4 | +25c | 7 | 0.53685 | 0.54350 | 0.54773 | 0.52990 | 0.60043 | 0.65779 | 0.69229 | 0.65054 | 0.52479 |
| 4 | +26c | 3 | 0.58179 | 0.58698 | 0.59024 | 0.61830 | 0.32754 | 0.45612 | 0.50933 | 0.42479 | 0.61803 |
| 4 | +27c | 3 | 0.66838 | 0.66887 | 0.66922 | 0.66271 | 0.78082 | 0.74182 | 0.72183 | 0.72142 | 0.66876 |
| 4 | +28c | 2 | 0.55267 | 0.55956 | 0.56388 | 0.51712 | 0.91706 | 0.68320 | 0.63309 | 0.70357 | 0.56519 |
| 4 | +29c | 3 | 0.91786 | 0.86876 | 0.84402 | 0.99287 | 0.57677 | 0.54996 | 0.63297 | 0.55493 | 0.86721 |
| 4 | +30c | 1 | 0.43078 | 0.44305 | 0.45075 | 0.40048 | 0.77093 | 0.48804 | 0.43975 | 0.48467 | 0.49066 |
| 4 | +31c | 4 | 0.71040 | 0.70390 | 0.70035 | 0.70650 | 0.85796 | 0.72399 | 0.73011 | 0.76360 | 0.67813 |
| 4 | +32c | 2 | 0.61499 | 0.61770 | 0.61948 | 0.57616 | 1.07490 | 0.85874 | 0.87919 | 0.90285 | 0.60294 |
| 4 | +33c | 1 | 0.22314 | 0.24743 | 0.26253 | 0.19845 | 0.46436 | 0.46517 | 0.42551 | 0.43174 | 0.29944 |
| 4 | +34c | 2 | 0.38913 | 0.40474 | 0.41444 | 0.39527 | 0.34808 | 0.41115 | 0.38361 | 0.39156 | 0.44449 |
| 4 | +35c | 1 | 1.89712 | 1.75014 | 1.67076 | 2.04022 | 1.16986 | 0.93875 | 1.11547 | 1.05599 | 1.59118 |
| 4 | +39c | 2 | 0.32208 | 0.34149 | 0.35352 | 0.32821 | 0.25879 | 0.53660 | 0.56683 | 0.42005 | 0.37019 |
| 4 | +40c | 2 | 0.29430 | 0.31501 | 0.32787 | 0.26730 | 0.65864 | 0.52940 | 0.51405 | 0.48347 | 0.33300 |
| 4 | +47c | 2 | 0.20293 | 0.22450 | 0.23891 | 0.18336 | 0.54079 | 0.41760 | 0.50654 | 0.50047 | 0.23168 |
| 4 | +49c | 2 | 0.29116 | 0.30849 | 0.32023 | 0.26884 | 0.72837 | 0.47601 | 0.59867 | 0.55713 | 0.30109 |
| 4 | +55c | 1 | 0.35667 | 0.37281 | 0.38293 | 0.32850 | 1.06337 | 0.26834 | 0.34440 | 0.24086 | 0.42841 |
| 4 | +57c | 1 | 0.05129 | 0.08388 | 0.10455 | 0.03046 | 0.26140 | 0.48628 | 0.87674 | 0.72349 | 0.03218 |
| 4 | +58c | 2 | 0.19549 | 0.22155 | 0.23772 | 0.17139 | 0.70855 | 0.28918 | 0.34968 | 0.35322 | 0.25744 |
| 4 | +59c | 1 | 0.38566 | 0.40024 | 0.40938 | 0.35667 | 1.20577 | 0.19019 | 0.30476 | 0.20690 | 0.45332 |
| 4 | +63c | 1 | 0.31471 | 0.33324 | 0.34484 | 0.28768 | 1.11852 | 0.14483 | 0.28653 | 0.18121 | 0.39076 |
| 4 | +68c | 2 | 0.30976 | 0.32875 | 0.34062 | 0.28282 | 1.19888 | 0.17253 | 0.37626 | 0.27569 | 0.38357 |
| 4 | +69c | 1 | 1.51413 | 1.43780 | 1.39382 | 1.60944 | 0.42687 | 0.35027 | 1.21365 | 0.52180 | 1.27872 |
| 4 | +71c | 1 | 1.56065 | 1.47704 | 1.42920 | 1.66073 | 0.42685 | 0.33935 | 1.07732 | 0.46224 | 1.31410 |
| 4 | +77c | 1 | 2.30258 | 2.05490 | 1.93077 | 2.52573 | 0.69536 | 0.23918 | 0.62865 | 0.31741 | 2.00707 |
| 4 | +81c | 1 | 3.21888 | 2.65642 | 2.41742 | 3.91202 | 1.10280 | 0.25647 | 0.39680 | 0.20179 | 4.60517 |
| 3 | -70c | 1 | 4.60517 | 3.38851 | 2.97115 | 4.60517 | 2.89691 | 1.21028 | 1.00211 | 0.76418 | 4.60517 |
| 3 | -69c | 1 | 2.30259 | 2.05490 | 1.93077 | 2.52573 | 1.31036 | 0.92676 | 0.77969 | 0.86566 | 1.95940 |
| 3 | -67c | 1 | 0.37106 | 0.38642 | 0.39605 | 0.34249 | 0.91503 | 1.13901 | 0.78222 | 0.95975 | 0.45112 |
| 3 | -56c | 1 | 0.52763 | 0.53530 | 0.54010 | 0.49430 | 1.09502 | 0.81744 | 0.72417 | 0.73932 | 0.57794 |
| 3 | -52c | 1 | 1.07881 | 1.05399 | 1.03888 | 1.13943 | 0.56817 | 0.53375 | 0.63210 | 0.49674 | 0.95020 |
| 3 | -51c | 1 | 0.04082 | 0.07278 | 0.09338 | 0.02020 | 0.10618 | 0.36157 | 0.43206 | 0.37313 | 0.01005 |
| 3 | -46c | 2 | 0.53356 | 0.54169 | 0.54676 | 0.54502 | 0.43213 | 0.66108 | 0.73537 | 0.62780 | 0.54964 |
| 3 | -45c | 1 | 0.35667 | 0.37281 | 0.38293 | 0.32850 | 0.65485 | 0.71749 | 0.60404 | 0.74720 | 0.43853 |
| 3 | -43c | 1 | 0.49430 | 0.50351 | 0.50928 | 0.46204 | 0.87746 | 0.91495 | 0.81708 | 0.84389 | 0.55218 |
| 3 | -42c | 1 | 0.79851 | 0.79308 | 0.78972 | 0.75502 | 1.34818 | 0.89234 | 0.84044 | 0.82184 | 0.77394 |
| 3 | -40c | 1 | 0.16252 | 0.19086 | 0.20838 | 0.13926 | 0.28369 | 0.63176 | 0.55211 | 0.49575 | 0.23603 |
| 3 | -39c | 1 | 0.15082 | 0.17992 | 0.19791 | 0.12783 | 0.26003 | 0.51817 | 0.45896 | 0.51839 | 0.22045 |
| 3 | -38c | 2 | 0.43209 | 0.44236 | 0.44971 | 0.40428 | 0.70119 | 0.51043 | 0.47923 | 0.50042 | 0.40779 |
| 3 | -35c | 1 | 0.15082 | 0.17992 | 0.19791 | 0.12783 | 0.24239 | 0.29494 | 0.21888 | 0.28866 | 0.22045 |
| 3 | -34c | 1 | 0.40048 | 0.41428 | 0.42294 | 0.37106 | 0.62769 | 0.54474 | 0.51447 | 0.61645 | 0.47627 |
| 3 | -33c | 2 | 1.00870 | 0.98948 | 0.97771 | 0.95527 | 1.49334 | 1.14864 | 1.17403 | 1.05889 | 0.91904 |
| 3 | -32c | 4 | 0.79437 | 0.78026 | 0.77239 | 0.83131 | 0.61876 | 0.64994 | 0.68972 | 0.66525 | 0.75092 |
| 3 | -31c | 2 | 0.76972 | 0.76213 | 0.75765 | 0.78583 | 0.72138 | 0.71191 | 0.74009 | 0.71003 | 0.73595 |
| 3 | -30c | 4 | 0.53369 | 0.53981 | 0.54376 | 0.49807 | 0.77100 | 0.68982 | 0.65238 | 0.61566 | 0.56221 |
| 3 | -29c | 1 | 0.09431 | 0.12652 | 0.14646 | 0.07257 | 0.14170 | 0.27750 | 0.28076 | 0.25133 | 0.13262 |
| 3 | -27c | 7 | 0.38305 | 0.39631 | 0.40524 | 0.37081 | 0.44504 | 0.42969 | 0.41603 | 0.41532 | 0.37415 |
| 3 | -26c | 2 | 0.40126 | 0.41644 | 0.42586 | 0.40822 | 0.32976 | 0.43500 | 0.42979 | 0.41460 | 0.44882 |
| 3 | -25c | 4 | 0.29232 | 0.31301 | 0.32585 | 0.27987 | 0.33594 | 0.34269 | 0.36675 | 0.31426 | 0.35845 |
| 3 | -24c | 1 | 1.38629 | 1.32817 | 1.29403 | 1.46968 | 1.14570 | 0.88706 | 0.78654 | 0.89257 | 1.16289 |
| 3 | -23c | 4 | 0.69676 | 0.68539 | 0.67958 | 0.71683 | 0.66443 | 0.55627 | 0.53125 | 0.52814 | 0.65281 |
| 3 | -22c | 9 | 1.05595 | 0.92148 | 0.87578 | 1.05322 | 0.97844 | 0.83842 | 0.86287 | 0.79496 | 1.04653 |
| 3 | -21c | 11 | 0.95465 | 0.89602 | 0.86797 | 1.00668 | 0.90785 | 0.74165 | 0.73317 | 0.73707 | 0.88650 |
| 3 | -20c | 9 | 0.55276 | 0.55706 | 0.55995 | 0.55758 | 0.54062 | 0.55007 | 0.56317 | 0.57169 | 0.55272 |
| 3 | -19c | 12 | 0.85643 | 0.81622 | 0.79704 | 0.89577 | 0.80922 | 0.74622 | 0.71717 | 0.75341 | 0.81073 |
| 3 | -18c | 13 | 0.46847 | 0.47184 | 0.47518 | 0.46851 | 0.48025 | 0.49279 | 0.49906 | 0.47765 | 0.47280 |
| 3 | -17c | 16 | 0.54733 | 0.54405 | 0.54372 | 0.56070 | 0.52755 | 0.51824 | 0.50999 | 0.51290 | 0.55552 |
| 3 | -16c | 19 | 0.63063 | 0.61495 | 0.60914 | 0.65330 | 0.60683 | 0.60282 | 0.59712 | 0.60246 | 0.61336 |
| 3 | -15c | 17 | 0.54068 | 0.53184 | 0.52890 | 0.56216 | 0.52232 | 0.51628 | 0.52470 | 0.52543 | 0.52504 |
| 3 | -14c | 30 | 0.51263 | 0.51396 | 0.51604 | 0.51395 | 0.52220 | 0.52327 | 0.53655 | 0.51989 | 0.51633 |
| 3 | -13c | 32 | 0.53968 | 0.53503 | 0.53445 | 0.54888 | 0.54057 | 0.51748 | 0.49917 | 0.51834 | 0.54431 |
| 3 | -12c | 42 | 0.54005 | 0.54237 | 0.54461 | 0.55366 | 0.51912 | 0.52608 | 0.51969 | 0.52296 | 0.53591 |
| 3 | -11c | 68 | 0.55746 | 0.55601 | 0.55660 | 0.56169 | 0.56390 | 0.54917 | 0.54583 | 0.55363 | 0.56442 |
| 3 | -10c | 76 | 0.58724 | 0.57380 | 0.56991 | 0.62347 | 0.56902 | 0.55809 | 0.55787 | 0.56512 | 0.62574 |
| 3 | -9c | 124 | 0.43307 | 0.44363 | 0.45059 | 0.43301 | 0.43134 | 0.43990 | 0.44721 | 0.44165 | 0.45692 |
| 3 | -8c | 136 | 0.55095 | 0.55022 | 0.55124 | 0.55373 | 0.55970 | 0.55494 | 0.55854 | 0.55438 | 0.55843 |
| 3 | -7c | 179 | 0.52384 | 0.52501 | 0.52690 | 0.53068 | 0.51906 | 0.52396 | 0.51789 | 0.52331 | 0.52639 |
| 3 | -6c | 316 | 0.56979 | 0.56820 | 0.56840 | 0.56846 | 0.57907 | 0.57029 | 0.56592 | 0.56753 | 0.56206 |
| 3 | -5c | 497 | 0.53872 | 0.53867 | 0.54012 | 0.53843 | 0.54297 | 0.53352 | 0.52647 | 0.53169 | 0.54783 |
| 3 | -4c | 995 | 0.53455 | 0.53526 | 0.53696 | 0.53588 | 0.53597 | 0.53310 | 0.52804 | 0.53166 | 0.54007 |
| 3 | -3c | 3,033 | 0.51610 | 0.51842 | 0.52096 | 0.51735 | 0.51665 | 0.51612 | 0.51575 | 0.51610 | 0.52039 |
| 3 | -2c | 7,553 | 0.45263 | 0.45661 | 0.46062 | 0.45357 | 0.45319 | 0.45056 | 0.44861 | 0.45009 | 0.45735 |
| 3 | -1c | 23,417 | 0.37971 | 0.38539 | 0.39126 | 0.38067 | 0.37985 | 0.37744 | 0.37567 | 0.37723 | 0.38548 |
| 3 | +0c | 85,443 | 0.32882 | 0.33545 | 0.34252 | 0.32882 | 0.32919 | 0.32728 | 0.32486 | 0.32705 | 0.33562 |
| 3 | +1c | 23,356 | 0.37740 | 0.38334 | 0.38933 | 0.38160 | 0.37736 | 0.37494 | 0.37261 | 0.37462 | 0.38471 |
| 3 | +2c | 7,625 | 0.44579 | 0.44928 | 0.45327 | 0.45278 | 0.44444 | 0.44232 | 0.43985 | 0.44176 | 0.45505 |
| 3 | +3c | 2,932 | 0.50606 | 0.50886 | 0.51177 | 0.51076 | 0.50462 | 0.50383 | 0.50219 | 0.50394 | 0.51567 |
| 3 | +4c | 1,104 | 0.51180 | 0.51540 | 0.51853 | 0.51667 | 0.50901 | 0.50841 | 0.50595 | 0.50787 | 0.52139 |
| 3 | +5c | 452 | 0.51993 | 0.52133 | 0.52335 | 0.52734 | 0.51510 | 0.51077 | 0.50631 | 0.51026 | 0.52457 |
| 3 | +6c | 278 | 0.51487 | 0.51696 | 0.51944 | 0.52137 | 0.51115 | 0.51214 | 0.51333 | 0.51151 | 0.52456 |
| 3 | +7c | 184 | 0.55522 | 0.55664 | 0.55828 | 0.55574 | 0.55971 | 0.56445 | 0.55743 | 0.56299 | 0.55428 |
| 3 | +8c | 132 | 0.61058 | 0.60336 | 0.60073 | 0.62418 | 0.59919 | 0.59397 | 0.58163 | 0.59459 | 0.60443 |
| 3 | +9c | 93 | 0.61606 | 0.60800 | 0.60558 | 0.63247 | 0.61283 | 0.60457 | 0.60280 | 0.60361 | 0.62576 |
| 3 | +10c | 73 | 0.45460 | 0.46544 | 0.47239 | 0.46036 | 0.44041 | 0.45208 | 0.44477 | 0.45046 | 0.47522 |
| 3 | +11c | 56 | 0.48728 | 0.49166 | 0.49559 | 0.49681 | 0.47432 | 0.47456 | 0.46780 | 0.47416 | 0.50947 |
| 3 | +12c | 43 | 0.69437 | 0.67724 | 0.66956 | 0.70725 | 0.68430 | 0.66349 | 0.63548 | 0.67480 | 0.67845 |
| 3 | +13c | 43 | 0.61706 | 0.60800 | 0.60489 | 0.61947 | 0.65174 | 0.61874 | 0.59408 | 0.61245 | 0.62451 |
| 3 | +14c | 22 | 0.44544 | 0.45028 | 0.45475 | 0.45515 | 0.42848 | 0.45142 | 0.47326 | 0.44476 | 0.47147 |
| 3 | +15c | 17 | 0.40585 | 0.41829 | 0.42633 | 0.40893 | 0.39061 | 0.42204 | 0.42862 | 0.41657 | 0.42392 |
| 3 | +16c | 12 | 0.54465 | 0.54855 | 0.55134 | 0.55117 | 0.53058 | 0.54855 | 0.52796 | 0.55407 | 0.53529 |
| 3 | +17c | 12 | 0.47642 | 0.48662 | 0.49311 | 0.48967 | 0.41214 | 0.49093 | 0.49861 | 0.49646 | 0.48432 |
| 3 | +18c | 14 | 0.57964 | 0.57695 | 0.57639 | 0.58154 | 0.59959 | 0.57131 | 0.56069 | 0.57973 | 0.56744 |
| 3 | +19c | 9 | 0.51935 | 0.52130 | 0.52321 | 0.48626 | 0.70220 | 0.55304 | 0.55011 | 0.55670 | 0.50911 |
| 3 | +20c | 8 | 0.85360 | 0.79905 | 0.77596 | 0.90227 | 0.88155 | 0.73701 | 0.66896 | 0.68967 | 0.91914 |
| 3 | +21c | 5 | 0.83263 | 0.79287 | 0.77408 | 0.86719 | 0.79725 | 0.76710 | 0.78249 | 0.74465 | 0.78305 |
| 3 | +22c | 4 | 0.91914 | 0.90492 | 0.89625 | 0.91445 | 1.02102 | 0.92909 | 0.91523 | 0.87248 | 0.84830 |
| 3 | +23c | 7 | 0.48384 | 0.49458 | 0.50128 | 0.46027 | 0.66182 | 0.64743 | 0.62128 | 0.64599 | 0.48897 |
| 3 | +24c | 2 | 0.34958 | 0.36611 | 0.37648 | 0.34978 | 0.37528 | 0.36322 | 0.39891 | 0.32235 | 0.42715 |
| 3 | +25c | 1 | 0.19845 | 0.22439 | 0.24048 | 0.17435 | 0.34697 | 0.38224 | 0.39024 | 0.36040 | 0.27170 |
| 3 | +26c | 4 | 0.49325 | 0.50361 | 0.51004 | 0.45938 | 0.79226 | 0.76290 | 0.67408 | 0.73577 | 0.50486 |
| 3 | +27c | 2 | 0.84424 | 0.83618 | 0.83120 | 0.89190 | 0.50798 | 0.56625 | 0.61433 | 0.52810 | 0.80569 |
| 3 | +28c | 5 | 0.74725 | 0.74127 | 0.73784 | 0.74192 | 0.85639 | 0.79681 | 0.81902 | 0.81872 | 0.68558 |
| 3 | +29c | 3 | 0.55786 | 0.56525 | 0.56980 | 0.55246 | 0.64254 | 0.61851 | 0.59470 | 0.58873 | 0.56223 |
| 3 | +30c | 1 | 0.43078 | 0.44305 | 0.45075 | 0.46204 | 0.19941 | 0.24623 | 0.26265 | 0.16636 | 0.50143 |
| 3 | +31c | 2 | 0.52093 | 0.52894 | 0.53396 | 0.48776 | 0.92821 | 0.72428 | 0.67854 | 0.82291 | 0.56043 |
| 3 | +32c | 1 | 0.17435 | 0.20191 | 0.21896 | 0.15082 | 0.36724 | 0.60466 | 0.46638 | 0.47513 | 0.24278 |
| 3 | +34c | 3 | 1.08787 | 1.05659 | 1.03806 | 1.15200 | 0.60200 | 0.60497 | 0.52401 | 0.65344 | 0.97464 |
| 3 | +39c | 1 | 0.18633 | 0.21308 | 0.22966 | 0.16252 | 0.45833 | 0.54448 | 0.43475 | 0.56501 | 0.25740 |
| 3 | +40c | 1 | 0.13926 | 0.16909 | 0.18752 | 0.11653 | 0.36952 | 0.55049 | 0.60635 | 0.42732 | 0.19632 |
| 3 | +42c | 1 | 0.82098 | 0.81430 | 0.81015 | 0.86750 | 0.34459 | 0.56486 | 0.55674 | 0.51439 | 0.78958 |
| 3 | +46c | 1 | 0.11653 | 0.14769 | 0.16693 | 0.09431 | 0.37025 | 0.59465 | 0.66445 | 0.47017 | 0.16245 |
| 3 | +50c | 2 | 2.55800 | 2.10618 | 1.93060 | 2.93907 | 1.35522 | 0.78800 | 0.58876 | 0.88488 | 2.82971 |
| 3 | +52c | 1 | 0.18633 | 0.21308 | 0.22966 | 0.16252 | 0.61361 | 0.52031 | 0.50586 | 0.47418 | 0.25740 |
| 3 | +58c | 1 | 1.56065 | 1.47704 | 1.42920 | 1.66073 | 0.57279 | 1.41840 | 1.36271 | 1.19754 | 1.31410 |
| 3 | +66c | 1 | 0.09431 | 0.12652 | 0.14646 | 0.07257 | 0.50382 | 0.81348 | 0.50966 | 0.63639 | 0.12516 |
| 3 | +80c | 1 | 2.99573 | 2.52000 | 2.30987 | 3.50656 | 1.00059 | 0.29153 | 0.33889 | 0.27768 | 3.45259 |
| 2 | -72c | 1 | 1.34707 | 1.29401 | 1.26267 | 1.42712 | 0.58662 | 0.47946 | 0.54108 | 0.43307 | 1.13511 |
| 2 | -69c | 1 | 0.01005 | 0.03434 | 0.05260 | 0.01005 | 0.05557 | 0.21104 | 0.11349 | 0.31283 | 0.01005 |
| 2 | -60c | 1 | 0.22314 | 0.24743 | 0.26253 | 0.19845 | 0.52446 | 1.00046 | 1.04899 | 0.87056 | 0.30833 |
| 2 | -57c | 2 | 0.53647 | 0.54270 | 0.54688 | 0.55463 | 0.31501 | 0.39284 | 0.45580 | 0.35951 | 0.47522 |
| 2 | -47c | 2 | 0.58382 | 0.58890 | 0.59209 | 0.54846 | 1.07759 | 0.82978 | 0.78745 | 0.81468 | 0.61887 |
| 2 | -44c | 2 | 0.78752 | 0.76854 | 0.75812 | 0.82201 | 0.58758 | 0.63178 | 0.62164 | 0.58235 | 0.68739 |
| 2 | -43c | 1 | 1.13943 | 1.10915 | 1.09082 | 1.20397 | 0.70016 | 0.74010 | 0.80166 | 0.68410 | 0.99143 |
| 2 | -42c | 5 | 0.99478 | 0.91896 | 0.88539 | 1.06481 | 1.00758 | 0.79588 | 0.78535 | 0.79785 | 1.01048 |
| 2 | -40c | 1 | 0.38566 | 0.40024 | 0.40938 | 0.35667 | 0.65741 | 0.65853 | 0.63738 | 0.60780 | 0.46370 |
| 2 | -39c | 3 | 0.61448 | 0.59995 | 0.59349 | 0.64042 | 0.51108 | 0.47801 | 0.49476 | 0.49429 | 0.53743 |
| 2 | -38c | 3 | 0.41208 | 0.42385 | 0.43185 | 0.41759 | 0.36132 | 0.46036 | 0.48784 | 0.49164 | 0.40972 |
| 2 | -37c | 1 | 0.11653 | 0.14769 | 0.16693 | 0.09431 | 0.19751 | 0.31681 | 0.20213 | 0.29236 | 0.17020 |
| 2 | -36c | 4 | 0.91041 | 0.85116 | 0.82341 | 0.98336 | 0.71324 | 0.71213 | 0.73502 | 0.71535 | 0.87590 |
| 2 | -35c | 2 | 0.62240 | 0.62544 | 0.62736 | 0.61653 | 0.74464 | 0.70097 | 0.70814 | 0.73216 | 0.63854 |
| 2 | -34c | 4 | 0.31244 | 0.33224 | 0.34451 | 0.30140 | 0.36785 | 0.39348 | 0.36962 | 0.42091 | 0.36553 |
| 2 | -33c | 4 | 0.16021 | 0.18638 | 0.20312 | 0.13944 | 0.24912 | 0.36871 | 0.29072 | 0.36292 | 0.21947 |
| 2 | -32c | 4 | 0.98094 | 0.93430 | 0.91048 | 1.01784 | 0.96388 | 0.78004 | 0.87449 | 0.83354 | 0.90509 |
| 2 | -31c | 6 | 0.47902 | 0.48610 | 0.49118 | 0.47001 | 0.57062 | 0.52639 | 0.58555 | 0.55963 | 0.49858 |
| 2 | -30c | 3 | 2.17416 | 1.75606 | 1.60939 | 2.21105 | 1.66570 | 1.32770 | 1.22765 | 1.20709 | 2.11385 |
| 2 | -29c | 2 | 0.95527 | 0.93999 | 0.93061 | 0.95866 | 1.01402 | 0.93516 | 0.86722 | 0.87508 | 0.87419 |
| 2 | -28c | 3 | 0.88826 | 0.87637 | 0.86909 | 0.90454 | 0.84741 | 0.83482 | 0.76296 | 0.79685 | 0.82535 |
| 2 | -27c | 10 | 0.75247 | 0.74218 | 0.73666 | 0.75569 | 0.78765 | 0.74206 | 0.68310 | 0.73865 | 0.71020 |
| 2 | -26c | 9 | 0.58816 | 0.58773 | 0.58807 | 0.60835 | 0.50747 | 0.53925 | 0.49879 | 0.52350 | 0.56984 |
| 2 | -25c | 13 | 0.36638 | 0.38129 | 0.39104 | 0.36721 | 0.35182 | 0.40866 | 0.42684 | 0.40136 | 0.39846 |
| 2 | -24c | 9 | 1.13361 | 1.02308 | 0.97873 | 1.23879 | 0.98955 | 0.79312 | 0.69582 | 0.80173 | 1.15779 |
| 2 | -23c | 12 | 0.43740 | 0.44948 | 0.45712 | 0.44755 | 0.38907 | 0.41079 | 0.42395 | 0.40735 | 0.46296 |
| 2 | -22c | 15 | 0.79264 | 0.76801 | 0.75616 | 0.82218 | 0.76983 | 0.70748 | 0.71808 | 0.69780 | 0.76566 |
| 2 | -21c | 16 | 0.79479 | 0.75335 | 0.73564 | 0.86211 | 0.70792 | 0.67105 | 0.63231 | 0.65665 | 0.85734 |
| 2 | -20c | 8 | 0.53526 | 0.54172 | 0.54598 | 0.53509 | 0.54472 | 0.54631 | 0.50818 | 0.55213 | 0.51335 |
| 2 | -19c | 24 | 0.58046 | 0.57600 | 0.57519 | 0.59757 | 0.55296 | 0.56102 | 0.56612 | 0.54858 | 0.58341 |
| 2 | -18c | 15 | 0.40802 | 0.42170 | 0.43046 | 0.40931 | 0.39700 | 0.42105 | 0.43731 | 0.42784 | 0.43084 |
| 2 | -17c | 26 | 0.73769 | 0.70242 | 0.68922 | 0.76945 | 0.71579 | 0.68021 | 0.66149 | 0.67215 | 0.74573 |
| 2 | -16c | 27 | 0.55924 | 0.55386 | 0.55282 | 0.57768 | 0.53701 | 0.51596 | 0.53047 | 0.51986 | 0.56465 |
| 2 | -15c | 46 | 0.42862 | 0.43720 | 0.44332 | 0.43301 | 0.42067 | 0.41986 | 0.41860 | 0.42858 | 0.44647 |
| 2 | -14c | 36 | 0.46135 | 0.47108 | 0.47752 | 0.45458 | 0.47564 | 0.49720 | 0.50493 | 0.49784 | 0.47513 |
| 2 | -13c | 50 | 0.72039 | 0.69122 | 0.67930 | 0.75297 | 0.69543 | 0.64982 | 0.62119 | 0.66130 | 0.69517 |
| 2 | -12c | 60 | 0.65614 | 0.64646 | 0.64197 | 0.66175 | 0.66736 | 0.65074 | 0.64463 | 0.64607 | 0.62461 |
| 2 | -11c | 98 | 0.50531 | 0.50935 | 0.51272 | 0.50912 | 0.50397 | 0.49765 | 0.49217 | 0.49741 | 0.51492 |
| 2 | -10c | 86 | 0.45652 | 0.46571 | 0.47179 | 0.45781 | 0.45552 | 0.45790 | 0.45022 | 0.45478 | 0.47772 |
| 2 | -9c | 123 | 0.47031 | 0.47042 | 0.47297 | 0.48345 | 0.45608 | 0.46073 | 0.45763 | 0.46447 | 0.50172 |
| 2 | -8c | 197 | 0.50356 | 0.50787 | 0.51137 | 0.50299 | 0.50625 | 0.50474 | 0.49977 | 0.50455 | 0.50884 |
| 2 | -7c | 234 | 0.51129 | 0.51273 | 0.51497 | 0.50871 | 0.51449 | 0.51462 | 0.51298 | 0.51319 | 0.52098 |
| 2 | -6c | 386 | 0.51527 | 0.51745 | 0.51987 | 0.51559 | 0.51668 | 0.51628 | 0.51069 | 0.51690 | 0.51834 |
| 2 | -5c | 650 | 0.55388 | 0.55001 | 0.54970 | 0.55644 | 0.55662 | 0.55110 | 0.54791 | 0.55128 | 0.55374 |
| 2 | -4c | 1,125 | 0.50795 | 0.51041 | 0.51326 | 0.50801 | 0.50872 | 0.50626 | 0.50483 | 0.50615 | 0.51688 |
| 2 | -3c | 2,896 | 0.48445 | 0.48840 | 0.49204 | 0.48513 | 0.48436 | 0.48191 | 0.48063 | 0.48221 | 0.49457 |
| 2 | -2c | 7,065 | 0.41996 | 0.42571 | 0.43092 | 0.42021 | 0.42000 | 0.41745 | 0.41536 | 0.41736 | 0.42747 |
| 2 | -1c | 21,514 | 0.34420 | 0.35186 | 0.35901 | 0.34577 | 0.34414 | 0.34179 | 0.34002 | 0.34186 | 0.35223 |
| 2 | +0c | 86,489 | 0.27119 | 0.28117 | 0.29036 | 0.27119 | 0.27230 | 0.26951 | 0.26714 | 0.26956 | 0.27726 |
| 2 | +1c | 21,920 | 0.34628 | 0.35405 | 0.36118 | 0.35065 | 0.34648 | 0.34393 | 0.34233 | 0.34395 | 0.35371 |
| 2 | +2c | 6,965 | 0.43043 | 0.43402 | 0.43831 | 0.43628 | 0.42967 | 0.42777 | 0.42587 | 0.42827 | 0.43988 |
| 2 | +3c | 2,795 | 0.49446 | 0.49735 | 0.50044 | 0.49841 | 0.49366 | 0.49302 | 0.48941 | 0.49260 | 0.50372 |
| 2 | +4c | 1,166 | 0.50648 | 0.50936 | 0.51235 | 0.50838 | 0.50749 | 0.50593 | 0.50270 | 0.50593 | 0.51639 |
| 2 | +5c | 565 | 0.49807 | 0.50143 | 0.50466 | 0.50198 | 0.49494 | 0.49699 | 0.50104 | 0.49654 | 0.50920 |
| 2 | +6c | 369 | 0.47341 | 0.47851 | 0.48273 | 0.47557 | 0.47269 | 0.47055 | 0.46892 | 0.46971 | 0.48547 |
| 2 | +7c | 252 | 0.50383 | 0.50778 | 0.51107 | 0.50681 | 0.50478 | 0.50337 | 0.49768 | 0.50106 | 0.50888 |
| 2 | +8c | 191 | 0.49537 | 0.49505 | 0.49732 | 0.50480 | 0.49217 | 0.48440 | 0.48252 | 0.48554 | 0.51728 |
| 2 | +9c | 131 | 0.46940 | 0.47356 | 0.47755 | 0.47302 | 0.47319 | 0.46762 | 0.46058 | 0.46755 | 0.49271 |
| 2 | +10c | 113 | 0.46432 | 0.47148 | 0.47675 | 0.47025 | 0.45613 | 0.46749 | 0.46362 | 0.46167 | 0.47303 |
| 2 | +11c | 86 | 0.45662 | 0.46100 | 0.46544 | 0.46203 | 0.45847 | 0.45692 | 0.45758 | 0.45525 | 0.48608 |
| 2 | +12c | 48 | 0.69825 | 0.67357 | 0.66389 | 0.71699 | 0.69027 | 0.65805 | 0.65451 | 0.66508 | 0.69522 |
| 2 | +13c | 42 | 0.51473 | 0.51978 | 0.52357 | 0.52548 | 0.48656 | 0.49749 | 0.48743 | 0.48725 | 0.52953 |
| 2 | +14c | 48 | 0.52064 | 0.51757 | 0.51770 | 0.52753 | 0.52140 | 0.52328 | 0.53610 | 0.52659 | 0.50791 |
| 2 | +15c | 32 | 0.45229 | 0.46105 | 0.46691 | 0.44856 | 0.46887 | 0.45182 | 0.43896 | 0.45378 | 0.46843 |
| 2 | +16c | 22 | 0.63500 | 0.63122 | 0.62965 | 0.62713 | 0.68592 | 0.63970 | 0.59849 | 0.63235 | 0.61154 |
| 2 | +17c | 29 | 0.66292 | 0.65349 | 0.64895 | 0.65694 | 0.71552 | 0.64134 | 0.61459 | 0.65745 | 0.62313 |
| 2 | +18c | 28 | 0.53876 | 0.54074 | 0.54271 | 0.53061 | 0.59636 | 0.52568 | 0.50313 | 0.53147 | 0.53428 |
| 2 | +19c | 21 | 0.55076 | 0.55429 | 0.55690 | 0.54062 | 0.63497 | 0.57078 | 0.54952 | 0.57789 | 0.56149 |
| 2 | +20c | 11 | 0.68980 | 0.67312 | 0.66566 | 0.70995 | 0.66645 | 0.67232 | 0.69212 | 0.66565 | 0.66668 |
| 2 | +21c | 18 | 0.60334 | 0.60176 | 0.60162 | 0.61173 | 0.59899 | 0.59596 | 0.56408 | 0.59368 | 0.58831 |
| 2 | +22c | 6 | 0.50885 | 0.51627 | 0.52113 | 0.52157 | 0.43362 | 0.44334 | 0.43202 | 0.46276 | 0.50638 |
| 2 | +23c | 10 | 0.76048 | 0.72536 | 0.71042 | 0.80141 | 0.72464 | 0.64405 | 0.62053 | 0.65876 | 0.78860 |
| 2 | +24c | 13 | 0.34007 | 0.35482 | 0.36468 | 0.32853 | 0.41868 | 0.42477 | 0.39391 | 0.41763 | 0.35225 |
| 2 | +25c | 12 | 0.90903 | 0.83024 | 0.79832 | 1.03088 | 0.77155 | 0.66249 | 0.65691 | 0.70098 | 0.98912 |
| 2 | +26c | 8 | 0.36510 | 0.38135 | 0.39159 | 0.35266 | 0.46700 | 0.46190 | 0.46172 | 0.48024 | 0.39955 |
| 2 | +27c | 7 | 0.71021 | 0.69856 | 0.69295 | 0.75017 | 0.47268 | 0.56945 | 0.52437 | 0.53689 | 0.66646 |
| 2 | +28c | 9 | 1.05768 | 0.96866 | 0.92772 | 1.14215 | 0.92017 | 0.73785 | 0.79985 | 0.76631 | 1.02662 |
| 2 | +29c | 4 | 0.62273 | 0.62567 | 0.62755 | 0.60968 | 0.81214 | 0.65471 | 0.69955 | 0.72260 | 0.63183 |
| 2 | +30c | 5 | 0.63728 | 0.63959 | 0.64106 | 0.65322 | 0.54774 | 0.61785 | 0.54390 | 0.55300 | 0.64511 |
| 2 | +31c | 4 | 0.27866 | 0.30000 | 0.31326 | 0.28287 | 0.23820 | 0.24029 | 0.23975 | 0.21950 | 0.34060 |
| 2 | +32c | 4 | 1.67218 | 1.45945 | 1.36850 | 1.98907 | 1.26475 | 0.91268 | 0.89896 | 0.95305 | 1.88479 |
| 2 | +33c | 3 | 0.97762 | 0.95550 | 0.94232 | 0.97292 | 1.19741 | 1.18458 | 1.17306 | 1.23347 | 0.88602 |
| 2 | +34c | 4 | 0.51372 | 0.52249 | 0.52797 | 0.49879 | 0.70523 | 0.81406 | 0.88627 | 0.81939 | 0.51159 |
| 2 | +35c | 4 | 0.74300 | 0.73997 | 0.73812 | 0.74739 | 0.81286 | 0.77368 | 0.75091 | 0.77580 | 0.72559 |
| 2 | +36c | 4 | 1.27197 | 1.13056 | 1.07059 | 1.44958 | 1.03223 | 0.81342 | 0.82266 | 0.79371 | 1.61601 |
| 2 | +37c | 4 | 0.35084 | 0.36739 | 0.37800 | 0.34351 | 0.44069 | 0.48702 | 0.51910 | 0.50478 | 0.37692 |
| 2 | +39c | 1 | 1.17118 | 1.13784 | 1.11772 | 1.23787 | 0.58392 | 0.63927 | 0.63277 | 0.70556 | 1.03135 |
| 2 | +40c | 1 | 0.01005 | 0.03434 | 0.05260 | 0.01005 | 0.05438 | 0.13994 | 0.18058 | 0.25289 | 0.01005 |
| 2 | +41c | 1 | 0.51083 | 0.51927 | 0.52456 | 0.47804 | 1.09748 | 0.67858 | 0.61368 | 0.69643 | 0.55354 |
| 2 | +54c | 1 | 0.35667 | 0.37281 | 0.38293 | 0.32850 | 1.04858 | 0.57449 | 0.53626 | 0.63314 | 0.42841 |
| 2 | +62c | 1 | 0.17435 | 0.20191 | 0.21896 | 0.15082 | 0.72198 | 0.48219 | 0.50207 | 0.57216 | 0.24278 |
| 2 | +65c | 1 | 1.34707 | 1.29401 | 1.26267 | 1.42712 | 0.39156 | 0.62220 | 0.73442 | 0.57754 | 1.15574 |
| 2 | +75c | 1 | 0.19845 | 0.22439 | 0.24048 | 0.17435 | 1.01520 | 0.52644 | 0.49638 | 0.55569 | 0.27170 |
