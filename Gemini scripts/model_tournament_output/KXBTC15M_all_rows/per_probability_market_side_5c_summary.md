# KXBTC15M Tournament By Market Probability And Market Outcome Side

- Metric: `log_loss` on the held-out test split
- Probability buckets: implied market probability in `5`-cent bins
- Market side buckets: `NO-leaning` for implied probability `< 50c`, `YES-leaning` for `>= 50c`
- Models: same tournament lineup as `smarterpred.py`

## Winners By Joint Bucket

| Market Side | Probability Bucket | Test Rows | Best Model | Best Log-Loss |
| --- | --- | ---: | --- | ---: |
| NO-leaning | 0-5c | 128,711 | LightGBM | 0.06686 |
| NO-leaning | 5-10c | 106,359 | LightGBM | 0.20438 |
| NO-leaning | 10-15c | 96,639 | LightGBM | 0.33397 |
| NO-leaning | 15-20c | 96,454 | LightGBM | 0.43478 |
| NO-leaning | 20-25c | 97,649 | LightGBM | 0.50415 |
| NO-leaning | 25-30c | 102,724 | LightGBM | 0.57832 |
| NO-leaning | 30-35c | 116,671 | LightGBM | 0.63155 |
| NO-leaning | 35-40c | 130,738 | LightGBM | 0.65717 |
| NO-leaning | 40-45c | 143,392 | LightGBM | 0.68031 |
| NO-leaning | 45-50c | 157,898 | LightGBM | 0.68906 |
| YES-leaning | 50-55c | 161,678 | LightGBM | 0.68868 |
| YES-leaning | 55-60c | 148,096 | LightGBM | 0.67605 |
| YES-leaning | 60-65c | 133,643 | LightGBM | 0.65653 |
| YES-leaning | 65-70c | 117,606 | LightGBM | 0.62743 |
| YES-leaning | 70-75c | 104,484 | LightGBM | 0.58773 |
| YES-leaning | 75-80c | 95,238 | LightGBM | 0.54395 |
| YES-leaning | 80-85c | 93,841 | LightGBM | 0.46303 |
| YES-leaning | 85-90c | 95,366 | LightGBM | 0.38504 |
| YES-leaning | 90-95c | 104,122 | LightGBM | 0.26385 |
| YES-leaning | 95-100c | 155,674 | LightGBM | 0.09823 |

## Full Score Matrix

| Market Side | Probability Bucket | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NO-leaning | 0-5c | 128,711 | 0.07406 | 0.09057 | 0.10543 | 0.07599 | 0.07540 | 0.06822 | 0.06686 | 0.06883 | 0.07578 |
| NO-leaning | 5-10c | 106,359 | 0.21171 | 0.22328 | 0.23356 | 0.21405 | 0.21045 | 0.20687 | 0.20438 | 0.20728 | 0.21637 |
| NO-leaning | 10-15c | 96,639 | 0.33941 | 0.34531 | 0.35112 | 0.34082 | 0.33811 | 0.33606 | 0.33397 | 0.33614 | 0.35432 |
| NO-leaning | 15-20c | 96,454 | 0.44032 | 0.44328 | 0.44645 | 0.44142 | 0.43990 | 0.43726 | 0.43478 | 0.43718 | 0.45585 |
| NO-leaning | 20-25c | 97,649 | 0.50952 | 0.51166 | 0.51375 | 0.51033 | 0.50871 | 0.50630 | 0.50415 | 0.50613 | 0.52476 |
| NO-leaning | 25-30c | 102,724 | 0.58432 | 0.58466 | 0.58532 | 0.58533 | 0.58482 | 0.58095 | 0.57832 | 0.58036 | 0.59223 |
| NO-leaning | 30-35c | 116,671 | 0.63591 | 0.63556 | 0.63558 | 0.63648 | 0.63685 | 0.63356 | 0.63155 | 0.63315 | 0.63877 |
| NO-leaning | 35-40c | 130,738 | 0.66009 | 0.66010 | 0.66021 | 0.66060 | 0.66026 | 0.65898 | 0.65717 | 0.65868 | 0.66289 |
| NO-leaning | 40-45c | 143,392 | 0.68327 | 0.68313 | 0.68308 | 0.68395 | 0.68318 | 0.68208 | 0.68031 | 0.68151 | 0.68356 |
| NO-leaning | 45-50c | 157,898 | 0.69228 | 0.69222 | 0.69219 | 0.69280 | 0.69218 | 0.69085 | 0.68906 | 0.69037 | 0.69209 |
| YES-leaning | 50-55c | 161,678 | 0.69144 | 0.69147 | 0.69149 | 0.69205 | 0.69160 | 0.69027 | 0.68868 | 0.69004 | 0.69157 |
| YES-leaning | 55-60c | 148,096 | 0.67898 | 0.67921 | 0.67938 | 0.67966 | 0.67859 | 0.67776 | 0.67605 | 0.67728 | 0.68058 |
| YES-leaning | 60-65c | 133,643 | 0.65977 | 0.66008 | 0.66035 | 0.66036 | 0.65971 | 0.65811 | 0.65653 | 0.65770 | 0.66281 |
| YES-leaning | 65-70c | 117,606 | 0.63092 | 0.63131 | 0.63173 | 0.63157 | 0.63097 | 0.62956 | 0.62743 | 0.62918 | 0.63570 |
| YES-leaning | 70-75c | 104,484 | 0.59106 | 0.59161 | 0.59232 | 0.59163 | 0.59140 | 0.58981 | 0.58773 | 0.58949 | 0.59788 |
| YES-leaning | 75-80c | 95,238 | 0.54783 | 0.54796 | 0.54877 | 0.54872 | 0.54875 | 0.54619 | 0.54395 | 0.54589 | 0.55420 |
| YES-leaning | 80-85c | 93,841 | 0.46815 | 0.46988 | 0.47219 | 0.46895 | 0.46822 | 0.46551 | 0.46303 | 0.46517 | 0.47872 |
| YES-leaning | 85-90c | 95,366 | 0.39014 | 0.39231 | 0.39584 | 0.39164 | 0.38982 | 0.38752 | 0.38504 | 0.38740 | 0.39761 |
| YES-leaning | 90-95c | 104,122 | 0.26953 | 0.27566 | 0.28285 | 0.27196 | 0.26892 | 0.26635 | 0.26385 | 0.26637 | 0.27276 |
| YES-leaning | 95-100c | 155,674 | 0.10300 | 0.11607 | 0.12922 | 0.10577 | 0.10507 | 0.09966 | 0.09823 | 0.10032 | 0.10858 |
