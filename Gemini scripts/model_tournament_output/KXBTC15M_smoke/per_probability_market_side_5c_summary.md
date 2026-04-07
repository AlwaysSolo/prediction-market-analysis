# KXBTC15M Tournament By Market Probability And Market Outcome Side

- Metric: `log_loss` on the held-out test split
- Probability buckets: implied market probability in `5`-cent bins
- Market side buckets: `NO-leaning` for implied probability `< 50c`, `YES-leaning` for `>= 50c`
- Models: same tournament lineup as `smarterpred.py`

## Winners By Joint Bucket

| Market Side | Probability Bucket | Test Rows | Best Model | Best Log-Loss |
| --- | --- | ---: | --- | ---: |
| NO-leaning | 0-5c | 186 | XGBoost | 0.07785 |
| NO-leaning | 5-10c | 176 | Momentum-Adjusted Gaussian | 0.21368 |
| NO-leaning | 10-15c | 176 | Momentum-Adjusted Gaussian | 0.35409 |
| NO-leaning | 15-20c | 171 | LightGBM | 0.44203 |
| NO-leaning | 20-25c | 168 | XGBoost | 0.52483 |
| NO-leaning | 25-30c | 169 | XGBoost | 0.62135 |
| NO-leaning | 30-35c | 195 | Momentum-Adjusted Gaussian | 0.63180 |
| NO-leaning | 35-40c | 234 | Machine Learning (Logistic Regression) | 0.62079 |
| NO-leaning | 40-45c | 234 | LightGBM | 0.68099 |
| NO-leaning | 45-50c | 249 | Machine Learning (Logistic Regression) | 0.69321 |
| YES-leaning | 50-55c | 299 | CatBoost | 0.68772 |
| YES-leaning | 55-60c | 239 | Machine Learning (Logistic Regression) | 0.67711 |
| YES-leaning | 60-65c | 241 | LightGBM | 0.64469 |
| YES-leaning | 65-70c | 191 | XGBoost | 0.60172 |
| YES-leaning | 70-75c | 168 | LightGBM | 0.60736 |
| YES-leaning | 75-80c | 170 | Momentum-Adjusted Gaussian | 0.48356 |
| YES-leaning | 80-85c | 146 | XGBoost | 0.44119 |
| YES-leaning | 85-90c | 172 | PySR (Symbolic Regression) | 0.47790 |
| YES-leaning | 90-95c | 148 | Student-T df=3 | 0.36834 |
| YES-leaning | 95-100c | 268 | PySR (Symbolic Regression) | 0.09420 |

## Full Score Matrix

| Market Side | Probability Bucket | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NO-leaning | 0-5c | 186 | 0.07916 | 0.09552 | 0.11016 | 0.08777 | 0.08186 | 0.07785 | 0.08492 | 0.07995 | 0.08417 |
| NO-leaning | 5-10c | 176 | 0.21716 | 0.22818 | 0.23814 | 0.21368 | 0.21839 | 0.22011 | 0.21868 | 0.21596 | 0.21823 |
| NO-leaning | 10-15c | 176 | 0.35477 | 0.35879 | 0.36361 | 0.35409 | 0.35544 | 0.36345 | 0.35806 | 0.36500 | 0.36413 |
| NO-leaning | 15-20c | 171 | 0.44899 | 0.45091 | 0.45353 | 0.44521 | 0.45158 | 0.44480 | 0.44203 | 0.44955 | 0.46053 |
| NO-leaning | 20-25c | 168 | 0.52522 | 0.52608 | 0.52743 | 0.52491 | 0.52656 | 0.52483 | 0.52818 | 0.52770 | 0.53383 |
| NO-leaning | 25-30c | 169 | 0.63999 | 0.63660 | 0.63500 | 0.64187 | 0.64366 | 0.62135 | 0.62515 | 0.62865 | 0.63238 |
| NO-leaning | 30-35c | 195 | 0.63507 | 0.63478 | 0.63482 | 0.63180 | 0.63938 | 0.63311 | 0.63880 | 0.63447 | 0.63723 |
| NO-leaning | 35-40c | 234 | 0.62279 | 0.62483 | 0.62620 | 0.62373 | 0.62079 | 0.62487 | 0.62879 | 0.62809 | 0.63533 |
| NO-leaning | 40-45c | 234 | 0.68430 | 0.68410 | 0.68402 | 0.68875 | 0.68107 | 0.68197 | 0.68099 | 0.68703 | 0.68405 |
| NO-leaning | 45-50c | 249 | 0.69663 | 0.69636 | 0.69620 | 0.70088 | 0.69321 | 0.70134 | 0.70032 | 0.70138 | 0.69519 |
| YES-leaning | 50-55c | 299 | 0.69315 | 0.69309 | 0.69306 | 0.69449 | 0.69309 | 0.69107 | 0.69519 | 0.68772 | 0.69291 |
| YES-leaning | 55-60c | 239 | 0.67759 | 0.67789 | 0.67810 | 0.67834 | 0.67711 | 0.68052 | 0.68460 | 0.68004 | 0.67981 |
| YES-leaning | 60-65c | 241 | 0.65291 | 0.65358 | 0.65408 | 0.65179 | 0.65380 | 0.65650 | 0.64469 | 0.65419 | 0.65806 |
| YES-leaning | 65-70c | 191 | 0.60805 | 0.60979 | 0.61104 | 0.60784 | 0.60673 | 0.60172 | 0.60233 | 0.60384 | 0.61925 |
| YES-leaning | 70-75c | 168 | 0.61788 | 0.61655 | 0.61614 | 0.61649 | 0.62194 | 0.61133 | 0.60736 | 0.61319 | 0.61795 |
| YES-leaning | 75-80c | 170 | 0.48688 | 0.49182 | 0.49544 | 0.48356 | 0.48682 | 0.48767 | 0.49118 | 0.48800 | 0.50859 |
| YES-leaning | 80-85c | 146 | 0.45294 | 0.45664 | 0.45998 | 0.45178 | 0.45396 | 0.44119 | 0.44383 | 0.44871 | 0.46693 |
| YES-leaning | 85-90c | 172 | 0.48998 | 0.48133 | 0.47901 | 0.48787 | 0.49396 | 0.49621 | 0.49151 | 0.49970 | 0.47790 |
| YES-leaning | 90-95c | 148 | 0.37863 | 0.36889 | 0.36834 | 0.37652 | 0.37874 | 0.38002 | 0.37900 | 0.38153 | 0.37487 |
| YES-leaning | 95-100c | 268 | 0.09641 | 0.10919 | 0.12259 | 0.10108 | 0.09722 | 0.09972 | 0.10161 | 0.10308 | 0.09420 |
