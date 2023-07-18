# DSLR


## Data Analysis
The following commands can be used to obtain statistical information about the data.

```bash
python describe.py dataset/dataset_train.csv
```

<img src='images/describe.png' width='700'>

<br></br>

## Data Visualization
### Histogram

```bash
python histogram.py dataset/dataset_train.csv
```

<img src='images/histogram.png' width='700'>

### Scatter plot

```bash
python scatter_plot.py --data_file_path dataset/dataset_train.csv --x_item Astronomy --y_item Herbology
```

<img src='images/scatter_plot.png' width='700'>

### Pair plot

```bash
python pair_plot.py dataset/dataset_train.csv
```

<img src='images/pair_plot.png' width='700'>

<br></br>

## Logistic Regression

```bash
python logreg_train.py dataset/dataset_train.csv
```
