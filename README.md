# Installation
```
pip install -r requirements.txt
```


# Usage
'''
usage: rf.py [-h] --id ID --data DATA --outcome OUTCOME --cv CV --ntrees NTREES

Run the RF

optional arguments:
  -h, --help         show this help message and exit
  --id ID            The name of this dataset
  --data DATA        The input CSV data file
  --outcome OUTCOME  The outcome variable
  --cv CV            Number of cross validation iterations
  --ntrees NTREES    Number of trees for the RF
'''

Make sure that you have only numbers in your data input csv

# Example
```
python rf.py --id test --data data.csv --outcome Class --cv 10 --ntrees 500
```
