# Digital Payment Transaction Reversal Prediction

## What this project does
Predicts whether a digital payment transaction will be reversed, using transaction 
details like type, amount, device, risk level, and account info, with Logistic Regression.

## Dataset
Fintech transaction dataset with 5,000 records and 22 features, including transaction 
type, amount, account balance, credit score, KYC level, and risk level.

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook

## What I did
- Explored the target variable and found the data was imbalanced (only 14.1% of 
  transactions were reversed)
- Checked for missing values and duplicates (found none — clean dataset)
- Extracted hour/day/month features from the timestamp column
- Removed identifier columns that don't help prediction (IDs, names) to avoid data leakage
- Built a correlation heatmap and found numeric features had almost no relationship 
  with reversal — an important finding, not a failure
- Encoded categorical features and trained a Logistic Regression model with 
  class balancing to handle the imbalance
- Evaluated using Accuracy, Precision, Recall, F1 Score, and ROC-AUC (not just accuracy, 
  since the data is imbalanced)

## Results
| Metric     | Score |
|------------|-------|
| Accuracy   | 0.529 |
| Precision  | 0.150 |
| Recall     | 0.511 |
| F1 Score   | 0.232 |
| ROC-AUC    | 0.521 |

## Key Insight
The model performs close to random guessing (ROC-AUC ≈ 0.52). The correlation analysis 
showed none of the available numeric features have a meaningful relationship with 
transaction reversal — meaning this dataset likely doesn't contain the real drivers 
of reversal, rather than the model being poorly built.

## How to run it
Open `payment_transaction_reversal_prediction.ipynb` in Jupyter Notebook or Google 
Colab and run all cells.
