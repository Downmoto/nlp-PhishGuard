# PhishGuard Evaluation Report

## Classification Report

```
              precision    recall  f1-score   support

  Legitimate       0.99      1.00      0.99      5940
    Phishing       1.00      0.99      0.99      6434

    accuracy                           0.99     12374
   macro avg       0.99      0.99      0.99     12374
weighted avg       0.99      0.99      0.99     12374

```

## ROC-AUC

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9998 |

## Confusion Matrix

|  | Predicted Legitimate | Predicted Phishing |
|--|--|--|
| **Actual Legitimate** | 5924 | 16 |
| **Actual Phishing** | 49 | 6385 |

## Figures

- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`
