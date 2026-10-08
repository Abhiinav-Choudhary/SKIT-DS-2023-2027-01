# Heart Disease Final Test Evaluation Report

Target: Existing multiclass `num` target with classes [0, 1, 2, 3, 4].

The selected tuned pipeline was loaded from disk and evaluated once on the test split. No retraining or retuning was performed.

## Test Metrics

- Accuracy: 0.4783
- Precision macro: 0.3615
- Precision weighted: 0.5608
- Recall macro: 0.3980
- Recall weighted: 0.4783
- F1 macro: 0.3528
- F1 weighted: 0.5104
- ROC-AUC OVR macro: 0.755
- ROC-AUC OVR weighted: 0.7816

## Confusion Matrix

Class order: `[0, 1, 2, 3, 4]`

```text
[[42, 10, 1, 5, 4], [8, 15, 3, 9, 5], [0, 2, 6, 5, 3], [0, 5, 7, 1, 3], [0, 0, 1, 1, 2]]
```

## Training CV vs Test

| Metric | Training CV | Test | Delta test-CV |
| --- | ---: | ---: | ---: |
| accuracy | 0.5483 | 0.4783 | -0.07 |
| precision_macro | 0.4135 | 0.3615 | -0.052 |
| precision_weighted | 0.6014 | 0.5608 | -0.0406 |
| recall_macro | 0.4495 | 0.398 | -0.0515 |
| recall_weighted | 0.5483 | 0.4783 | -0.07 |
| f1_macro | 0.4139 | 0.3528 | -0.0611 |
| f1_weighted | 0.5671 | 0.5104 | -0.0567 |
| roc_auc_ovr_macro | 0.7659 | 0.755 | -0.0109 |
| roc_auc_ovr_weighted | 0.7954 | 0.7816 | -0.0138 |

## Per-Class Results

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.8400 | 0.6774 | 0.7500 | 62 |
| 1 | 0.4688 | 0.3750 | 0.4167 | 40 |
| 2 | 0.3333 | 0.3750 | 0.3529 | 16 |
| 3 | 0.0476 | 0.0625 | 0.0541 | 16 |
| 4 | 0.1176 | 0.5000 | 0.1905 | 4 |

## Files

- `reports\figures\17_heart_final_test_confusion_matrix.png`
- `reports/heart_disease_final_test_evaluation_report.json`
- `reports/heart_disease_final_test_evaluation_report.md`
