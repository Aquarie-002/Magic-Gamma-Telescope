# MAGIC Gamma Telescope — Gamma/Hadron Classification Using SVM

## Project Overview

This project explores the application of **Support Vector Machines (SVM)** to distinguish gamma-ray signals from hadronic background events using simulated data from the MAGIC Gamma Telescope.

The main objective is not only to achieve high classification performance but also to investigate the trade-off between gamma-ray detection and background contamination, particularly under low false positive rate (FPR) conditions.

## Dataset

**Source:** [UCI Machine Learning Repository — MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)

The dataset contains 19,020 Monte Carlo-simulated events described by 10 numerical features derived from atmospheric Cherenkov shower images.

After removing 115 exact duplicate records, 18,905 observations were retained for analysis.

- **Gamma (g):** Signal, positive class
- **Hadron (h):** Background, negative class

## Methodology

The analysis was conducted in Python using scikit-learn and included:

1. **Exploratory Data Analysis:** Examining feature distributions, outliers, correlations, and differences between gamma and hadron events.
2. **Data Preprocessing:** Removing exact duplicates, performing an 80/20 stratified train-test split, and applying StandardScaler within the modeling pipeline.
3. **Model Development:** Comparing LinearSVC and RBF-kernel SVM with hyperparameter tuning using 5-fold stratified cross-validation.
4. **Model Evaluation:** Assessing accuracy, balanced accuracy, ROC AUC, partial ROC AUC (pAUC), and false positive rates.
5. **Threshold Optimization:** Investigating how stricter decision thresholds reduce hadronic background contamination at the expense of gamma-ray recall.

## Results

The tuned RBF-SVM model achieved the strongest validation pAUC among the evaluated models, with the following test-set results:

| Metric | Result |
|---|---:|
| Accuracy | 86.70% |
| Balanced Accuracy | 85.08% |
| ROC AUC | 0.9257 |
| Standardized partial ROC AUC (FPR ≤ 0.10) | 0.7467 |
| False Positive Rate | 20.23% |

Adjusting the decision threshold using out-of-fold training predictions reduced the test false positive rate from **20.23% to 5.32%**, while gamma-ray recall decreased from **90.39% to 56.73%**.

This illustrates an important trade-off: improving background rejection can significantly reduce the number of gamma-ray events successfully detected.

## Key Takeaways

This project highlights the importance of:

- Understanding feature distributions and preprocessing requirements before model development.
- Selecting evaluation metrics based on the scientific objective rather than accuracy alone.
- Understanding the relationship between classification thresholds, false positives, and signal detection.
- Recognizing the limitations of machine learning models trained on simulated data.

The results are based on Monte Carlo simulations and should not be interpreted as validated performance on real telescope observations.

## Tools & Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Support Vector Machines

## Project Files

- `MAGIC_Gamma_SVM.ipynb` — Python notebook containing data exploration, preprocessing, model training, and evaluation.
- `Laporan_MAGIC_SVM_10324002.pdf` — Detailed technical report in Indonesian, including visualizations, experimental results, and discussion.

## References

Bock, R. (2004). *MAGIC Gamma Telescope*. UCI Machine Learning Repository. https://doi.org/10.24432/C52C8B

## Author

**Praneeta Bhojwani**  
Astronomy Undergraduate — Institut Teknologi Bandung (ITB)

*Interested in the intersection of astronomy, statistics, and machine learning.*
