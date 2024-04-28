

# [Machine learning analysis of bleeding status in venous thromboembolism patients](https://doi.org/10.1016/j.rpth.2024.102403)

## Description
The performance of the clinical models used to predict bleeding in patients on anticoagulants with Venous Thromboembolism has stayed at around 70% as measured with the area under the ROC curve. In this paper, we explored the baseline information of 2542 patients using unsupervised dimensionality reduction and clustering algorithms and demonstrated a high degree of similarity between their baseline datasets. Furthermore, we observed no benefits in using supervised machine learning algorithms to predict bleeding risk compared to the conventional clinical models.

To perform the 5-fold nested cross-validation for all the models:
```
python just_baseline_experiments.py experiment
```
To perform the unsupervised analysis:
```
python just_baseline_experiments.py unsupervised
```
To perform feature-selection with SFS method:

```
python just_baseline_experiments.py feature_selection
```

# Prerequisites
### Python Environment
[Conda](https://docs.conda.io/) was used to manage python packages and their dependencies. Following commands could be used to install and activate the correct environment:
```
conda env create -f environment.yml
conda activate bleeding-risk-env
```


### Patient Dataset
The raw dataset can be obtained by contacting the corresponding author Dr. Phil Wells (pwells@toh.ca).


