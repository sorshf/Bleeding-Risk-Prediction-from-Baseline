

# Supervised and Unsupervised Machine Learning Analysis of Bleeding Status in Venous Thromboembolism Patients

## Description
The performance of the clinical models used to predict bleeding in patients on anticoagulants with Venous Thromboembolism have stayed at around 70% as measured with area under ROC curve. In this paper, we explored the baseline information of 2542 patients using unsupervised dimentionality reduction and clustering algorithms and demonstrated high degree of similarity between their baseline dataset. Furthermore, we observed no benefits in using supervised machine learning algorithms to predict bleeding risk compared to the conventional clinical models.

To perform the 5-fold nested cross-validation for all the models:
```
python just_baseline_experiments.py experiment all_models
```
To perform the unsupervised analysis:
```
python just_baseline_experiments.py unsupervised
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


