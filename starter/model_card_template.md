# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Trained a RandomForest model from the scikit-learn package - scikit-learn version = 1.0.2. 
Hyper-parameter tuning, with 3-fold cross-validation for 10 candidates was applied.

## Intended Use
The model will be used to predict the salary of a person based on attributes defined in census.csv 
The users of the potential application, would be recruiters, HR, etc.

## Training Data
The dataset was sourced from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). For training purposes, the features were transformed using a One Hot Encoder, while the labels were processed with a label binarizer.

## Evaluation Data
The assessment is carried out using a randomly chosen subset that comprises 20% of the entire dataset.

## Metrics
The performance of the model was gauged through metrics such as precision, recall, and the F1 score.
Precision: 0.78, Recall: 0.6, FBeta: 0.68

## Ethical Considerations
There's a danger in implying that the features present in this dataset are the sole determinants of an individual's income, even though we are aware that this is not the complete picture.

## Caveats and Recommendations
Census income dataset may have class imbalance problem. However, in this model, appropriate data selection or weighting for class is not conducted.