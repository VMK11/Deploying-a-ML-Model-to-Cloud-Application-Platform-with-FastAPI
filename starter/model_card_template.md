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
Bias and Equity Concerns: The dataset may harbor biases due to its collection methodology. For instance, certain demographic groups might be underrepresented, causing the predictive model to produce skewed results. This imbalance may lead to unfair outcomes for specific individuals or communities.

Protection of Personal Data: The dataset holds delicate information like earnings, job type, and educational qualifications. Safeguarding this data to prevent unauthorized access or harmful usage is paramount.

Integrity and Precision of Data: The dataset could contain mistakes or inaccuracies that may affect the quality of the predictive outcomes. Therefore, it's crucial to rigorously verify the dataset's reliability for its intended purpose.

Openness and Interpretability: Being clear about the data collection, data handling, and application procedures is essential for fostering trust among users and key stakeholders. This also helps to ensure the credibility and fairness of the generated predictions.

Answerability in Algorithmic Decisions: Implementing machine learning models for prediction introduces an ethical dimension to data analytics, raising questions about who or what is accountable for the decisions made. Ensuring that the algorithms operate in an equitable and transparent manner, and that their outcomes can be both interpreted and contested, is essential.

## Caveats and Recommendations
Census income dataset may have class imbalance problem. However, in this model, appropriate data selection or weighting for class is not conducted.