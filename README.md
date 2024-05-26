Credit-Risk-Classification Report for Module 20
Module20Challenge
# Module 12 Report Template

## Overview of the Analysis

* The purpose of the analysis:  Puropse of this analysis is to use various techniques to train and evauate a model based on loan risk.
    
*   The financial information used for this analysis is a dataset of historic lending activity from a peer to peer lending service company.
*   The model will use logistic regression to predict both the both the `0` (healthy loan) and `1` (high-risk loan) labels?

    #Columns include:
    *loan_size
    *interest_rate
    *borrower_income
    *debt_to_income in decimal
    *num_of_accounts
    *derogatory_marks
    *total_debt
    *loan_status
      
* Setting the Variables to be:  Dependent Variable: "loan_status" (y) and Independent Variables: "all other columns above" (x)

* Describe the stages of the machine learning process you went through as part of this analysis.
   # Pre-process the data:
        #Reading in the data
        #Separating the data into labels (y) and features (X)
        #Reviewed the y data series and X variables to confirm correct categorization
        #Splite the data into train_test_split and assigned a random_state
        #Created a standard scaler instance
        #Fitted scaler and scaled data
        #Reviewed trained and tested data
    * Step 1: Fit a logistic regression model by using the training data (X_train and y_train)
            #Instigated the model and assigned a parameter of 1 to model
            #Fit logistic regression model using training data
    * Step 2: Saved the predictions on the testing data lables by using testing feature data(X_test) and the fitted model.
            #Made a prediction using the testing data
    * Step 3: Evaluate the model's performance by doing the following
            #Generate a Confusion MatriX
            #Print the Classification Report
       
* Briefly touch on any method you used: Created a logistic regression model with original data

## Results

 * Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.
    Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

## Confusion Matrix Results:
   *    ACTUAL 0 / Predicted 0 = 18663 (True Negative)(.9628) - Correct Answer
   *    ACTUAL 0 / Predicted 1 = 102 (False Positive)(.0052)
   *    ACTUAL 1 / Predicted 0 = 56 (False Negative) (.0028)
   *    ACTUAL 1 / Predicted 1 = 563 (True Positive) (.0290) - Correct Answer
##    To determine if the the model is accurate and precise the aggregate sums of each column summed across and down equal = 19,384
*    RESULTS: This model is considered both accurate and precise.   
 
##   Description of Model 1 Accuracy, Precision, and Recall scores.  
#    A classification report holds the test results so we can assess and evaluate the number of predicted occurrences for each class:
    *Accuracy is how often the model is correct—the ratio of correctly predicted observations to the total number of observations.
    (TP + TN) / (TP + TN + FP + FN)
    
    *Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.
    TP / (TP + FP) - High precision relates to a low false positive rate.

    *Recall is the ratio of correctly predicted positive observations to all predicted observations for that class. 
    TP / (TP + FN) - High recall correlates to a more comprehensive output and a low false negative rate.

## Summary

We were able to determine through the Confusion Matrix that this model is both accurate and precise, which is supported by the numbers in the Classification Report.
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384


##This model was excellent at predicting with a 99% accuracy the likelihood of healthy (0) and high risk (1) labels (loans).
##The precision, recall and f1-score for forecasting healthy loans (0) is extremely high (1.0, .99, 1.0, 18765/19384).
##The precision, recall and f1-score for predicting high-risk loans (1) is high (.85, .91, .88, 619/19384).

##The model is more effective at predicting healthy loans than predicting the precision of unhealthy or high-risk loans.  However I would still recommend using the model.
