"""
Zain Kazi
Kaggle: Titanic Competition
Created on Sat Nov 21 
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

#%% Import and split data
# Read training data
train = pd.read_csv('C:/Users/zaink/Downloads/train.csv')

# Drop ticket number 
train = train.drop(['Ticket'], axis = 1)

# Make categorical variables dummy variables
train = pd.get_dummies(train)
#train =train[~train.isin([np.nan, np.inf, -np.inf]).any(1)]

# Read testing data
test = pd.read_csv('C:/Users/zaink/Downloads/test.csv')
test = test.drop(['Ticket'], axis = 1 )
test = pd.get_dummies(test)

# Align column from train and test sets
train,test = train.align(test, join='outer', axis=1)

# Separate independent and dependent variables 
xvars = train.drop('Survived', axis = 1)
yvars = train[['Survived']]

x_train, x_valid, y_train, y_valid = train_test_split(xvars, yvars, test_size=0.30, random_state=42)

#%% Optimize iterations
# Define list of estimator numbers for testing
iterations = [500,1000,1500,2000]

# Define lists for accuracy and area under curve metrics
auc = []
acc = []

# Train model for each value in iterations
for i in iterations:
    cat =  CatBoostClassifier(iterations = i,
                              learning_rate = .2,
                              depth = 2)
                            
    # Fit model on training set
    cat.fit(x_train, y_train)

    # Create predictions for validation and get accuracy
    cat_predictions = cat.predict(x_valid)
    cat_probs = cat.predict_proba(x_valid)
    
    # Obtain AUC and acuracies
    aucscore = roc_auc_score(y_valid, cat_probs[:,1])
    accuracy = accuracy_score(y_valid ,cat_predictions)
    
    # Append AUCs and accuracies to list
    auc.append(aucscore)
    acc.append(accuracy)

# Plot learning rate and AUC score
plt.scatter(iterations, acc)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Number of Trees")
plt.figure()

#%% Optimize learning rate
# Define list of estimator numbers for testing
lr = [.05,.1,.15,.2,.25,.3]

# Define lists for accuracy and area under curve metrics
auc = []
acc = []

# Train model for each value in iterations
for i in lr:
    cat =  CatBoostClassifier(iterations = 1000,
                              learning_rate = i,
                              depth = 2)
                            
    # Fit model on training set
    cat.fit(x_train, y_train)

    # Create predictions for validation and get accuracy
    cat_predictions = cat.predict(x_valid)
    cat_probs = cat.predict_proba(x_valid)
    
    # Obtain AUC and acuracies
    aucscore = roc_auc_score(y_valid, cat_probs[:,1])
    accuracy = accuracy_score(y_valid ,cat_predictions)
    
    # Append AUCs and accuracies to list
    auc.append(aucscore)
    acc.append(accuracy)

# Plot learning rate and AUC score
plt.scatter(lr, acc)
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Learning Rate")
plt.figure()

#%% Optimize learning rate
# Define list of estimator numbers for testing
depth = [2,3,4,5]

# Define lists for accuracy and area under curve metrics
auc = []
acc = []

# Train model for each value in iterations
for i in depth:
    cat =  CatBoostClassifier(iterations = 1000,
                              learning_rate = .2,
                              depth = i)
                            
    # Fit model on training set
    cat.fit(x_train, y_train)

    # Create predictions for validation and get accuracy
    cat_predictions = cat.predict(x_valid)
    cat_probs = cat.predict_proba(x_valid)
    
    # Obtain AUC and acuracies
    aucscore = roc_auc_score(y_valid, cat_probs[:,1])
    accuracy = accuracy_score(y_valid ,cat_predictions)
    
    # Append AUCs and accuracies to list
    auc.append(aucscore)
    acc.append(accuracy)

# Plot learning rate and AUC score
plt.scatter(depth, acc)
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Depth")
plt.figure()

#%%
# Optimal probability cutoff with youden (Not used based on results)
fpr, tpr, thresholds = roc_curve(y_valid, cat_probs[:,1])
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold: " + str(optimal_threshold))

#%% Create predictions for test
# Define final model (validation accuracy approximately 81%)
cat =  CatBoostClassifier(iterations = 1000,
                          learning_rate = .2,
                          depth = 2)

#Fit model on training/validation combined set
cat.fit(xvars, yvars)
predictions = pd.DataFrame(cat.predict(test))

# Create predictions for test (Approximately 79%: About top 9% on Kaggle)
cat_predictions = pd.DataFrame({'Survived': predictions.iloc[:,0]})
submission = pd.merge(test['PassengerId'], cat_predictions, left_index=True, right_index = True)
submission.to_csv('C:/Users/zaink/Downloads/Titanic_Submission.csv', index = False)




