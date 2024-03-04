# %%
# Built-in python libraries
import os
import re

# General libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Preprocessing and training libraries
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import pickle
import multiprocessing

# Custom Setup
%matplotlib inline
# matplotlib.style.use('seaborn-paper')
# sns.set(style='darkgrid')
# sns.set_theme(context='notebook', style='darkgrid', palette='husl')
# pd.set_option('display.max_columns', None) # display all columns in pandas

sns.set(style='darkgrid')
sns.set_theme(context='notebook', style='darkgrid', palette='deep') # Changed palette to 'deep'
import pandas as pd
pd.set_option('display.max_columns', None) # display all columns in pandas

# %%
import warnings
# Filter out future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
num_cores = multiprocessing.cpu_count()
cores_used = num_cores//5
pd.options.mode.chained_assignment = None  # default='warn'

# %%
dataset_path = '../data/'
save_fig_path = '../figures/'

df = pd.read_csv(os.path.join(dataset_path, 'C2T1_Train.csv'), na_values='?') # make ?? as nan
df.rename(columns={"encounter_id2": "encounter_id", "patient_nbr2": "patient_nbr"}, inplace=True)

print("The shape of the dataset is {}.\n\n".format(df.shape))

print(df.head())

# %%
X = df.drop(['readmitted'],axis=1)
y = df['readmitted'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.1,
                                                    random_state=42)

# %%
train_data = pd.concat([X_train,y_train], axis=1)
print(train_data.head())

# %%
cat_cols = [
    'race', 'gender', 'age', 'weight', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id', 'payer_code',
    'medical_specialty', 'max_glu_serum',
    'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
    'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
    'diabetesMed'
]
num_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

ID_cols = ["encounter_id", "patient_nbr"]
diag_cols = ['diag_1', 'diag_2', 'diag_3']


# %%
# Categorical Columns counts
fig, axs = plt.subplots(nrows=9, ncols=4, figsize=(30, 80))
plt.subplots_adjust(hspace=0.5)

for column, ax in zip(cat_cols, axs.ravel()):
    sns.countplot(x=train_data[column], ax=ax, palette='viridis')  # Changed palette to 'viridis'
    ax.set_xlabel(column, fontsize=15)
    ax.set_ylabel('Count', fontsize=15)  # Added y-axis label
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust tick label size
    ax.set_title(f'Count of {column}', fontsize=18)  # Added title
    for container in ax.containers:
        ax.bar_label(container, fontsize=12)

plt.savefig(save_fig_path+'value_count_of_each_cat_col.png', bbox_inches='tight')  
plt.show()


# %%
cols_to_drop = ['weight','medical_specialty','chlorpropamide','acetohexamide','weight','tolbutamide',
    'acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton',
    'glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone',
    'metformin-rosiglitazone','metformin-pioglitazone']

# %%
train_data[num_cols].describe().T


# %%
train_data[num_cols].hist(figsize=(20, 20), bins=50)
plt.suptitle('The Distribution Of Numerical Columns', fontsize=18)
plt.tight_layout(pad = 2)
plt.savefig(save_fig_path+'distribution_of_n.png', bbox_inches='tight')  
plt.show()

# %%
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
plt.subplots_adjust(hspace=0.5)
fig.suptitle("Checking For Outliers Using Boxplot", fontsize=18, y=0.95)

for column, ax in zip(num_cols, axs.ravel()):
    ax.boxplot(train_data[column], patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'), whiskerprops=dict(color='black'), capprops=dict(color='black'), medianprops=dict(color='red'))
    ax.set_xlabel(column, fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)  # Put grid behind the plots

plt.savefig(save_fig_path+'check_outliers_using_boxplot.png')
plt.show()

# %%
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
plt.subplots_adjust(hspace=0.5)
fig.suptitle("Mean for Numerical Values for Different Target Classes",
             fontsize=18,
             y=0.95)

for column, ax in zip(num_cols, axs.ravel()):
    sns.barplot(x=train_data['readmitted'], y=train_data[column], ax=ax, palette='coolwarm')
    ax.set_xlabel("Readmission", fontsize=15)
    ax.set_ylabel(f"Average {column}", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    for container in ax.containers:
        ax.bar_label(container, fontsize=12, fmt='%.2f', label_type='edge')
plt.savefig(save_fig_path+'mean_of_num_cols_for_each_target_class.png', bbox_inches='tight')
plt.show()

# %%
print(train_data.groupby(['patient_nbr'])['patient_nbr'].count().sort_values(ascending=False))

# %%
train_data[diag_cols[0]].value_counts()


# %%
X_num = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

X_cat = [
    'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'max_glu_serum', 'A1Cresult', 'metformin',
    'repaglinide', 'nateglinide', 'glimepiride', 'glipizide', 'glyburide',
    'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed',
    'payer_code'
]

X_diag = ['diag_1', 'diag_2', 'diag_3']

X_id = ['encounter_id', "patient_nbr"]

# %%
class IDColumnsTransformer:
    def fit(self, X, y=None):
        self.columns = ['encounter_id', 'patient_nbr']
        self.history = X[self.columns].copy(
            deep=True).set_index('encounter_id').sort_index()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy(deep=True)
        new_data = X_copy[self.columns].set_index('encounter_id').sort_index()
        self.history = pd.concat([self.history, new_data])
        self.history = (self.history.reset_index().drop_duplicates(
            subset='encounter_id',
            keep='last').set_index('encounter_id').sort_index())

        ###visit number
        visits = self.history[['patient_nbr']]

        df_visits = visits.groupby(
            'patient_nbr',
            group_keys=False).apply(lambda x: x.rank(method='first'))
        ans_visits = df_visits.loc[new_data.index, :]['patient_nbr']
        X_copy = pd.merge(X_copy,
                          ans_visits,
                          left_on='encounter_id',
                          right_on=ans_visits.index).rename(
                              {
                                  'patient_nbr_x': 'patient_nbr',
                                  'patient_nbr_y': 'visit_no'
                              },
                              axis=1)

        #how many visits
        patient_visit_count = X_copy['patient_nbr'].map(
            visits['patient_nbr'].value_counts())

        X_copy['visit_times'] = patient_visit_count

        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

# %%
class DiagColumnsTransformer():
    def map_diag(self, s):
        if not isinstance(s, str):
            #handle nans and other types
            return s

        if re.findall(r'^39\d|4[0-5]\d|785$', s):
            '''390–459, 785'''
            return 'Circulatory'

        if re.findall(r'^4[6-9]\d|5[0-1]\d|786$', s):
            '''460–519, 786'''
            return 'Respiratory'

        if re.findall(r'^5[2-7]\d|787$', s):
            '''520–579, 787'''
            return 'Digestive'

        if re.findall(r'^250.*', s):
            '''250.xx'''
            return 'Diabetes'

        if re.findall(r'^[8-9]\d\d$', s): 
            '''800–999'''
            return 'Injury'

        if re.findall(r'^7[1-3]\d$', s):
            '''710–739'''
            return 'Musculoskeletal'

        if re.findall(r'^5[8-9]\d|6[0-2]\d|788$', s):
            '''580–629, 788'''
            return 'Genitourinary'

        if re.findall(r'^[EV]', s, re.I):
            '''contains V or E with codes'''
            return 'other'
        else:
            # we need at least one condition to return just incase
            return 'Neoplasms' 

        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy(deep=True)
        for col in ['diag_1', 'diag_2', 'diag_3']:
            X_copy[col] = X[col].map(self.map_diag)
        return X_copy
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y) 

# %%
cat_pipeline = Pipeline([("mode imputer"  , SimpleImputer(strategy="most_frequent")),
                         ("target encoder", TargetEncoder())])

num_pipeline = Pipeline([('median imputer', SimpleImputer(strategy="median")),
                         ("MinMax scaler" , MinMaxScaler())])

diag_pipeline = Pipeline([('Diagnosis transformer', DiagColumnsTransformer()),
                          ("mode imputer"         , SimpleImputer(strategy="most_frequent")),
                          ("target encoder"       , TargetEncoder())])

id_pipeline = Pipeline([("encoder"       , IDColumnsTransformer()),
                        ("Min max scaler", MinMaxScaler())])

full_pipeline = ColumnTransformer(transformers=[("num", num_pipeline, X_num),
                                                ("cat", cat_pipeline, X_cat),
                                                ("id", id_pipeline, X_id),
                                                ("diag", diag_pipeline, X_diag)],
                                                remainder='drop') # remove other columns

# %%
le = LabelEncoder()
y_train_prepared = le.fit_transform(y_train)
print("Shape of y_train:", y_train_prepared.shape)


# %%
y_test_prepared = le.transform(y_test)
print("Shape of y_test:", y_test_prepared.shape)

# %%
X_train_prepared = full_pipeline.fit_transform(X_train,y_train_prepared)
print("Shape of X_train:", X_train_prepared.shape)

# %%
X_test_prepared = full_pipeline.transform(X_test)
print("Shape of X_test:", X_test_prepared.shape)

# %%


# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

# Define models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    "XGBoost": xgb.XGBClassifier(random_state=42, verbosity=0)
}

# Define parameter grids for hyperparameter tuning
param_grids = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100], "max_iter": [100, 500, 1000], "solver": ['saga']},
    "Random Forest": {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20, 30]},
    "Decision Tree": {"max_depth": [None, 10, 20, 30]},
    "CatBoost": {"learning_rate": [0.01, 0.05, 0.1], "depth": [4, 6, 10]},
    "XGBoost": {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 6, 9]}
}

best_model = None
best_f1_score = 0

for name in models:
    print(f"Training and tuning {name}")
    model = models[name]
    param_grid = param_grids[name]
    
    # Hyper-parameter tuning using GridSearchCV
    clf = GridSearchCV(model, param_grid, scoring="f1_micro", cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), n_jobs = num_cores, verbose = 1)
    
    clf.fit(X_train_prepared, y_train_prepared)
    best_estimator = clf.best_estimator_
    print(f"Best parameters for {name}: {clf.best_params_}")
    
    # Predict on the test set
    predictions = best_estimator.predict(X_test_prepared)

    # Accuracy
    accuracy = accuracy_score(y_test_prepared, predictions)
    print(f"\n Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\n ##### Classification Report: #####")
    print(classification_report(y_test_prepared, predictions))

    # Confusion Matrix
    cm = confusion_matrix(y_test_prepared, predictions)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'{name} - Confusion Matrix')
    plt.savefig(save_fig_path+'_'+name+'_Confusion_Matrix', bbox_inches='tight')
    plt.show()
    
    # Evaluate F1 score on the test set
    f1 = f1_score(y_test_prepared, predictions, average="micro")
    print(f"F1 Score on Test set: {f1}")

    file_name = '../models/'+name+'.pkl'
    dump_file = open(file_name, 'wb')
    pickle.dump(best_estimator, dump_file)
    
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = best_estimator

print("Best Model:")
print(best_model)
print("Best F1 Score on Test set:", best_f1_score)


# %%
test_df = pd.read_csv(os.path.join(dataset_path, 'C2T1_Test.csv'), na_values='?')
test_df_prepared = full_pipeline.transform(test_df)
test_df_predictions = final_model.predict(test_df_prepared)
y_test_predicted = le.inverse_transform(test_df_predictions)
test_df['readmitted'] = y_test_predicted
test_df[['encounter_id','patient_nbr', 'readmitted']].to_csv('C2T1_Test_Labled.csv', index=False)

# %%


# %%


# %%


# %%
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from catboost import CatBoostClassifier
# models = {
#     "Logistic Regression": LogisticRegression(random_state=42),
#     "Random Forest": RandomForestClassifier(random_state=42),
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "CatBoost": CatBoostClassifier(random_state=42, verbose=0)  # verbose=0 to keep the output clean
# }


# %%
# # Assuming X_train_prepared, X_test_prepared, y_train_prepared, y_test_prepared are already defined

# # Store evaluation metrics
# evaluation_results = {}

# for name, model in models.items():
#     print(f"Training {name}...")
#     model.fit(X_train_prepared, y_train_prepared)
    
#     # Predict on training and test sets
#     train_pred = model.predict(X_train_prepared)
#     test_pred = model.predict(X_test_prepared)
    
#     # Evaluation
#     train_f1 = f1_score(y_train_prepared, train_pred, average="micro")
#     test_f1 = f1_score(y_test_prepared, test_pred, average="micro")
    
#     evaluation_results[name] = {
#         "train_f1": train_f1,
#         "test_f1": test_f1
#     }
    
#     # Confusion Matrix for test set
#     cm = confusion_matrix(y_test_prepared, test_pred)
#     plt.figure(figsize=(5,4))
#     sns.heatmap(cm, annot=True, fmt='g', cmap="rocket_r")
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.title(f'{name} - Test data Confusion Matrix')
#     plt.show()
    
#     # Classification report for test set
#     print(classification_report(y_test_prepared, test_pred))
    
#     # Store or display other evaluation metrics as needed


# %%
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from catboost import CatBoostClassifier

# # Define your models
# models = {
#     "LogisticRegression": LogisticRegression(random_state=42),
#     "RandomForest": RandomForestClassifier(random_state=42),
#     "DecisionTree": DecisionTreeClassifier(random_state=42),
#     "CatBoost": CatBoostClassifier(random_state=42, verbose=0)  # `verbose=0` to silence CatBoost output
# }

# # Define parameter grids for hyperparameter tuning
# param_grids = {
#     "LogisticRegression": {"C": [0.01, 0.1, 1, 10, 100]},
#     "RandomForest": {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20, 30]},
#     "DecisionTree": {"max_depth": [None, 10, 20, 30]},
#     "CatBoost": {"learning_rate": [0.01, 0.05, 0.1], "depth": [4, 6, 10]}
# }


# %%
# best_estimators = {}
# for name in models:
#     print(f"Training and tuning {name}")
#     model = models[name]
#     param_grid = param_grids[name]
    
#     # Choose between GridSearchCV or RandomizedSearchCV based on your preference
#     clf = GridSearchCV(model, param_grid, scoring="f1_micro", cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42))
    
#     clf.fit(X_train_prepared, y_train_prepared)
#     best_estimators[name] = clf.best_estimator_
#     print(f"Best parameters for {name}: {clf.best_params_}")


# %%
# from sklearn.metrics import classification_report, accuracy_score

# for name, model in best_estimators.items():
#     print(f"Evaluating {name}")
    
#     # Predict on the test set
#     predictions = model.predict(X_test_prepared)
    
#     # Accuracy
#     accuracy = accuracy_score(y_test_prepared, predictions)
#     print(f"Accuracy: {accuracy:.4f}")
    
#     # Classification report
#     print("\nClassification Report:")
#     print(classification_report(y_test_prepared, predictions))
    
#     # Confusion Matrix
#     cm = confusion_matrix(y_test_prepared, predictions)
#     plt.figure(figsize=(10,7))
#     sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.title(f'{name} - Confusion Matrix')
#     plt.show()

# %%
# from sklearn.linear_model import LogisticRegression

# final_model = LogisticRegression(random_state=42)
# final_model.fit(X_train_prepared, y_train_prepared)

# %%
# train_pred = final_model.predict(X_train_prepared)
# print("train f1 score:",f1_score(train_pred, y_train_prepared,average="micro"))

# %%
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# scores = cross_val_score(final_model,
#                          X_train_prepared,
#                          y_train_prepared,
#                          scoring="f1_micro",
#                          cv=skf)

# print("cross validation Scores:\n", scores,"\n")
# print("cross validation Scores std:", scores.std())
# print("cross validation Scores mean:", scores.mean())

# %%
# test_pred = final_model.predict(X_test_prepared)
# print("test f1 score:",f1_score(test_pred, y_test_prepared,average="micro"))

# %%
# f = plt.figure(figsize = (5,4))
# ax = f.add_subplot()
# cm = confusion_matrix(y_test_prepared, test_pred)
# sns.heatmap(cm, annot=True, fmt='g',cmap="rocket_r", ax=ax)

# ax.set_xlabel('Predicted labels')
# ax.set_ylabel('True labels')
# ax.set_title('Test data Confusion Matrix')
# ax.xaxis.set_ticklabels(le.classes_)
# ax.yaxis.set_ticklabels(le.classes_)

# %%
# print(classification_report(y_test_prepared, test_pred, target_names=le.classes_))


# %%
# mapping = dict(zip(le.classes_, range(len(le.classes_))))
# print(mapping)

# %%
# test_pred[(test_pred == 0)] = 1
# y_test_prepared[(y_test_prepared == 0)] = 1
# print("test Binary f1 score:",f1_score(test_pred, y_test_prepared, average="micro"))

# %%



# %% [markdown]
# 

# %%


# %%
# test_df = pd.read_csv(os.path.join(dataset_path, 'C2T1_Test.csv'), na_values='?')
# test_df_prepared = full_pipeline.transform(test_df)
# test_df_predictions = final_model.predict(test_df_prepared)
# y_test_predicted = le.inverse_transform(test_df_predictions)
# test_df['readmitted'] = y_test_predicted
# test_df[['encounter_id','patient_nbr', 'readmitted']].to_csv('C2T1_Test_Labled.csv', index=False)

# %%



