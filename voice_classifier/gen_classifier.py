import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sb.set_style("whitegrid", {'axes.grid': False})
pd.set_option('display.max_columns', None)
show_plot = False

df = pd.read_csv('voice-data.csv')

print(df.dtypes)

# gender ['Male' 'Female' 'not_specified']

# split population male/female using patterns
female = df.loc[df.gender == 'Female']
male = df.loc[df.gender == 'Male']
male_ = male.sample(int(len(female) * 1.2))

female.loc[df['gender'] == 'Female', 'gender'] = 0
male_.loc[df['gender'] == 'Male', 'gender'] = 1

df = pd.concat([male_, female])
# shuffle df
df = df.sample(frac=1).reset_index(drop=True)

features = ['mean', 'skew', 'kurtosis', 'median', 'mode', 'std', 'low', 'peak', 'q25', 'q75', 'iqr']

# plot gender
sb.countplot(x='gender', data=df)
plt.savefig('log_gen/data_distribution.png')
if show_plot:
    plt.show()

# remove Nan
df = df.dropna()

print(df.tail())

# split df into train and test
train_df, test_df = train_test_split(df, random_state=0, test_size=.2)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler.fit(train_df.loc[:, features])

X_train = scaler.transform(train_df.loc[:, features])
X_test = scaler.transform(test_df.loc[:, features])

y_train = list(train_df['gender'].values)
y_test = list(test_df['gender'].values)

dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=5, random_state=42)
gb = GradientBoostingClassifier(random_state=42)
sv = SVC(probability=True, random_state=42)
mlp = MLPClassifier(random_state=42)

# train models
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
sv.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# accuracy
dt_accuracy = dt.score(X_test, y_test)
rf_accuracy = rf.score(X_test, y_test)
gb_accuracy = gb.score(X_test, y_test)
sv_accuracy = sv.score(X_test, y_test)
mlp_accuracy = mlp.score(X_test, y_test)

# create log_gen.txt
with open('log_gen/log_gen.txt', 'w') as f:
    f.write('Decision Tree: {:.3f}\n'.format(dt_accuracy))
    f.write('Random Forest: {:.3f}\n'.format(rf_accuracy))
    f.write('Gradient Boosting: {:.3f}\n'.format(gb_accuracy))
    f.write('Support Vector Machine: {:.3f}\n'.format(sv_accuracy))
    f.write('Multilayer Perceptron: {:.3f}\n'.format(mlp_accuracy))

# print accuracy
print('Decision Tree: {:.3f}'.format(dt_accuracy))
print('Random Forest: {:.3f}'.format(rf_accuracy))
print('Gradient Boosting: {:.3f}'.format(gb_accuracy))
print('Support Vector Machine: {:.3f}'.format(sv_accuracy))
print('Multilayer Perceptron: {:.3f}'.format(mlp_accuracy))

# pick best model
best_model = max(dt_accuracy, rf_accuracy, gb_accuracy, sv_accuracy, mlp_accuracy)

if best_model == dt_accuracy:
    model = dt
    model_name = 'Decision Tree'
    mode_accuracy = dt_accuracy
elif best_model == rf_accuracy:
    model = rf
    model_name = 'Random Forest'
    model_accuracy = rf_accuracy
elif best_model == gb_accuracy:
    model = gb
    model_name = 'Gradient Boosting'
    model_accuracy = gb_accuracy
elif best_model == sv_accuracy:
    model = sv
    model_name = 'Support Vector Machine'
    model_accuracy = sv_accuracy
elif best_model == mlp_accuracy:
    model = mlp
    model_name = 'Multilayer Perceptron'
    model_accuracy = mlp_accuracy

# classification report
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# log_gen
with open('log_gen/log_gen.txt', 'a') as f:
    f.write("\n\n-----------------------------------------------\n")
    f.write('\nClassification Report\n')
    f.write(classification_report(y_test, y_pred))
    f.write("\n\n-----------------------------------------------\n")
    f.write('\nBest model: {}\n'.format(model_name))
    f.write('Accuracy: {:.3f}\n'.format(model_accuracy))

# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 10))
sb.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {:.3f}'.format(model_accuracy)
plt.title(all_sample_title, size=15)
plt.savefig('log_gen/confusion_matrix.png')
if show_plot:
    plt.show()

# # plot feature importance
# importances = model.feature_importances_
# indices = np.argsort(importances)[::-1]
#
# plt.figure(figsize=(10, 5))
# plt.title("Feature importances")
# plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
# plt.xticks(range(X_train.shape[1]), indices)
# plt.xlim([-1, X_train.shape[1]])
# plt.savefig('log_gen/feature_importance.png')
# plt.show()

# lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=features, class_names=['0', '1'],
#                                                         discretize_continuous=False)

#
# def lime_explain_instance(idx):
#     exp = lime_explainer.explain_instance(X_test[idx], model.predict_proba, num_features=5)
#     # save as jpeg
#     exp.save_to_file('log_gen/lime_{}.html'.format(idx))
#
#     exp.save_to_file('log_gen/lime.html')
#
#     return exp
#
#
# # plot lime explanation
# exp = lime_explain_instance(0)
# exp.show_in_notebook(show_table=True, show_all=False)
#
# # plot lime explanation
# exp = lime_explain_instance(1)
# exp.show_in_notebook(show_table=True, show_all=False)
#


# plot ROC curve
from sklearn.metrics import roc_curve, auc

y_score = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.3f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('Receiver operating characteristic', fontsize=15)
plt.legend(loc="lower right")
plt.savefig('log_gen/roc_curve.png')
if show_plot:
    plt.show()

# plot learning curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1,
                                                        train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

plt.figure(figsize=(10, 10))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Validation score')
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.title('Learning curve', fontsize=15)
plt.legend(loc='best')
plt.savefig('log_gen/learning_curve.png')
if show_plot:
    plt.show()

# plot validation curve
from sklearn.model_selection import validation_curve

param_range = np.logspace(-6, -1, 5)
# if model if MLPClassifier then param_name='alpha' else param_name='ccp_alpha'
if model_name == 'Multilayer Perceptron':
    param_name = 'alpha'
else:
    param_name = 'ccp_alpha'

train_scores, test_scores = validation_curve(model, X_train, y_train, param_name=param_name, param_range=param_range,
                                             cv=5, scoring="accuracy", n_jobs=-1, verbose=0)

plt.figure(figsize=(10, 10))
plt.plot(param_range, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
plt.plot(param_range, np.mean(test_scores, axis=1), 'o-', color='g', label='Validation score')
plt.xlabel('alpha', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.title('Validation curve', fontsize=15)
plt.legend(loc='best')
plt.savefig('log_gen/validation_curve.png')
if show_plot:
    plt.show()

# plot precision-recall curve
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.figure(figsize=(10, 10))
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-recall curve')
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
plt.title('Precision-recall curve', fontsize=15)
plt.legend(loc="lower right")
plt.savefig('log_gen/precision_recall_curve.png')
if show_plot:
    plt.show()

if not model_name == 'Multilayer Perceptron':
    # plot feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X_train.shape[1]), [features[i] for i in indices])
    plt.xlim([-1, X_train.shape[1]])
    plt.savefig('log_gen/feature_importance.png')
    if show_plot:
        plt.show()

# save model
import pickle

with open('log_gen/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# save scaler
with open('log_gen/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('../models/voice/gen_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# save scaler
with open('../models/voice/gen_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
