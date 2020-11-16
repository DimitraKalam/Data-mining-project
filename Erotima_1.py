#Καλαματιανού Δήμητρα
#up1054406@upnet.gr

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Import the dataset
data = pd.read_csv(r'C:\Users\Δημητρα\Documents\ΣΧΟΛΗ\8o ΕΞΑΜΗΝΟ\εξορυξη\ΥΛΟΠΟΙΗΤΙΚΟ_PROJECT_2020\winequality-red.csv')
data.head()
# Drop quality
X=data.drop("quality",axis=1)
y=data["quality"]

# EROTIMA 1-A

# Split into training(75%) and test(25%) sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Default settings sypport vector machine (SVM) classifier
# Create, fit, and test default SVM
# The Radial basis function kernel(rbf) is a popular kernel function commonly used in support vector machine classification
rbf_SVM = SVC()
rbf_SVM.fit(X_train, y_train)
svm_predictions = rbf_SVM.predict(X_test)

# gamma is a parameter, which ranges from 0 to 1.
# A higher value of gamma will perfectly fit the training dataset, which causes over-fitting.
# Gamma=0.1 is considered to be a good default value
print("Default SVC parameters are: ", format(rbf_SVM.get_params(deep=True)))

# print precision,recall,f1-score,support
print(metrics.classification_report(y_test, svm_predictions,zero_division=1))
print("Overall Accuracy:", round(metrics.accuracy_score(y_test, svm_predictions), 3))

# Randomized search to try to improve on this accuracy.
# Define distributions, sample hyperparameters values from and create a dictionary of possible values

# Define distributions to sample hyperparameters from
np.random.seed(123)
# uniform(lower boundary,upper boundary
g_range = np.random.uniform(0.0, 0.3, 5).astype(float)
# random.normal(mean(“centre”) of the distribution,
# standard deviation (spread or “width”) of the distribution,
# Output shape
C_range = np.random.normal(1, 0.1, 5).astype(float)

# Check that gamma>0 and C>0
C_range[C_range < 0] = 0.0001

# Dicrionary with possible values of gamma and C
hyperparameters = {'gamma': list(g_range),'C': list(C_range)}

# Pass this dictionary to param_distributions argument of RandomizedSearchCV:
# Run randomized search
randomCV = RandomizedSearchCV(SVC(kernel='rbf'), param_distributions=hyperparameters, n_iter=20)
randomCV.fit(X_train, y_train)

# Identify optimal hyperparameter values
best_gamma = randomCV.best_params_['gamma']
best_C = randomCV.best_params_['C']

print("\nThe best performing gamma value is: {:5.2f}".format(best_gamma))
print("The best performing C value is: {:5.2f}".format(best_C))

# Train SVM and output predictions with best C and best gamma
rbf_SVM = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
rbf_SVM.fit(X_train, y_train)
svm_predictions = rbf_SVM.predict(X_test)

print(metrics.classification_report(y_test, svm_predictions,zero_division=1))
print("Overall Accuracy:", round(metrics.accuracy_score(y_test, svm_predictions), 4))



# EROTIMA 1-B

new_x_train=pd.DataFrame(X_train)
# delete 33% of values in column pH
dataupdate=new_x_train.sample(frac = 0.33)
dataupdate.pH=0
new_x_train.update(dataupdate)
update_list = dataupdate.index.tolist()

# FILLING MISSING VALUES
# gia kathe tropo kano copy to new_x_train

#1os tropos --> afairesi stilis
x_train_protos_tropos=new_x_train.copy()
x_train_protos_tropos=x_train_protos_tropos.drop('pH',axis=1)
x_test_protos_tropos=X_test.drop('pH',axis=1)

# SVM 1os TROPOS
rbfSVM = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
rbfSVM.fit(x_train_protos_tropos, y_train)
svm_predictions = rbfSVM.predict(x_test_protos_tropos)

print("1os tropos: \n",metrics.classification_report(y_test, svm_predictions, zero_division=1))
print("Overall Accuracy 1ou tropou:", round(metrics.accuracy_score(y_test, svm_predictions), 4))


# 2os tropos --> mean
x_train_deuteros_tropos=new_x_train.copy()
x_train_deuteros_tropos["pH"]=x_train_deuteros_tropos.mask(x_train_deuteros_tropos.pH ==0, x_train_deuteros_tropos["pH"].mean(skipna=True))
# x_train_deuteros_tropos=new_x_train.copy()
# x_train_deuteros_tropos["pH"]=x_train_deuteros_tropos["pH"].mask(x_train_deuteros_tropos.pH ==0, x_train_deuteros_tropos["pH"].mean(skipna=True))

# SVM 2os TROPOS
rbfSVM = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
rbfSVM.fit(x_train_deuteros_tropos, y_train)
svm_predictions = rbfSVM.predict(X_test)

print("\n2os tropos: \n",metrics.classification_report(y_test, svm_predictions, zero_division=1))
print("Overall Accuracy 2ou tropou:", round(metrics.accuracy_score(y_test, svm_predictions), 4))


# 3os tropos --> Logistic Regression
x_train_tritos_tropos=new_x_train.copy()
x_test_tritos_tropos=X_test.copy()
# perform feature scaling
sc_x = StandardScaler()
x_train_tritos_tropos = sc_x.fit_transform(x_train_tritos_tropos)
x_test_tritos_tropos = sc_x.transform(x_test_tritos_tropos)
# features values are sacled and now there in the -1 to 1.
# each feature will contribute equally in decision making
# training Logistic Regression model
classifier = LogisticRegression(random_state = 0,max_iter=2000)
# max_iter=2000 when an optimization algorithm does not converge,
# it is usually because the problem is not well-conditioned, perhaps due to a poor scaling of the decision variables.
classifier.fit(x_train_tritos_tropos, y_train)

# SVM 3oS TROPOS
rbfSVM = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
rbfSVM.fit(x_train_tritos_tropos, y_train)
svm_predictions = rbfSVM.predict(x_test_tritos_tropos)

print("\n3os tropos: \n",metrics.classification_report(y_test, svm_predictions, zero_division=1))
print("Overall Accuracy 3ou tropou:", round(metrics.accuracy_score(y_test, svm_predictions), 4))


# 4os tropos --> K means
x_train_tetartos_tropos=new_x_train.copy()
x_test_tetartos_tropos=X_test.copy()
# Cluster the data
kmeans = KMeans(n_clusters=7).fit(x_train_tetartos_tropos)
clusters = kmeans.predict(x_train_tetartos_tropos)
# add 'clusters' column
x_train_tetartos_tropos['clusters'] = clusters

for i in x_train_tetartos_tropos.iterrows():
    for x in range(7): #oso o arithmos ton  clusters(0-6)
        x_train_tetartos_tropos.loc[(x_train_tetartos_tropos['pH'] == 0) & (x_train_tetartos_tropos['clusters'] == x), 'pH'] = x_train_tetartos_tropos["pH"].mean(skipna=True)

# drop 'clusters' column
x_train_tetartos_tropos = x_train_tetartos_tropos.drop("clusters",axis=1)

# SVM 4oS TROPOS
rbfSVM = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
rbfSVM.fit(x_train_tetartos_tropos, y_train)
svm_predictions = rbfSVM.predict(x_test_tetartos_tropos)

print("\n4os tropos: \n",metrics.classification_report(y_test, svm_predictions, zero_division=1))
print("Overall Accuracy 4ou tropou:", round(metrics.accuracy_score(y_test, svm_predictions), 4))


