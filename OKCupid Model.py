import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# what could be a fun thing to predict about people? Height? Income? To what degree they drink or smoke?
data = pd.read_csv('profiles.csv')
# print(data.columns)
# print(data.status.unique())
# Map "drinks" according to the following: not at all: 0, rarely: 1, socially: 2, often: 3, very often: 4
data['drinks_mapped'] = data['drinks'].map({'not at all': 0, 'rarely': 1, 'socially': 2, 'often': 3, 'very often': 4})
# Map 'smokes' according to the following: 'no': 0, 'when drinking': 1, 'sometimes': 2, 'trying to quit': 3, 'yes': 4
data['smokes_mapped'] = data['smokes'].map({'no': 0, 'when drinking': 1, 'sometimes': 2, 'trying to quit': 3, 'yes': 4})
data['drugs_mapped'] = data['drugs'].map({'never': 0, 'sometimes': 1, 'often': 2})
data['orientation_mapped'] = data['orientation'].map({'straight': 0, 'bisexual': 1, 'gay':2 })
data['sex_mapped'] = data['sex'].map({'m': 0, 'f': 1})
data['status_mapped'] = data['status'].map({'available': 0, 'single': 1, 'unknown': 2, 'seeing someone': 3, 'married': 4})
data['pets_mapped'] = data['pets'].map({'dislikes dogs and dislikes cats': 0,
                                        'dislikes cats': 0,
                                        'dislikes dogs': 0,
                                        'dislikes dogs and likes cats': 1,
                                        'dislikes dogs and has cats': 1,
                                        'likes dogs and dislikes cats': 2,
                                        'has dogs and dislikes cats': 2,
                                        'likes cats': 1,
                                        'likes dogs': 2,
                                        'likes dogs and likes cats': 3,
                                        'likes dogs and has cats': 3,
                                        'has dogs and likes cats': 3,
                                        'has dogs': 2,
                                        'has cats': 1,
                                        'has dogs and has cats': 3
                                        })

sns.pairplot(data, plot_kws=dict(s=80, alpha=0.3))
plt.show()
# Conduct KMeans clustering on pets, drinks, smokes, orientation, age, height, income, and sex.
data = data.fillna(method='ffill')
features = data[['pets_mapped', 'drinks_mapped', 'smokes_mapped', 'orientation_mapped', 'age', 'height', 'income', 'sex_mapped']]

scaler = StandardScaler()
scaled_feat = scaler.fit_transform(features)

# sns.pairplot(data_scaled, plot_kws=dict(s=80, alpha=0.3))
# plt.show()
inertias = []
for n in range(1, 20):
    cluster = KMeans(n_clusters=n)
    cluster.fit(scaled_feat)
    inertias.append(cluster.inertia_)

plt.plot(inertias, '-o', color='cadetblue')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
# Looks like there are maybe 5 different groups of people, or 5 different clusters, in this data. What could those
# clusters correspond to?

# Train a random forest classifier to predict whether a person likes cats or likes dogs or both or neither
# features = data[['drinks_mapped', 'smokes_mapped', 'orientation_mapped', 'age', 'height', 'income', 'sex_mapped']]
# pets = data['pets_mapped']
# x_train, x_test, y_train, y_test = train_test_split(features, pets)
# scaler2 = StandardScaler()
# scaler2.fit_transform(x_train, y_train)
# scaler2.transform(x_test)
# # gini_score = []
# # for est in range(10, 100, 5):
# #     classifier = RandomForestClassifier(max_depth=5, n_estimators=est)
# #     classifier.fit(x_train, y_train)
# #     gini_score.append(classifier.score(x_test, y_test))
# #
# # plt.plot(range(10, 100, 5), gini_score, '-o')
# # plt.xlabel('Number of Estimators')
# # plt.ylabel('Mean Accuracy')
#
#
# entropy_score = []
# gini_score = []
# log_score = []
# ent_f1 = []
# gini_f1 = []
# for n in range(2, 40, 2):
#     ent_classifier = RandomForestClassifier(criterion='entropy', max_depth=n)
#     ent_classifier.fit(x_train, y_train)
#     prediction_ent = ent_classifier.predict(x_test)
#     entropy_score.append(ent_classifier.score(x_test, y_test))
#     ent_f1.append(f1_score(y_test, prediction_ent, average='weighted'))
#
#     gini_classifier = RandomForestClassifier(max_depth=n)
#     gini_classifier.fit(x_train, y_train)
#     prediction_gini = gini_classifier.predict(x_test)
#     gini_score.append(gini_classifier.score(x_test, y_test))
#     gini_f1.append(f1_score(y_test, prediction_gini, average='weighted'))
#     # log_classifier = RandomForestClassifier(criterion="log_loss", max_depth=n)
#     # log_classifier.fit(x_train, y_train)
#     # log_score.append(log_classifier.score(x_test, y_test))
#
# plt.plot(range(2, 40, 2), gini_score, '-o', label='Gini')
# plt.plot(range(2, 40, 2), entropy_score, '-o', label='Entropy')
# plt.xlabel('Max Depth')
# plt.ylabel('Mean Accuracy')
# plt.legend()
# plt.title('Average Accuracy using Random Forest Classifier')
# plt.show()
#
# plt.plot(range(2, 40, 2), gini_f1, '-o', label='Gini')
# plt.plot(range(2, 40, 2), ent_f1, '-o', label='Entropy')
# plt.xlabel('Max Depth')
# plt.ylabel('F1 Score')
# plt.title('F1 Score using Random Forest Classifier')
# plt.legend()
# plt.show()
# print('Random Forest done!')
# # Looks like there's very little difference in accuracy, recall, or precision for these two Criterion. However, there
# # seems to be a trade-off between accuracy and F1 which is a function of max depth.
#
# # Try the same thing with SVM!
# SVM_score = []
# SVM_f1 = []
# C_span = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
# for c in C_span:
#     SVM = SVC(C=c)
#     SVM.fit(x_train, y_train)
#     prediction = SVM.predict(x_test)
#     SVM_score.append(SVM.score(x_test,y_test))
#     SVM_f1.append(f1_score(y_test, prediction, average='weighted'))
#
# plt.plot(range(-4, 3), SVM_score, label='Mean Accuracy')
# plt.plot(range(-4, 3), SVM_f1, label='F1 Score')
# plt.xlabel('Regularization Parameter Log')
# plt.legend()
# plt.title('Support Vector Machine')
# plt.show()
# print('SVM done!')
# # Try the same thing with K-Nearest Neighbors!
# KNN_score = []
# KNN_F1 = []
# for n in range(1, 51, 5):
#     KNN = KNeighborsClassifier(n_neighbors=n)
#     KNN.fit(x_train, y_train)
#     KNN_prediction = KNN.predict(x_test)
#     KNN_score.append(KNN.score(x_test, y_test))
#     KNN_F1.append(f1_score(y_test, KNN_prediction, average='weighted'))
#
# plt.plot(range(1, 51, 5), KNN_score, label='Mean Accuracy')
# plt.plot(range(1, 51, 5), KNN_F1, label='F1 Score')
# plt.xlabel('K Nearest Neighbors')
# plt.legend()
# plt.title('K Nearest Neighbors')
# plt.show()
# print('KNN done!')

# There doesn't appear to be an obviously BEST model. Perhaps there is very little dependence of pet preference to  the
# other features we investigated. Let's try something easier, such as gender, which only has two options according to
# the data set provided.

features = data[['drinks_mapped', 'smokes_mapped', 'orientation_mapped', 'age', 'height', 'income', 'pets_mapped']]
sex = data['sex_mapped']
x_train, x_test, y_train, y_test = train_test_split(features, sex)
scaler2 = StandardScaler()
scaler2.fit_transform(x_train, y_train)
scaler2.transform(x_test)


entropy_score = []
gini_score = []
log_score = []
ent_f1 = []
gini_f1 = []
for n in range(2, 40, 2):
    ent_classifier = RandomForestClassifier(criterion='entropy', max_depth=n)
    ent_classifier.fit(x_train, y_train)
    prediction_ent = ent_classifier.predict(x_test)
    entropy_score.append(ent_classifier.score(x_test, y_test))
    ent_f1.append(f1_score(y_test, prediction_ent, average='weighted'))

    gini_classifier = RandomForestClassifier(max_depth=n)
    gini_classifier.fit(x_train, y_train)
    prediction_gini = gini_classifier.predict(x_test)
    gini_score.append(gini_classifier.score(x_test, y_test))
    gini_f1.append(f1_score(y_test, prediction_gini, average='weighted'))

plt.plot(range(2, 40, 2), gini_score, '-o', label='Gini')
plt.plot(range(2, 40, 2), entropy_score, '-o', label='Entropy')
plt.xlabel('Max Depth')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.title('Average Accuracy using Random Forest Classifier')
plt.show()

plt.plot(range(2, 40, 2), gini_f1, '-o', label='Gini')
plt.plot(range(2, 40, 2), ent_f1, '-o', label='Entropy')
plt.xlabel('Max Depth')
plt.ylabel('F1 Score')
plt.title('F1 Score using Random Forest Classifier')
plt.legend()
plt.show()
print('Random Forest done!')

# Try the same thing with SVM!
SVM_score = []
SVM_f1 = []
C_span = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for c in C_span:
    SVM = SVC(C=c)
    SVM.fit(x_train, y_train)
    prediction = SVM.predict(x_test)
    SVM_score.append(SVM.score(x_test,y_test))
    SVM_f1.append(f1_score(y_test, prediction, average='weighted'))

plt.plot(range(-4, 3), SVM_score, label='Mean Accuracy')
plt.plot(range(-4, 3), SVM_f1, label='F1 Score')
plt.xlabel('Regularization Parameter Log')
plt.legend()
plt.title('Support Vector Machine')
plt.show()
print('SVM done!')
# Try the same thing with K-Nearest Neighbors!
KNN_score = []
KNN_F1 = []
for n in range(1, 51, 5):
    KNN = KNeighborsClassifier(n_neighbors=n)
    KNN.fit(x_train, y_train)
    KNN_prediction = KNN.predict(x_test)
    KNN_score.append(KNN.score(x_test, y_test))
    KNN_F1.append(f1_score(y_test, KNN_prediction, average='weighted'))

plt.plot(range(1, 51, 5), KNN_score, label='Mean Accuracy')
plt.plot(range(1, 51, 5), KNN_F1, label='F1 Score')
plt.xlabel('K Nearest Neighbors')
plt.legend()
plt.title('K Nearest Neighbors')
plt.show()
print('KNN done!')

