import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
data = data.dropna()
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
features = data[['drinks_mapped', 'smokes_mapped', 'orientation_mapped', 'age', 'height', 'income', 'sex_mapped']]
pets = data['pets_mapped']
# x_train, x_test, y_train, y_test = train_test_split(features, pets)
# scaler2 = StandardScaler()
# scaler2.fit_transform(x_train, y_train)
# scaler2.transform(x_test, y_test)
# score = []
# for depth in range(2, 40):
#     classifier = RandomForestClassifier(max_depth=depth)
#     classifier.fit(x_train, y_train)
#     score.append(classifier.score(x_test, y_test))
# plt.plot(score, '-o', color='cadetblue')
# plt.show()
