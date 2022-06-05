import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# what could be a fun thing to predict about people? Height? Income? To what degree they drink or smoke?
data = pd.read_csv('profiles.csv')
print(data.income.describe())
# Map "drinks" according to the following: not at all: 0, rarely: 1, socially: 2, often: 3, very often: 4
data['drinks_mapped'] = data['drinks'].map({'not at all': 0, 'rarely': 1, 'socially': 2, 'often': 3, 'very often': 4})
# Map 'smokes' according to the following: 'no': 0, 'when drinking': 1, 'sometimes': 2, 'trying to quit': 3, 'yes': 4
data['smokes_mapped'] = data['smokes'].map({'no': 0, 'when drinking': 1, 'sometimes': 2, 'trying to quit': 3, 'yes': 4})
sns.pairplot(data, plot_kws=dict(s=80, alpha=0.3))
plt.show()