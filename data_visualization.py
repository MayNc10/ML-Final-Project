import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np


file = "Data/vgchartz-2024.csv"
data = pd.read_csv(file)
limit = 5.0
data_big = data.loc[data['total_sales'] > limit]
num_publishers = data_big['publisher'].drop_duplicates().count()
num_developers = data_big['developer'].drop_duplicates().count()
print(f"Out of {data_big['total_sales'].count()} filtered games, there are {num_publishers} unique publishers and {num_developers} unique developers")
print(f"Publishers: {data_big['publisher'].drop_duplicates()}")

print(f"There are {data['genre'].drop_duplicates().count()} genres")
genre_names = data['genre'].drop_duplicates().to_numpy()
genre_dictionary = dict(zip(genre_names, list(range(len(genre_names)))))
color = cm.rainbow(np.linspace(0, 1, len(genre_names)))
data['genre_idxs'] = data['genre'].map(genre_dictionary)
plt.scatter(data['total_sales'], data['critic_score'], label=data['genre'], color=color[data['genre_idxs']])
plt.show()