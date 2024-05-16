import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np


file = "Data/vgchartz-2024.csv"
data = pd.read_csv(file)
genre_names = data['genre'].drop_duplicates().to_numpy()
genre_dictionary = dict(zip(genre_names, list(range(len(genre_names)))))
color = cm.rainbow(np.linspace(0, 1, len(genre_names)))
data['genre_idxs'] = data['genre'].map(genre_dictionary)
limit = 1.0
data_big = data.loc[data['total_sales'] > limit]
num_publishers = data_big['publisher'].drop_duplicates().count()
num_developers = data_big['developer'].drop_duplicates().count()
print(f"Out of {data_big['total_sales'].count()} filtered games, there are {num_publishers} unique publishers and {num_developers} unique developers")
print(f"Publishers: {data_big['publisher'].drop_duplicates()}")
print(f"There are {data['genre'].drop_duplicates().count()} genres")

plt.scatter(data_big['total_sales'], data_big['critic_score'], label=data_big['genre'], color=color[data_big['genre_idxs']])
plt.xscale("log")
plt.show()

japanese_data = data.loc[data['na_sales'] < data['jp_sales']]
american_data = data.loc[data['na_sales'] > data['jp_sales']]
print(f"Japanese game publishers: {japanese_data['publisher'].drop_duplicates()}")
print(japanese_data['genre'].drop_duplicates())
print(american_data['genre'].drop_duplicates())

genre_sales_jp = [ (genre, data.loc[data['genre'] == genre]['jp_sales'].sum()) for genre in genre_names]
print(f"Sales by Genre in Japan: {genre_sales_jp}")

genre_sales_na = [ (genre, data.loc[data['genre'] == genre]['na_sales'].sum()) for genre in genre_names]
print(f"Sales by Genre in NA: {genre_sales_na}")

