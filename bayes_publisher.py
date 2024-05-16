import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def query(console, genre, developer, data, publishers, publisher_dict):
    games = data.loc[(data['console'] == console) & (data['genre'] == genre) & (data['developer'] == developer)]
    publisher_counts = np.array([0 for _ in range(len(publishers))])
    value_counts = games['publisher'].value_counts()
    for publisher in publishers:
        publisher_counts[publisher_dict[publisher]] = 0 if publisher not in value_counts else value_counts[publisher]
    return publisher_counts

file = "Data/vgchartz-2024.csv"
data = pd.read_csv(file)

# create publisher vector
# first, get the publishers
publishers = data['publisher'].drop_duplicates().to_numpy()
# next, create dictionary of name to index
publisher_dict = dict(zip(publishers, list(range(len(publishers)))))

total_publisher_counts = np.array([len(data.loc[data['publisher'] == publisher]) for publisher in publishers])

# create frequency data
# we can do this by matching all possible combinations of console, genre, and developer
# then computing probabilities for each publisher
consoles = data['console'].drop_duplicates().to_numpy()
genres = data['genre'].drop_duplicates().to_numpy()
developers = data['developer'].drop_duplicates().to_numpy()

avg_diff = 0
avg_diff_norm = 0

for (idx, console) in enumerate(consoles):
    console_games = data.loc[data['console'] == console]
    for (g_idx, genre) in enumerate(genres):
        print(f"{100 * (idx * len(genres) + g_idx) / ( len(consoles) * len(genres)) :.3f}%")

        genre_games = console_games.loc[console_games['genre'] == genre]
        for developer in developers:
            #print(console, genre, developer)

            games = genre_games.loc[genre_games['developer'] == developer]
            publisher_counts = np.array([0 for _ in range(len(publishers))])
            value_counts = games['publisher'].value_counts().to_numpy()
            if len(value_counts) == 0: continue
            elif len(value_counts) == 1:  
                avg_diff += value_counts[0]
                avg_diff_norm += 1
            else: 
                value_counts_part = np.partition(value_counts, len(value_counts) - 2)
                avg_diff += (value_counts_part[-1] - value_counts_part[-2]) 
                avg_diff_norm += (value_counts_part[-1] - value_counts_part[-2]) / np.sum(value_counts)
    
        
avg_diff /= (len(consoles) * len(genres) * len(developers) )
avg_diff_norm /= (len(consoles) * len(genres) * len(developers) )
print(f"Average diff: {avg_diff}, normalized: {avg_diff_norm}")

