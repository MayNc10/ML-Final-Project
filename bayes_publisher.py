import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def query(console, genre, developer, data, publishers):
    prob_con = len(data.loc[data['console'] == console]) / len(data)
    prob_gen = len(data.loc[data['genre'] == genre]) / len(data)
    prob_dev = len(data.loc[data['developer'] == developer]) / len(data)
    total_denom = prob_con * prob_gen * prob_dev
    prob_list = []
    for publisher in publishers:
        data_pub = data.loc[data['publisher'] == publisher]
        prob_pub = len(data_pub) / len(data)
        prob_pub_con = len(data_pub.loc[data_pub['console'] == console]) / len(data_pub)
        prob_pub_gen = len(data_pub.loc[data_pub['genre'] == genre]) / len(data_pub)
        prob_pub_dev = len(data_pub.loc[data_pub['developer'] == developer]) / len(data_pub)
    
        # watch for underflow
        prob = (prob_pub * prob_pub_con * prob_pub_gen * prob_pub_dev) / total_denom
        prob_list.append(prob)
    return prob_list


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

data['query_tup'] = list(zip(data['console'], data['genre'], data['developer']))

loss_sum = 0
correct_count = 0

tups = data['query_tup'].drop_duplicates().to_numpy()
total_count = 0

for (e_idx, tup) in enumerate(tups):
    prob_list = query(tup[0], tup[1], tup[2], data, publishers)
    for publisher in data.loc[(data['console'] == tup[0]) & (data['genre'] == tup[1]) & (data['developer'] == tup[2])]['publisher'].drop_duplicates():
        idx = publisher_dict[publisher]
        loss_sum += (1 - prob_list[idx])
        correct_count += 1 if np.argmax(prob_list) == idx else 0
        total_count += 1

    print(f"{e_idx}, {100 * e_idx / total_count}% done, {100 * correct_count / total_count}% correct")
    
    

print(f"Total loss: {loss_sum}, correct: {100 * correct_count / total_count}%")
    
    


