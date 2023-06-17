import pandas as pd
import math
import numpy as np

data_global = pd.read_csv('../dataset/covid19_global.csv')
data_global = data_global.fillna(0)
# print(data_global.info)

dist_arr = np.zeros((289, 289))
position_arr = np.zeros((289, 289, 2))
# print(data_global.head())

for i, src_row in data_global.iterrows():
    for j, dst_row in data_global.iterrows():
        dist = (src_row["Lat"] - dst_row["Lat"])**2 + (src_row["Long"] - dst_row["Long"])**2
    
        dist_arr[i][j] = dist     


sigma =  np.sum(np.sqrt(dist_arr))/(256*256)
dist_arr = np.exp(-dist_arr/sigma**2)

# print(dist_arr)

adjacency_matrix = np.zeros((289, 289))

for i in range(dist_arr.shape[0]):
    sorted_idx = np.argsort(dist_arr[i, :])
    nearest_neighbours = sorted_idx[-11:-1]
    
    for idx in nearest_neighbours:
        adjacency_matrix[i][idx] = dist_arr[i][idx]
    # print(i, nearest_neighbours)

# print(adjacency_matrix)
# print(np.sum(adjacency_matrix))

# np.savetxt("../dataset/adjacency.csv", adjacency_matrix, delimiter=",")


positions = data_global[["Lat", "Long"]]
positions.to_csv("../dataset/positions.csv")
