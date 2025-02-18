import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load("training_data.npy")

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []
stops = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]
    
    if choice == [1,0,0,0]:
        forwards.append([img, choice])
    elif choice == [0,1,0,0]:
        lefts.append([img, choice])
    elif choice == [0,0,1,0]:
        rights.append([img, choice])
    elif choice == [0,0,0,1]:
        stops.append([img, choice])
    else:
        print("No matches!")
        
forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]
stops = stops[:len(forwards)]

final_data = forwards + lefts + rights + stops
shuffle(final_data)

np.save('training_data_balanced.npy', final_data) 