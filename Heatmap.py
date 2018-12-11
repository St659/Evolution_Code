import matplotlib.pyplot as plt
import numpy as np


fig, heatmap = plt.subplots()

list_1 = [1,2,3,4]
list_2 = [4,3,2,1]
list_3 = [1,3,4,2]

combined_list = [list_1,list_2,list_3]
heatmap.imshow(np.asarray(combined_list))
plt.show()