import matplotlib.pyplot as plt
import numpy as np
import sys
import re

args = sys.argv[1:]

if len(args) == 0:
    print("No file specified.")
    sys.exit(1)

file_path = args[0]

with open("logs/"+file_path+"/logs.csv", 'r') as file:
    data = file.read()

pattern = r"Transition Matrix tensor\(\[([\s\S]+?)\]\)"
match = re.search(pattern, data)
label_map = []
if "min20" in file_path:
    # label_map = [
    #     'tailed frog', 
    #     # 'frog', 
    #     'scorpion', 
    #     'snail', 
    #     'american lobster', 
    #     # 'lobster', 
    #     'tabby', 
    #     'persian cat', 
    #     # 'cat', 
    #     'gazelle', 
    #     'chimpanzee', 
    #     'bannister', 
    #     'barrel', 
    #     'christmas stocking', 
    #     # 'stocking', 
    #     'gasmask', 
    #     'hourglass', 
    #     'ipod', 
    #     'scoreboard', 
    #     'snorkel', 
    #     'suspension bridge', 
    #     # 'bridge', 
    #     'torch', 
    #     'tractor', 
    #     'triumphal arch'
    # ]
    label_map = [str(i+1) for i in range(20)]
elif "cifar20" in file_path:
    # label_map = [
    #     "aquatic mammals", 
    #     "fish", 
    #     "flowers", 
    #     "food containers", 
    #     "fruit and vegetables and mushrooms", 
    #     "electrical devices", 
    #     "furniture", 
    #     "insects", 
    #     "carnivores and bears", 
    #     "man-made buildings", 
    #     "natural scenes", 
    #     "omnivores and herbivores", 
    #     "medium-sized mammals", 
    #     "invertebrates", 
    #     "people", 
    #     "reptiles", 
    #     "small mammals", 
    #     "trees", 
    #     "transportation vehicles", 
    #     "non-transport vehicles"
    # ]
    label_map = [str(i+1) for i in range(20)]
elif "min10" in file_path:
    label_map = [
        "sulphur butterfly", 
        # "butterfly", 
        "backpack", 
        "cardigan", 
        "kimono", 
        "magnetic compass", 
        # "compass", 
        "oboe", 
        "sandal", 
        "torch", 
        "pizza", 
        "alp", 
    ]
elif "cifar10" in file_path:
    label_map = [
        "airplane", 
        "automobile", 
        "bird", 
        "cat", 
        "deer", 
        "dog", 
        "frog", 
        "horse", 
        "ship", 
        "truck"
    ]
matrix_str = match.group(1)

rows = matrix_str.strip().split(']')
matrix = []
for row in rows:
    numbers = row.strip().replace('[', '').replace('\n', '').split(',')
    matrix.append([float(num) for num in numbers if num])
matrix.pop()
np_matrix = np.array(matrix)
plt.figure(figsize=(10, 8))
plt.imshow(np_matrix, cmap='gist_heat_r')
max_val = np_matrix.max()
plt.xticks(range(len(label_map)), label_map, rotation=45, ha='right', fontsize=6)
plt.yticks(range(len(label_map)), label_map, fontsize=6)
for i in range(len(label_map)):
    for j in range(len(label_map)):
        plt.text(j, i, round(np_matrix[i, j],4), ha='center', va='center', color='black' if np_matrix[i, j] < max_val/2  else 'white', fontsize=5)
plt.colorbar()
plt.savefig(file_path+"/heatmap.png", dpi=300, bbox_inches='tight')
