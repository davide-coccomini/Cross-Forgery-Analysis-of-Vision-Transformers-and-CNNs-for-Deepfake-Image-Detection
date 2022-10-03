import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import pandas as pd
from utils import approaches, groups
import numpy as np
CROPS_DIR = "data/forgerynet/Training/crops"
STATS_DIR = "stats"
LIST_FILE_PATH = "data/forgerynet/training_image_list.txt"
'''
all_classes = dict.fromkeys([i for i in range(1, 20)], 0)
two_classes = dict.fromkeys(["Real", "Fake"], 0)
for i in range(1, 20):
    print("Checking", i)
    tmp = os.path.join(CROPS_DIR, str(i))
    folder_names = os.listdir(tmp)
    for folder_name in folder_names:
        folder_path = os.path.join(tmp, folder_name)
        
        if os.path.isdir(folder_path): # We need to search more
            internal_folders = os.listdir(folder_path)
            
            for internal_folder in internal_folders: # We found files
                internal_path = os.path.join(folder_path, internal_folder)
                if os.path.isdir(internal_path): 
                    file_names = os.listdir(internal_path)
                    all_classes[i] += len(file_names)
                    if i > 15:
                        two_classes["Real"] += len(file_names)
                    else:
                        two_classes["Fake"] += len(file_names)

                else:
                    all_classes[i] += len(internal_folders)
                    if i > 15:
                        two_classes["Real"] += len(internal_folders)
                    else:
                        two_classes["Fake"] += len(internal_folders)
                    break

        else: #  We found the files
            all_classes[i] += len(folder_names)
            if i > 15:
                two_classes["Real"] += len(folder_names)
            else:
                two_classes["Fake"] += len(folder_names)
            break


print(all_classes)
print(two_classes)

names = list(all_classes.keys())
values = list(all_classes.values())
plt.bar(range(len(all_classes)),values, tick_label=names)

plt.savefig(os.path.join(STATS_DIR, 'all_classes.png'))

plt.clf()

names = list(two_classes.keys())
values = list(two_classes.values())
plt.pie(values, labels=names, autopct='%1.2f%%')
plt.savefig(os.path.join(STATS_DIR, 'two_classes.png'))



df = pd.read_csv(LIST_FILE_PATH, sep=' ', usecols=[0, 1, 3])
binary_counters = dict(df['binary_cls_label'].value_counts())
all_classes_counters = dict(df['16cls_label'].value_counts())
groups_counters = dict.fromkeys(list(groups.keys()), 0)

for key in all_classes_counters:
    for group_key in groups:
        if key in groups[group_key]:
            groups_counters[group_key] += all_classes_counters[key]

print(groups_counters)

names = list(all_classes_counters.keys())
values = list(all_classes_counters.values())
plt.bar(range(len(all_classes_counters)),values, tick_label=names)

plt.savefig(os.path.join(STATS_DIR, 'all_classes_from_csv.png'))

plt.clf()

names = list(binary_counters.keys())
values = list(binary_counters.values())
plt.pie(values, labels=names, autopct='%1.2f%%')
plt.savefig(os.path.join(STATS_DIR, 'two_classes_from_csv.png'))


plt.clf()

names = list(groups_counters.keys())
values = list(groups_counters.values())
plt.pie(values, labels=names, autopct='%1.2f%%')
plt.savefig(os.path.join(STATS_DIR, 'groups_from_csv.png'))

plt.clf()


plt.bar(range(len(groups_counters)),values, tick_label=names)
plt.savefig(os.path.join(STATS_DIR, 'two_classes_from_csv_bar.png'))



'''


# LINE PLOTS
results =  {"1": {"ViT": [0.631,0.698,0.459,0.304,0.501,0.516,0.398,0.48,0.461,0.5,0.399,0.531,0.364,0.373,0.417,0.414] , 
                 "EfficientNet": [0.737,0.669,0.425,0.203,0.369,0.256,0.416,0.383,0.458,0.439,0.296,0.48,0.35,0.447,0.367,0.564],
                 "Hybrid (Mean)": [0.754,0.712,0.403,0.16,0.386,0.35,0.35,0.423,0.427,0.455,0.285,0.474,0.273,0.373,0.331,0.45]},
            "2": {"ViT": [0.820,0.194,0.640,0.168,0.194,0.299,0.238,0.164,0.201,0.432,0.179,0.245,0.181,0.360,0.136,0.232],
                 "EfficientNet": [0.709,0.425,0.838,0.407,0.416,0.449,0.539,0.431,0.556,0.598,0.344,0.605,0.545,0.515,0.381,0.571],
                 "Hybrid (Mean)": [0.843,0.214,0.72,0.197,0.245,0.319,0.295,0.151,0.279,0.492,0.168, 0.326,0.224,0.385,0.151,0.314]},
            "3": {"ViT": [0.756,0.187,0.226,0.595,0.169,0.276,0.394,0.245,0.327,0.242,0.250,0.200,0.280,0.300,0.100,0.403],
                "EfficientNet": [0.873,0.12,0.202,0.698,0.146,0.167,0.247,0.089,0.280,0.182,0.125,0.091,0.174,0.137,0.093,0.218],
                "Hybrid (Mean)": [0.866,0.088,0.137,0.632,0.093,0.154,0.276,0.092,0.261,0.136, 0.128,0.091,0.133,0.124,0.086,0.229],
                "Hybrid (CNN+VIT)": [ 0.637,0.401,0.495,0.846,0.381,0.443,0.586,0.307,0.658,0.364,0.473,0.348,0.468,0.416,0.388,0.493]},
            "4": {"ViT": [0.545,0.529,0.666,0.435,0.705,0.698,0.48,0.385,0.432,0.682,0.433,0.526,0.462,0.528,0.489,0.521],
                "EfficientNet": [0.811,0.222,0.34,0.19,0.695,0.556,0.315,0.218,0.254,0.758,0.234,0.28,0.294,0.335,0.223,0.386],
                "Hybrid (Mean)": [0.79,0.252,0.428,0.198,0.722,0.618,0.313,0.232,0.258,0.758,0.228, 0.337,0.315,0.366,0.245,0.382],
                "Hybrid (CNN+VIT)": [ 0.757,0.335,0.477,0.311,0.781,0.680,0.454,0.291,0.392,0.818,0.293,0.468,0.384,0.441,0.288,0.550 ]},
            "5": {"ViT": [0.721,0.326,0.391,0.338,0.365,0.627,0.319,0.272,0.298,0.386,0.296,0.366,0.329,0.255,0.237,0.357],
                "EfficientNet": [0.653,0.312,0.445,0.509,0.536,0.931,0.431,0.326,0.371,0.432,0.43,0.366,0.503,0.329,0.36,0.514],
                "Hybrid (Mean)":[0.748,0.268,0.374,0.374,0.419,0.824,0.337,0.229,0.273,0.348,0.308,0.297,0.392,0.248,0.266,0.418]},
            "6": {"ViT": [0.762,0.253,0.277,0.329,0.243,0.271,0.524,0.334,0.420,0.416,0.350,0.324,0.251,0.360,0.266,0.442],
                "EfficientNet": [0.715,0.353,0.504,0.538,0.381,0.424,0.824,0.361,0.404,0.704,0.330,0.457,0.405,0.658,0.323,0.646],
                "Hybrid (Mean)": [0.812,0.238,0.388,0.386,0.258,0.31,0.702,0.294,0.355,0.568,0.271, 0.394,0.224,0.54,0.194,0.536],
                "Hybrid (CNN+VIT)": [0.919,0.078,0.182,0.194,0.103,0.128,0.524,0.08,0.15,0.452,0.125,0.188,0.126,0.428,0.129,0.318]},
            "7": {"ViT": [0.74,0.358,0.378,0.27,0.348,0.328,0.473,0.695,0.683,0.402,0.487,0.474,0.301,0.373,0.317,0.414],
                "EfficientNet": [0.756,0.333,0.36,0.18,0.326,0.266,0.351,0.817,0.683,0.424,0.459,0.469,0.259,0.398,0.374,0.471],
                "Hybrid (Mean)":[0.819,0.269,0.309,0.145,0.258,0.202,0.312,0.757,0.69,0.402,0.402, 0.4,0.203,0.255,0.245,0.393]},
            "8": {"ViT": [0.812,0.262,0.309,0.165,0.186,0.229,0.267,0.418,0.664,0.28,0.319,0.274,0.147,0.168,0.158,0.286],
                "EfficientNet": [0.631,0.475,0.539,0.277,0.439,0.34,0.368,0.65,0.894,0.545,0.479,0.537,0.392,0.54,0.353,0.543],
                "Hybrid (Mean)":[0.798,0.282,0.354,0.126,0.227,0.208,0.232,0.499,0.796,0.318,0.362,0.309,0.161,0.224,0.151,0.311]},
            "9": {"ViT": [0.615,0.472,0.624,0.355,0.529,0.680,0.536,0.496,0.503,0.651,0.433,0.537,0.426,0.491,0.474,0.550],
                "EfficientNet": [0.758,0.306,0.411,0.189,0.458,0.433,0.451,0.399,0.458,0.712,0.328,0.508,0.273,0.484,0.396,0.496],
                "Hybrid (Mean)":[0.734,0.377,0.545,0.223,0.472,0.559,0.469,0.415,0.458,0.697,0.33,0.48, 0.259,0.453,0.41, 0.529]},
            "10": {"ViT": [0.617,0.476,0.427,0.383,0.388,0.458,0.533,0.606,0.671,0.333,0.689,0.411,0.384,0.310,0.432,0.432],
                "EfficientNet": [0.633,0.472,0.404,0.226,0.416,0.361,0.454,0.639,0.594,0.598,0.516,0.508,0.342,0.497,0.482,0.510 ],
                "Hybrid (Mean)": [0.648,0.48,0.383,0.255,0.361,0.362,0.497,0.606,0.671,0.371,0.638, 0.411,0.329,0.335,0.403,0.432]},
            "11": {"ViT": [0.578,0.481,0.555,0.388,0.427,0.609,0.526,0.604,0.623,0.561,0.481,0.594,0.51,0.478,0.439,0.629],
                "EfficientNet": [0.555,0.591,0.626,0.358,0.555,0.425,0.553,0.636,0.733,0.712,0.501,0.754,0.601,0.621,0.604,0.736],
                "Hybrid (Mean)":[0.599,0.509,0.607,0.318,0.452,0.557,0.534,0.644,0.718,0.621,0.47,0.72, 0.476,0.522,0.468,0.668],
                "Hybrid (CNN+VIT)": [0.749,0.365,0.445,0.178,0.291,0.272,0.340,0.337,0.483,0.507,0.259,0.612,0.301,0.428,0.453,0.543]},
            "12": {"ViT": [0.785,0.312,0.297,0.242,0.329,0.441,0.245,0.196,0.229,0.272,0.250,0.280,0.700,0.192,0.165,0.385],
                "EfficientNet": [0.866,0.183,0.253,0.213,0.270,0.354,0.144,0.172,0.175,0.325,0.156,0.228,0.811,0.198,0.165,0.310],
                "Hybrid (Mean)":[0.899,0.159,0.195,0.149,0.231,0.354,0.132,0.113,0.123,0.242,0.12, 0.194,0.825,0.099,0.108,0.289]},
            "13": {"ViT": [0.593,0.503,0.575,0.456,0.550,0.483,0.686,0.633,0.653,0.598,0.521,0.577,0.384,0.658,0.460,0.650],
                "EfficientNet": [0.615,0.510,0.612,0.259,0.538,0.442,0.765,0.598,0.608,0.825,0.404,0.720,0.440,0.819,0.553,0.814],
                "Hybrid (Mean)":[0.682,0.447,0.607,0.244,0.503,0.408,0.736,0.58,0.623,0.742,0.427,0.674,0.364,0.764,0.496,0.764],
                "Hybrid (CNN+VIT)": [0.673,0.426,0.582,0.237,0.431,0.419,0.695,0.528,0.582,0.757,0.380,0.64,0.426,0.720,0.424,0.686]},
            "14": {"ViT": [0.776,0.293,0.299,0.213,0.317,0.313,0.291,0.299,0.300,0.311,0.228,0.257,0.231,0.236,0.575,0.371],
                "EfficientNet": [0.520,0.634,0.627,0.290,0.592,0.456,0.600,0.668,0.696,0.750,0.530,0.783,0.468,0.758,0.820,0.803],
                "Hybrid (Mean)":[0.578,0.581,0.556,0.32,0.559,0.527,0.572,0.601,0.659,0.652,0.51,0.743,0.476,0.64,0.791,0.736]},
            "15": {"ViT": [0.798,0.244,0.284,0.237,0.276,0.312,0.351,0.247,0.297,0.325,0.207,0.365,0.321,0.310,0.151,0.614],
                  "EfficientNet": [0.909,0.108,0.138,0.070,0.100,0.212,0.105,0.070,0.111,0.159,0.094,0.177,0.167,0.142,0.080,0.603],
                  "Hybrid (Mean)": [0.969,0.05,0.087,0.047,0.068,0.116,0.082,0.038,0.054,0.114,0.037,0.086,0.063,0.068,0.043,0.486]}}
           #"1,2,3": {"ViT": [0.593,0.622,0.603,0.64,0.491,0.517,0.539,0.488,0.51,0.568,0.459,0.497,0.462,0.509,0.367,0.568],
           #             "EfficientNet": [0.798,0.47,0.57,0.811,0.328,0.346,0.476,0.27,0.516,0.477,0.265,0.389,0.441,0.342,0.273,0.507]},
           #"7,8,10": {"ViT": [0.631,0.496,0.471,0.36,0.431,0.465,0.57,0.741,0.775,0.455,0.627,0.577,0.455,0.416,0.41,0.5],
           #             "EfficientNet": [0.716,0.398,0.448,0.212,0.377,0.277,0.32,0.674,0.801,0.394,0.43,0.423,0.322,0.385,0.353,0.489]}}

for method in results:
    plt.xlabel("Deepfake Generation Method")
    plt.ylabel("Accuracy")
    
    
    vit_results = results[method]["ViT"]
    plt.xticks(np.arange(0, 16, step=1))
    length = len(vit_results)
    ranges = [[min(vit_results) for i in range(length)], [max(vit_results) for i in range(length)]]
    plt.plot(ranges[0], linestyle='dashed', color='royalblue')
    plt.plot(vit_results, label = "ViT-Base", color='royalblue')
    plt.plot(ranges[1], linestyle='dashed', color='royalblue')
    
    
    efficientnet_results = results[method]["EfficientNet"]
    ranges = [[min(efficientnet_results) for i in range(length)], [max(efficientnet_results) for i in range(length)]]
    plt.plot(ranges[0], linestyle='dashed', color='firebrick')
    plt.plot(efficientnet_results, label="EfficientNetV2-M", color='firebrick')
    plt.plot(ranges[1], linestyle='dashed', color='firebrick')

    '''
    hybrid_mean_results = results[method]["Hybrid (Mean)"]
    ranges = [[min(hybrid_mean_results) for i in range(length)], [max(hybrid_mean_results) for i in range(length)]]
    plt.plot(ranges[0], linestyle='dashed', color='green')
    plt.plot(hybrid_mean_results, label="Hybrid (Mean)", color='green')
    plt.plot(ranges[1], linestyle='dashed', color='green')
    '''
    
    if "Hybrid (CNN+VIT)" in results[method]:

        hybrid_results = results[method]["Hybrid (CNN+VIT)"]
        ranges = [[min(hybrid_results) for i in range(length)], [max(hybrid_results) for i in range(length)]]
        plt.plot(ranges[0], linestyle='dashed', color='green')
        plt.plot(hybrid_results, label="Hybrid (CNN+VIT)", color='green')
        plt.plot(ranges[1], linestyle='dashed', color='green')



    plt.legend(bbox_to_anchor = (1.05, 0.6))
    plt.title("Training Set: " + method)
    plt.savefig(os.path.join("plots", method + ".png"), bbox_inches='tight')
    plt.clf()
