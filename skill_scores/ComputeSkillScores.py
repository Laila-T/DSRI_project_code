import glob
import os

import numpy
from scipy import stats


DATA_DIRECTORY = r"C:\Users\Matthew\Downloads\CVC"
filepaths = glob.glob(os.path.join(DATA_DIRECTORY, "**", "*.tsv"), recursive=True)

# Compile a dictionary of metrics
recording_names = []
metrics_dict = dict()
for curr_filepath in filepaths:
  curr_table = numpy.loadtxt(curr_filepath, delimiter="\t", dtype="str", skiprows=1)

  curr_metrics_dict = dict()
  for row in range(curr_table.shape[0]):
    metric = " ".join(curr_table[row, :-1])
    value = float(curr_table[row, -1])
    curr_metrics_dict[metric] = value

  for metric in curr_metrics_dict:
    if metric not in metrics_dict:
      metrics_dict[metric] = []
    metrics_dict[metric].append(curr_metrics_dict[metric])

  recording_names.append(curr_filepath)

print(metrics_dict)

# Compute z-scores for each metric
for metric in metrics_dict:
  metrics_dict[metric] = stats.zscore(metrics_dict[metric])

print(metrics_dict)

# Compute skill score for each intervention by averaging z-scores
skill_scores = [0] * len(recording_names)
for i in range(len(recording_names)):
  curr_zscores = []
  for values in metrics_dict.values():
    curr_zscores.append(values[i])
  skill_scores[i] = numpy.mean(curr_zscores)
  
print(skill_scores)

# Write the combined skill scores to file
output_list = list(map(list, zip(recording_names, map(str, skill_scores))))
print(output_list)
numpy.savetxt("blah.csv", output_list, fmt="%s")

