import argparse
import csv
import matplotlib.pyplot as plt


# get csv file name
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--plot_path', type=str)
args = parser.parse_args()
data_path = args.data_path
plot_path = args.plot_path

# get data
with open(data_path, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

data = data[1:]  # skip header

labels = [row[0] for row in data]
values = [[float(row[i]) for i in range(1, 6)] for row in data]
#values = [[float(row[i]) for i in (2, 3, 5)] for row in data]

# make plot
fig, ax = plt.subplots()
bottom = None
for i, section_label in enumerate(["initialization", "communication", "computation", "host_device_once", "host_device_iterations"]):
#for i, section_label in enumerate(["communication", "computation", "host_device_iterations"]):
    bar = [value[i] for value in values]
    if bottom is None:
        ax.bar(labels, bar, label=section_label)
        bottom = bar
    else:
        ax.bar(labels, bar, bottom=bottom, label=section_label)
        bottom = [bottom[j] + bar[j] for j in range(len(bar))]

# set labels
ax.set_xlabel('# processes')
ax.set_ylabel('time')
ax.legend()
plt.xticks()
plt.tight_layout()
plt.savefig(plot_path)
