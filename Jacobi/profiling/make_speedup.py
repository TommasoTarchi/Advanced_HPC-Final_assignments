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

# skip header row
header = data[0]
data = data[1:]

# compute the speedup
n_procs = [int(row[0]) for row in data]
times = [float(row[1]) + float(row[2]) + float(row[3]) + float(row[4]) + float(row[5]) for row in data]

sequential_time = times[0]
speedup = [sequential_time / time for time in times]

ideal_speedup = n_procs

# make plot
fig, ax = plt.subplots()
ax.plot(n_procs, speedup, marker='o', linestyle='-', color='b', label='Speedup')

ax.plot(n_procs, ideal_speedup, linestyle='--', color='r', label='Ideal Speedup')
ax.fill_between(n_procs, 0, ideal_speedup, color='red', alpha=0.1)

# set labels
ax.set_xlabel('# processes')
ax.set_ylabel('time')
ax.legend()
plt.xticks()
plt.tight_layout()
plt.savefig(plot_path)
