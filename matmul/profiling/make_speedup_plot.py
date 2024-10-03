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

# compute the speedup relative to computation
n_procs = [int(row[0]) for row in data]
computation_times = [float(row[3]) for row in data]

sequential_time = computation_times[0]
speedup = [sequential_time / comp_time for comp_time in computation_times]

ideal_speedup = n_procs

# make plot
fig, ax = plt.subplots()
ax.plot(n_procs, speedup, marker='o', linestyle='-', color='b', label='Computation Speedup')

ax.plot(n_procs, ideal_speedup, linestyle='--', color='r', label='Ideal Speedup')
ax.fill_between(n_procs, 0, ideal_speedup, color='red', alpha=0.1)

# set labels
ax.set_xlabel('# processes')
ax.set_ylabel('time')
ax.legend()
plt.xticks()
plt.tight_layout()
plt.savefig(plot_path)
