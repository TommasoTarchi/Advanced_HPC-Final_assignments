import argparse
import csv
import matplotlib.pyplot as plt


# get csv file name
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=("openMP", "openACC", "aware", "MPI-RMA"))
args = parser.parse_args()
jacobi = args.mode
csv_file = ""
profiling_name = ""
if jacobi == "openMP":
    csv_file = "times_openMP.csv"
    profiling_name = "profiling_openMP.png"
elif jacobi == "openACC":
    csv_file = "times_openACC.csv"
    profiling_name = "profiling_openACC.png"
elif jacobi == "aware":
    csv_file = "times_aware.csv"
    profiling_name = "profiling_aware.png"
elif jacobi == "MPI-RMA":
    csv_file = "times_MPI-RMA.csv"
    profiling_name = "profiling_MPI-RMA.png"

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

data = data[1:]  # skip header

labels = [row[0] for row in data]
values = [[float(row[i]) for i in range(1, 4)] for row in data]

fig, ax = plt.subplots()
bottom = None
for i, section_label in enumerate(["initialization", "communication", "computation"]):
    bar = [value[i] for value in values]
    if bottom is None:
        ax.bar(labels, bar, label=section_label)
        bottom = bar
    else:
        ax.bar(labels, bar, bottom=bottom, label=section_label)
        bottom = [bottom[j] + bar[j] for j in range(len(bar))]

ax.set_xlabel('# nodes')
ax.set_ylabel('time')
#ax.set_title('')
ax.legend()

plt.xticks()
plt.tight_layout()
plt.savefig(profiling_name)
