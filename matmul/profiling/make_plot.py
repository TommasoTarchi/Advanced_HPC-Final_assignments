import argparse
import csv
import matplotlib.pyplot as plt


# get csv file name
parser = argparse.ArgumentParser()
parser.add_argument('--matmul', type=str, choices=("simple", "blas", "cublas"))
args = parser.parse_args()
matmul = args.matmul
csv_file = ""
profiling_name = ""
if matmul == "simple":
    csv_file = "times_simple.csv"
    profiling_name = "profiling_simple.png"
elif matmul == "blas":
    csv_file = "times_blas.csv"
    profiling_name = "profiling_blas.png"
elif matmul == "cublas":
    csv_file = "times_cublas.csv"
    profiling_name = "profiling_cublas.png"

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
