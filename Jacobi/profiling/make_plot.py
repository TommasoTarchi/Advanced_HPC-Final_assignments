import csv
import matplotlib.pyplot as plt


csv_file = "times.csv"


with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

data = data[1:]

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
ax.set_title('Profiling of parallel Jacobi')
ax.legend()

plt.xticks()
plt.tight_layout()
plt.savefig('profiling.png')
