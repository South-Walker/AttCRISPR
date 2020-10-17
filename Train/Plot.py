import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(figsize=(12, 5))

all_data = [np.random.normal(0, std, 10) for std in range(6, 10)]

axes.violinplot(all_data,
                   showmeans=False,
                   showmedians=True
                   )
axes.set_title('violin plot')

# adding horizontal grid lines

axes.yaxis.grid(True)
axes.set_xticks([y + 1 for y in range(len(all_data))], )
axes.set_xlabel('xlabel')
axes.set_ylabel('ylabel')

plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
         xticklabels=['x1', 'x2', 'x3', 'x4'],
         )

plt.show()
