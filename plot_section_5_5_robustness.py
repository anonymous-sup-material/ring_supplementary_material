import matplotlib.pyplot as plt
import myplotlib
import numpy as np
import ring

myplotlib.setup(major_fontsize=9, minor_fontsize=7)

data = ring.utils.pickle_load("eval_section_5_5_robustness.pickle")

method_data = {
    "RIANN": ("(1)", "C0"),
    "MadgwickAttitude": ("(2)", "C1"),
    "MahonyAttitude": ("(3)", "C2"),
    "SeelAttitude": ("(4)", "C3"),
    "VQFAttitude": ("(5)", "C4"),
    "VQF_9D": ("(5)", "C4"),
    "1d_corr_ollson_0": ("(5)+(6)", "C5"),
    "euler_1d_ollson_0": ("(5)+(7)", "C6"),
    "1d_corr_ollson_1": ("(5)+(6)+(8)", "C5"),
    "euler_1d_ollson_1": ("(5)+(7)+(8)", "C6"),
    "RNNO": ("(9)", "C7"),
    "RING": ("RING", "C8"),
}

n_cols = 4
n_rows = 2

fig, axes = plt.subplots(
    n_rows, n_cols, figsize=myplotlib.figsize(2, subplots=(n_rows, n_cols))
)
axes = np.atleast_2d(axes)
sections = list(data.keys())

for row in range(n_rows):
    for col in range(n_cols):
        ax = axes[row, col]
        i = row * 4 + col
        d = data[sections[i]]

        for method_name, (label, color) in method_data.items():
            if method_name not in d.keys():
                continue

            if method_name != "RING":
                continue

            values = d[method_name]
            xs = list(values.keys())
            ys = np.array([np.mean(values[k]) for k in xs])
            ys_std = np.array([np.std(values[k]) for k in xs])

            xs = np.array(xs) / max(xs) * 120
            ax.plot(xs, ys, label=label, color=color)
            ax.fill_between(
                xs, ys - ys_std, ys + ys_std, alpha=0.2, color=color, linewidth=0
            )

        ax.legend()
        ax.grid()
        ax.set_title(f"Section {sections[i].replace('_', '.')}")
        ax.set_xticks(xs)
        if row == 1:
            ax.set_xlabel("Noise and Bias [%]")
        ax.set_ylabel("AMAE [deg]" if col == 0 else "RMAE [deg]")

plt.subplots_adjust(wspace=0.3, hspace=0.35)
myplotlib.savefig("plot_section_5_5_robustness.pdf")
