import matplotlib.pyplot as plt
import myplotlib
import numpy as np
import ring

myplotlib.setup(major_fontsize=10, minor_fontsize=8)

data = ring.utils.pickle_load("eval_section_5_5_robustness.pickle")

bc_s1 = "dimgray"
bc_s2 = "gray"
bc_s3 = "darkgray"
bc_s4 = "silver"
bc_s5 = "lightgray"

method_data = {
    "RIANN": ("(1)", bc_s1),
    "MadgwickAttitude": ("(2)", bc_s2),
    "MahonyAttitude": ("(3)", bc_s3),
    "SeelAttitude": ("(4)", bc_s4),
    "VQFAttitude": ("(5)", bc_s5),
    "VQF_9D": ("(5)", bc_s2),
    "1d_corr_ollson_0": ("(5)+(6)", bc_s3),
    "euler_1d_ollson_0": ("(5)+(7)", bc_s4),
    "1d_corr_ollson_1": ("(5)+(6)+(8)", bc_s3),
    "euler_1d_ollson_1": ("(5)+(7)+(8)", bc_s4),
    "RNNO": ("(9)", bc_s3),
    "RING": ("RING", "cornflowerblue"),
}
ylim_max = [13, 30, 30, 8, 14, 30, 9, 11]


def evenly_spaced_centered_around_zero(n):
    if n == 1:
        return [0]
    else:
        step = 1
        start = -(n - 1) / 2
        return [start + i * step for i in range(n)]


n_cols = 4
n_rows = 2
offset_delta = 2

fig, axes = plt.subplots(
    n_rows, n_cols, figsize=myplotlib.figsize(2, subplots=(n_rows, n_cols)), sharex=True
)
axes = np.atleast_2d(axes)
sections = list(data.keys())

for row in range(n_rows):
    for col in range(n_cols):
        ax = axes[row, col]
        i = row * 4 + col
        d = data[sections[i]]

        if sections[i] == "5_3_1A":
            d.pop("1d_corr_ollson_0")
            d.pop("euler_1d_ollson_0")

        x_offsets = evenly_spaced_centered_around_zero(len(d))
        i_method = -1
        for method_name, (label, color) in method_data.items():
            if method_name not in d.keys():
                continue
            i_method += 1

            values = d[method_name]
            xs = list(values.keys())
            ys = np.array([np.mean(values[k]) for k in xs])
            ys_std = np.array([np.std(values[k]) for k in xs])
            xs = np.array(xs) / max(xs) * 120

            x_offset = x_offsets[i_method]
            xs += offset_delta * x_offset

            bplot = ax.boxplot(
                [np.array(values[k]) for k in list(values.keys())],
                vert=True,
                positions=xs,
                patch_artist=True,
                widths=1.5,
                showcaps=False,
                whiskerprops=dict(linestyle="-", linewidth=0),
                flierprops=dict(marker=""),
                medianprops=dict(color="black", linewidth=1),
            )

            # fill with colors and add legend labels
            for j, patch in enumerate(bplot["boxes"]):
                patch.set_facecolor(color)
                patch.set_edgecolor("none")
                if j == 0:
                    patch.set_label(label)

            # Hide whiskers, caps, and outliers
            for element in ["whiskers", "caps", "fliers"]:
                for item in bplot[element]:
                    item.set_visible(False)

        ax.legend(ncol=2, fontsize=7, handlelength=0.75)
        ax.grid()
        ax.set_title(f"IMTP defined in Section {sections[i].replace('_', '.')}")
        ax.set_xticks(xs)
        if row == 1:
            ax.set_xlabel("Noise and Bias [%]")
        ax.set_ylabel("AMAE [deg]" if col == 0 else "RMAE [deg]")

        ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
        ax.set_xticklabels([0, 20, 40, 60, 80, 100, 120])
        ax.set_ylim(top=ylim_max[i])

plt.subplots_adjust(wspace=0.3, hspace=0.25)
myplotlib.savefig("plot_section_5_5_robustness.pdf")
