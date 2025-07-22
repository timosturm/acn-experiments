from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
from evaluation.utils import pv_for_sim
from gymportal.plotting.plotting import plot_sim_evaluation


def plot_sim_evaluation_pv(sim, df_pv, plot_rewards=True):
    fig = plot_sim_evaluation(sim, plot_rewards=plot_rewards)

    if plot_rewards:
        ax = fig.axes[-2]
    else:
        ax = fig.axes[-1]

    # TODO Convert P to A and use the right timesteps
    amps = pv_for_sim(sim, df_pv)

    ax.plot(amps, label="PV", color="green", alpha=0.5)
    ax.legend()

    def minutes_to_hhmm(x, _):
        hours = int(x) // 60
        minutes = int(x) % 60
        return f'{hours:02d}:{minutes:02d}'

    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FuncFormatter(minutes_to_hhmm))
        ax.xaxis.set_minor_locator(MultipleLocator(60))

        ax.set_xticks(np.arange(0, 1441, 120))
        ax.grid(True, axis='x', which='major')
        ax.grid(True, axis='x', which='minor', linestyle='--', alpha=0.5)
        ax.set_xlabel("Wall time in hh:mm")

    fig.show()
    return fig
