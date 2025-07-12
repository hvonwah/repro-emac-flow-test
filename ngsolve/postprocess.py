import pandas as pd
import numpy as np
from math import sqrt, log2


def lyapunov_exp(drag_x, drag_y, approx_half_period_idx, shift, m):
    offset = int(1.2 * approx_half_period_idx)
    nde = int(1.8 * approx_half_period_idx)
    diff = np.zeros(nde)
    for k in range(nde):
        d_k = 0
        for j in range(m):
            d_k += (drag_x[shift + j] - drag_x[shift + offset + k + j])**2
            d_k += (drag_y[shift + j] - drag_y[shift + offset + k + j])**2
        diff[k] = sqrt(d_k)

    i, dr = np.argmin(diff), 0
    if i > 0 and i < nde - 1:
        a, b, c = diff[i - 1:i + 2]
        dr = 0.5 * (c - a) / (2 * b - a - c)
    return i + dr + offset


approx_period = 8.8
sample_freq = 0.1
seg_len = 5
t_start = 280
t_end = 480


# Main loop over set of parameters
for file_pars in ['parameters.txt', 'parameters_imex.txt']:

    pars = pd.read_csv(file_pars, sep=' ', index_col=0, header=0)

    for i, p in pars.iterrows():
        if file_pars == 'parameters_imex.txt':
            bdf = 'IMEXSBDF2'
        else:
            bdf = p['bdf']
        Re, k, mesh, h = float(p['Re']), p['k'], p['mesh'], float(p['h'])
        dt, conv = p['dt'], p['conv']

        # Load data -----------------------------------------------------------
        base_dir = '../results/'
        file_i = f'cylinder_flow_Re{Re}_TH{k}{conv}_h{h}mesh{mesh}{bdf}dt{dt}'
        try:
            df = pd.read_csv(f'{base_dir}{file_i}.txt', sep=' ')
        except FileNotFoundError:
            print(f'File {base_dir}{file_i} not found, skipping...')
            continue

        # Strouhal period -----------------------------------------------------
        if df["time"].iloc[-1] < 330:
            out = f'{Re} {conv} {bdf} - {k} & {mesh}, $\\ell={int(log2(8 / h))}'
            out += f'$ & {dt} & {df["time"].iloc[-1]:.2f} & Solver failed\\\\'
            print(out)
            continue
        else:
            t_end = df["time"].iloc[-1] - 20

        approx_half_period_idx = int(approx_period / 2 / dt)
        start_idx = int(t_start / dt)
        end_idx = int(t_end / dt)
        sample_freq_idx = int(sample_freq / dt)
        many = int((end_idx - start_idx) / sample_freq_idx)
        v, p = np.zeros(many), np.zeros(many)

        drag_x = df['dragvol'].to_numpy()
        drag_y = df['liftvol'].to_numpy()
        div = df['div'].to_numpy()

        for j in range(many):
            search_idx = start_idx + j * sample_freq_idx
            len_per_idx = lyapunov_exp(drag_x, drag_y, approx_half_period_idx,
                                       shift=search_idx, m=seg_len)
            p[j] = dt * len_per_idx

        # Process Results -----------------------------------------------------
        period_mean, period_std = p.mean(), p.std()

        df2 = df[(df['time'] >= t_start) & (df['time'] <= t_end)]
        df_mean, df_std = df2.mean(axis=0), df2.std(axis=0)
        out = f'{Re} {conv} {bdf} - {k} & {mesh}, $\\ell={int(log2(8 / h))}$ '
        out += f'& {dt} & {t_end:.2f} & {df_mean["dragvol"]:.3f} & '
        out += f'${period_mean:.3f}\\pm{period_std:.5f}$ &'
        out += f' {df2["div"].max():.1e} \\\\'

        print(out)
