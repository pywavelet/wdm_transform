import re

with open("lisa_freq_mcmc.py", "r") as f:
    code = f.read()

# We need to collect the optimal SNRs.
code = code.replace(
    '''for i, src in enumerate(SOURCE_PARAMS):
    k_center = int(np.rint(src[0] * t_obs))
    kmin = max(k_center - jgb.n, 0)
    kmax = min(k_center + jgb.n, n_freq)
    band_freqs = freqs[kmin:kmax]
    noise_band = np.maximum(np.interp(band_freqs, freqs, noise_psd_full), 1e-60)
    
    h = source_a_band_jax(jnp.asarray(src, dtype=jnp.float64), int(kmin), int(kmax))
    snr = matched_filter_snr_rfft(
        np.asarray(h), noise_band, band_freqs, dt=dt
    )
    print(f"  GB {i + 1}: SNR = {snr:.1f}")''',
    '''snrs_optimal = []
for i, src in enumerate(SOURCE_PARAMS):
    k_center = int(np.rint(src[0] * t_obs))
    kmin = max(k_center - jgb.n, 0)
    kmax = min(k_center + jgb.n, n_freq)
    band_freqs = freqs[kmin:kmax]
    noise_band = np.maximum(np.interp(band_freqs, freqs, noise_psd_full), 1e-60)
    
    h = source_a_band_jax(jnp.asarray(src, dtype=jnp.float64), int(kmin), int(kmax))
    snr = matched_filter_snr_rfft(
        np.asarray(h), noise_band, band_freqs, dt=dt
    )
    snrs_optimal.append(float(snr))
    print(f"  GB {i + 1}: SNR = {snr:.1f}")'''
)

save_str = "    all_samples.append(samples_i)"
snr_code = """
    # Calculate SNR for each sample
    samples_i_full = np.tile(SOURCE_PARAMS[i], (samples_i.shape[0], 1))
    samples_i_full[:, 0] = samples_i[:, 0]
    samples_i_full[:, 1] = samples_i[:, 1]
    samples_i_full[:, 2] = samples_i[:, 2]
    samples_i_full[:, 7] = samples_i[:, 3]

    psd_j = jnp.asarray(band.noise_psd, dtype=jnp.float64)
    df_j = jnp.asarray(df, dtype=jnp.float64)
    kmin_stat = band.band_kmin
    kmax_stat = band.band_kmax

    @jax.jit
    def _get_snrs(ps):
        def _single_snr(p):
            h = source_a_band_jax(p.reshape(1, -1), kmin_stat, kmax_stat)
            h_tilde = dt * h
            snr2 = 4.0 * df_j * jnp.sum(jnp.abs(h_tilde)**2 / psd_j)
            return jnp.sqrt(snr2)
        return jax.vmap(_single_snr)(ps)

    snr_samples = np.asarray(_get_snrs(jnp.array(samples_i_full)))
    samples_i = np.column_stack([samples_i, snr_samples])
    truth_i = np.append(truth_i, snrs_optimal[i])
    PARAM_NAMES_PLOT = PARAM_NAMES + ["SNR"]

    all_samples.append(samples_i)
"""

code = code.replace(save_str, snr_code)

npz_str = """np.savez(
    _out_path,
    source_params=SOURCE_PARAMS,
    samples_gb1=all_samples[0],
    samples_gb2=all_samples[1],
)"""
npz_code = """np.savez(
    _out_path,
    source_params=SOURCE_PARAMS,
    samples_gb1=all_samples[0],
    samples_gb2=all_samples[1],
    snr_optimal=snrs_optimal,
)"""

code = code.replace(npz_str, npz_code)

corner_str = """corner_labels = [r"$f_0$", r"$\dot{f}$", r"$A$", r"$\phi_0$"]
for i, (samples_i, stem) in enumerate(
    zip(all_samples, ["gb1_corner", "gb2_corner"], strict=True)
):
    truth_i = [SOURCE_PARAMS[i, 0], SOURCE_PARAMS[i, 1],
               SOURCE_PARAMS[i, 2], wrap_phase(SOURCE_PARAMS[i, 7])]
    fig = corner.corner(
        samples_i, labels=corner_labels, truths=truth_i, truth_color="tab:red",
        quantiles=[0.05, 0.5, 0.95], show_titles=True,
    )"""

corner_code = """corner_labels = [r"$f_0$", r"$\dot{f}$", r"$A$", r"$\phi_0$", "SNR"]
for i, (samples_i, stem) in enumerate(
    zip(all_samples, ["gb1_corner", "gb2_corner"], strict=True)
):
    truth_i = [SOURCE_PARAMS[i, 0], SOURCE_PARAMS[i, 1],
               SOURCE_PARAMS[i, 2], wrap_phase(SOURCE_PARAMS[i, 7]), snrs_optimal[i]]
    fig = corner.corner(
        samples_i, labels=corner_labels, truths=truth_i, truth_color="tab:red",
        quantiles=[0.05, 0.5, 0.95], show_titles=True,
        title_kwargs={"fontsize": 10}
    )"""

code = code.replace(corner_str, corner_code)

with open("lisa_freq_mcmc.py", "w") as f:
    f.write(code)
