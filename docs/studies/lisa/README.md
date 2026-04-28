# LISA Study Notes

## Current status

The local LISA study is in a better state than the earlier runs, but it should still be treated as
an active debugging / calibration exercise rather than a finished inference pipeline.

The main improvements so far are:

- The local `f0` prior was tightened to a narrow absolute offset around the fixed reference
  frequency. The current default is `delta_f0 = f0 - f_ref ∈ [-3e-8, 3e-8] Hz`.
- Exploratory likelihood plotting was removed from this directory so the production MCMC and
  posterior-comparison path stays small.
- In the frequency-domain run, `phi0` is no longer sampled directly. The current sampler samples
  `(delta_f0, logfdot, logA)` and profiles over `phi0` inside the likelihood.
- In the WDM-domain run, `phi0` is also profiled out while `logA` remains sampled. The WDM sampler
  now samples `(delta_logf0, logfdot, logA)`.

These changes removed the worst failure mode seen earlier, where chains could settle into a clearly
wrong joint basin with biased `f0`, `A`, and `phi0`.

## Important caveat

Things are better, but not fully solved.

Empirically, some runs still appear to miss the dominant mode or fail to explore it robustly. The
trace plots can look reasonable for `A` and the profiled `phi0` while `f0` still appears fragile
across seeds. This means:

- good-looking single-run traces are not sufficient evidence of calibration,
- PP plots should be interpreted cautiously,
- repeated-seed studies are still necessary before trusting coverage claims.

At the moment, `phi0` is a profiled / derived quantity in both samplers, not a fully sampled
posterior parameter. That means a PP plot for `phi0` is not a standard posterior calibration test
and should not be interpreted the same way as `f0`, `fdot`, or `A`.

## What to trust most right now

If checking repeated runs, the most meaningful parameters are:

- `f0`
- `fdot`
- `A`
- derived `SNR`

`phi0` can still be saved and compared as a profiled estimator, but it should not currently be used
as a headline calibration variable.

## Known unresolved issue

The frequency-domain template / injection overlap sanity check still reports a mismatch on noisy
data, even though the pure-source overlap check is exact in the frequency-domain script. That issue
has not been resolved by the sampler changes above and may still affect calibration or mode
structure.

## Practical takeaway

The current sampler choices are pragmatic:

- sample amplitude,
- profile over phase,
- keep the local frequency prior narrow,
- keep diagnostics separate from the main MCMC scripts.

This is currently the most stable setup found in this study, but it should still be regarded as
provisional until repeated-seed tests show that the main mode is recovered reliably.
