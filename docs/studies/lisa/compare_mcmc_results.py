"""Compare two LISA MCMC posterior files.

This script loads two NPZ posterior outputs and compares:
- marginal posterior histograms
- median and 90% credible intervals
- optional source-level SNR values (if available in the files)

Expected input formats:
- WDM export from lisa_wdm_mcmc.py (theta_samples + labels + truth)
- A compatible file with equivalent labels, or a freq-style file containing
  gb1/gb2 samples with columns [f0, fdot, A, phi0]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
	import corner
except ImportError:
	corner = None

from lisa_common import save_figure


STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = STUDY_DIR / "compare_mcmc_results_assets"
DEFAULT_WDM_PATH = STUDY_DIR / "lisa_wdm_mcmc_assets" / "posteriors.npz"
DEFAULT_FREQ_PATH = STUDY_DIR / "lisa_freq_mcmc_assets" / "posteriors.npz"


@dataclass(frozen=True)
class RunPosterior:
	name: str
	path: Path
	samples: np.ndarray
	labels: list[str]
	truth_map: dict[str, float]
	snr_map: dict[str, float]


def _to_float_array(data: np.ndarray) -> np.ndarray:
	return np.asarray(data, dtype=float)


def _to_label_list(data: np.ndarray | list[str]) -> list[str]:
	arr = np.asarray(data)
	labels = []
	for item in arr.tolist():
		labels.append(str(item))
	return labels


def _normalize_phi(samples: np.ndarray, labels: list[str]) -> np.ndarray:
	wrapped = np.asarray(samples, dtype=float).copy()
	for idx, label in enumerate(labels):
		if "phi" in label.lower() or "phase" in label.lower():
			wrapped[:, idx] = (wrapped[:, idx] + np.pi) % (2.0 * np.pi) - np.pi
	return wrapped


def _try_extract_truth(data: np.lib.npyio.NpzFile, labels: list[str]) -> dict[str, float]:
	if "truth" in data:
		truth = _to_float_array(data["truth"]).reshape(-1)
		if truth.size == len(labels):
			return {label: float(val) for label, val in zip(labels, truth, strict=True)}

	if "source_params" in data:
		src = _to_float_array(data["source_params"])
		if src.shape == (2, 8):
			return {
				"source 1 frequency [Hz]": float(src[0, 0]),
				"source 1 amplitude": float(src[0, 2]),
				"source 1 phase [rad]": float((src[0, 7] + np.pi) % (2.0 * np.pi) - np.pi),
				"source 2 frequency [Hz]": float(src[1, 0]),
				"source 2 amplitude": float(src[1, 2]),
				"source 2 phase [rad]": float((src[1, 7] + np.pi) % (2.0 * np.pi) - np.pi),
			}

	return {}


def _try_extract_snr(data: np.lib.npyio.NpzFile) -> dict[str, float]:
	snr_map: dict[str, float] = {}

	for key in ["snr_source1", "source1_snr", "gb1_snr"]:
		if key in data:
			snr_map["source 1"] = float(np.asarray(data[key]).reshape(-1)[0])
			break
	for key in ["snr_source2", "source2_snr", "gb2_snr"]:
		if key in data:
			snr_map["source 2"] = float(np.asarray(data[key]).reshape(-1)[0])
			break
	if "snr" in data:
		snr_arr = np.asarray(data["snr"]).reshape(-1)
		if snr_arr.size >= 2:
			snr_map.setdefault("source 1", float(snr_arr[0]))
			snr_map.setdefault("source 2", float(snr_arr[1]))

	return snr_map


def _load_freq_style_samples(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str], dict[str, float]]:
	if "gb1_samples" in data and "gb2_samples" in data:
		gb1 = _to_float_array(data["gb1_samples"])
		gb2 = _to_float_array(data["gb2_samples"])
	elif "samples_gb1" in data and "samples_gb2" in data:
		gb1 = _to_float_array(data["samples_gb1"])
		gb2 = _to_float_array(data["samples_gb2"])
	elif "independent_samples" in data:
		indep = _to_float_array(data["independent_samples"])
		if indep.ndim != 3 or indep.shape[0] < 2 or indep.shape[2] < 4:
			raise ValueError("`independent_samples` does not match expected shape [2, ns, 4].")
		gb1, gb2 = indep[0], indep[1]
	else:
		raise ValueError("No recognized freq-style sample keys found.")

	if gb1.shape[1] < 4 or gb2.shape[1] < 4:
		raise ValueError("Freq-style samples must have columns [f0, fdot, A, phi0].")
	ns = min(gb1.shape[0], gb2.shape[0])
	merged = np.column_stack(
		[
			gb1[:ns, 0],
			gb1[:ns, 1],
			gb1[:ns, 2],
			gb1[:ns, 3],
			gb2[:ns, 0],
			gb2[:ns, 1],
			gb2[:ns, 2],
			gb2[:ns, 3],
		]
	)
	labels = [
		"source 1 frequency [Hz]",
		"source 1 chirp fdot [1/s]",
		"source 1 amplitude",
		"source 1 phase [rad]",
		"source 2 frequency [Hz]",
		"source 2 chirp fdot [1/s]",
		"source 2 amplitude",
		"source 2 phase [rad]",
	]

	truth_map: dict[str, float] = {}
	if "source_params" in data:
		src = _to_float_array(data["source_params"])
		if src.shape == (2, 8):
			truth_map = {
				labels[0]: float(src[0, 0]),
				labels[1]: float(src[0, 1]),
				labels[2]: float(src[0, 2]),
				labels[3]: float((src[0, 7] + np.pi) % (2.0 * np.pi) - np.pi),
				labels[4]: float(src[1, 0]),
				labels[5]: float(src[1, 1]),
				labels[6]: float(src[1, 2]),
				labels[7]: float((src[1, 7] + np.pi) % (2.0 * np.pi) - np.pi),
			}

	return merged, labels, truth_map


def load_run(path: Path, name: str) -> RunPosterior:
	if not path.exists():
		raise FileNotFoundError(f"Posterior file not found: {path}")

	data = np.load(path)

	if "theta_samples" in data and "labels" in data:
		samples = _to_float_array(data["theta_samples"])
		labels = _to_label_list(data["labels"])
		truth_map = _try_extract_truth(data, labels)
	elif all(k in data for k in ["f1", "A1", "phi1", "f2", "A2", "phi2"]):
		samples = np.column_stack(
			[
				_to_float_array(data["f1"]),
				_to_float_array(data["A1"]),
				_to_float_array(data["phi1"]),
				_to_float_array(data["f2"]),
				_to_float_array(data["A2"]),
				_to_float_array(data["phi2"]),
			]
		)
		labels = [
			"source 1 frequency [Hz]",
			"source 1 amplitude",
			"source 1 phase [rad]",
			"source 2 frequency [Hz]",
			"source 2 amplitude",
			"source 2 phase [rad]",
		]
		truth_map = _try_extract_truth(data, labels)
	else:
		samples, labels, truth_map = _load_freq_style_samples(data)

	samples = _normalize_phi(samples, labels)
	snr_map = _try_extract_snr(data)
	return RunPosterior(
		name=name,
		path=path,
		samples=samples,
		labels=labels,
		truth_map=truth_map,
		snr_map=snr_map,
	)


def summarize_run(run: RunPosterior, labels: list[str]) -> None:
	print(f"\n{run.name} ({run.path.name})")
	med = np.median(run.samples, axis=0)
	lo = np.percentile(run.samples, 5, axis=0)
	hi = np.percentile(run.samples, 95, axis=0)
	for idx, label in enumerate(labels):
		msg = (
			f"  {label:24s} median={med[idx]:.6e} "
			f"90% CI=[{lo[idx]:.6e}, {hi[idx]:.6e}]"
		)
		if label in run.truth_map:
			msg += f" true={run.truth_map[label]:.6e}"
		print(msg)


def compare_summary(run_a: RunPosterior, run_b: RunPosterior, labels: list[str]) -> None:
	print("\nRun-to-run median deltas (A - B)")
	med_a = np.median(run_a.samples, axis=0)
	med_b = np.median(run_b.samples, axis=0)
	for idx, label in enumerate(labels):
		print(f"  {label:24s} delta={med_a[idx] - med_b[idx]:+.6e}")


def plot_marginals(
	run_a: RunPosterior,
	run_b: RunPosterior,
	labels: list[str],
	output_dir: Path,
) -> None:
	n_params = len(labels)
	n_cols = 4
	n_rows = (n_params + n_cols - 1) // n_cols
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), constrained_layout=True)
	axes = np.asarray(axes).reshape(-1)

	for idx, label in enumerate(labels):
		ax = axes[idx]
		ax.hist(
			run_a.samples[:, idx],
			bins=40,
			density=True,
			alpha=0.45,
			color="tab:blue",
			label=run_a.name,
		)
		ax.hist(
			run_b.samples[:, idx],
			bins=40,
			density=True,
			alpha=0.45,
			color="tab:orange",
			label=run_b.name,
		)
		truth = run_a.truth_map.get(label, run_b.truth_map.get(label))
		if truth is not None:
			ax.axvline(float(truth), color="tab:red", ls="--", lw=1.5, label="truth")
		ax.set_title(label)
		ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))

	for idx in range(n_params, axes.size):
		axes[idx].set_visible(False)
	axes[0].legend(fontsize=8)
	_ = save_figure(fig, output_dir, "posterior_marginals_compare")


def plot_intervals(
	run_a: RunPosterior,
	run_b: RunPosterior,
	labels: list[str],
	output_dir: Path,
) -> None:
	med_a = np.median(run_a.samples, axis=0)
	lo_a = np.percentile(run_a.samples, 5, axis=0)
	hi_a = np.percentile(run_a.samples, 95, axis=0)
	med_b = np.median(run_b.samples, axis=0)
	lo_b = np.percentile(run_b.samples, 5, axis=0)
	hi_b = np.percentile(run_b.samples, 95, axis=0)

	x = np.arange(len(labels))
	fig_width = max(12, len(labels) * 1.2)
	fig, ax = plt.subplots(figsize=(fig_width, 5), constrained_layout=True)
	ax.errorbar(
		x - 0.12,
		med_a,
		yerr=[med_a - lo_a, hi_a - med_a],
		fmt="o",
		color="tab:blue",
		capsize=3,
		label=run_a.name,
	)
	ax.errorbar(
		x + 0.12,
		med_b,
		yerr=[med_b - lo_b, hi_b - med_b],
		fmt="o",
		color="tab:orange",
		capsize=3,
		label=run_b.name,
	)

	truth_vals = []
	for label in labels:
		truth_vals.append(run_a.truth_map.get(label, run_b.truth_map.get(label, np.nan)))
	truth_arr = np.asarray(truth_vals, dtype=float)
	if np.any(np.isfinite(truth_arr)):
		ax.scatter(x, truth_arr, color="tab:red", s=35, marker="x", label="truth")

	ax.set_xticks(x)
	short_labels = [label.replace(" [Hz]", "").replace(" [1/s]", "").replace(" [rad]", "").replace("source ", "S").replace(" ", "\n") for label in labels]
	ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
	ax.set_title("Posterior medians and 90% intervals")
	ax.legend()
	_ = save_figure(fig, output_dir, "posterior_interval_compare")


def plot_snr(run_a: RunPosterior, run_b: RunPosterior, output_dir: Path) -> None:
	labels = ["source 1", "source 2"]
	if not run_a.snr_map and not run_b.snr_map:
		print("\nSNR comparison skipped: no SNR keys found in either posterior file.")
		return

	y_a = [run_a.snr_map.get(label, np.nan) for label in labels]
	y_b = [run_b.snr_map.get(label, np.nan) for label in labels]

	print("\nSNR comparison")
	for label, va, vb in zip(labels, y_a, y_b, strict=True):
		print(f"  {label:8s} {run_a.name}={va:.3f}  {run_b.name}={vb:.3f}")

	x = np.arange(len(labels))
	fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
	width = 0.35
	ax.bar(x - width / 2, y_a, width=width, color="tab:blue", alpha=0.8, label=run_a.name)
	ax.bar(x + width / 2, y_b, width=width, color="tab:orange", alpha=0.8, label=run_b.name)
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.set_ylabel("SNR")
	ax.set_title("Source-wise SNR comparison")
	ax.legend()
	_ = save_figure(fig, output_dir, "snr_compare")


def plot_corner_per_source(
	run_a: RunPosterior,
	run_b: RunPosterior,
	labels: list[str],
	output_dir: Path,
) -> None:
	if corner is None:
		print("corner package not installed; skipping corner plots.")
		return

	# For each source (4 params per source): [f, fdot, A, phi]
	for source_idx, source_name in enumerate(["source_1", "source_2"]):
		param_indices = [4 * source_idx, 4 * source_idx + 1, 4 * source_idx + 2, 4 * source_idx + 3]
		source_labels = [labels[i].replace(f"source {source_idx + 1} ", "") for i in param_indices]

		samples_a = run_a.samples[:, param_indices]
		samples_b = run_b.samples[:, param_indices]

		# Compute truth vector for this source
		truth = []
		for idx in param_indices:
			label = labels[idx]
			truth.append(run_a.truth_map.get(label, run_b.truth_map.get(label, np.nan)))
		truth_arr = np.asarray(truth)

		# Create corner plot with run_a
		fig = corner.corner(
			samples_a,
			labels=source_labels,
			truths=truth_arr if np.any(np.isfinite(truth_arr)) else None,
			color="tab:blue",
			alpha=0.5,
			plot_datapoints=False,
			smooth=1.0,
		)

		# Overlay run_b on the same figure
		corner.corner(
			samples_b,
			fig=fig,
			labels=source_labels,
			truths=None,
			color="tab:orange",
			alpha=0.5,
			plot_datapoints=False,
			smooth=1.0,
		)

		# Add legend
		axes = np.asarray(fig.axes).reshape((len(source_labels), len(source_labels)))
		ax_legend = axes[0, -1]
		from matplotlib.patches import Patch

		legend_elements = [
			Patch(facecolor="tab:blue", alpha=0.5, label=run_a.name),
			Patch(facecolor="tab:orange", alpha=0.5, label=run_b.name),
		]
		if np.any(np.isfinite(truth_arr)):
			legend_elements.append(plt.Line2D([0], [0], color="tab:red", ls="--", label="truth"))
		ax_legend.legend(handles=legend_elements, loc="upper left", fontsize=8)

		_ = save_figure(fig, output_dir, f"corner_{source_name}")


def align_common_labels(run_a: RunPosterior, run_b: RunPosterior) -> tuple[RunPosterior, RunPosterior, list[str]]:
	common = [label for label in run_a.labels if label in run_b.labels]
	if not common:
		raise ValueError("No shared parameter labels across posterior files.")

	idx_a = [run_a.labels.index(label) for label in common]
	idx_b = [run_b.labels.index(label) for label in common]

	aligned_a = RunPosterior(
		name=run_a.name,
		path=run_a.path,
		samples=run_a.samples[:, idx_a],
		labels=common,
		truth_map=run_a.truth_map,
		snr_map=run_a.snr_map,
	)
	aligned_b = RunPosterior(
		name=run_b.name,
		path=run_b.path,
		samples=run_b.samples[:, idx_b],
		labels=common,
		truth_map=run_b.truth_map,
		snr_map=run_b.snr_map,
	)
	return aligned_a, aligned_b, common


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--run-a",
		type=Path,
		default=DEFAULT_WDM_PATH,
		help="Path to first posterior NPZ (default: WDM posterior).",
	)
	parser.add_argument(
		"--run-b",
		type=Path,
		default=DEFAULT_FREQ_PATH,
		help="Path to second posterior NPZ (default: freq posterior).",
	)
	parser.add_argument("--name-a", type=str, default="WDM")
	parser.add_argument("--name-b", type=str, default="Frequency")
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	return parser


def main() -> None:
	args = build_parser().parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)

	run_a = load_run(args.run_a, args.name_a)
	run_b = load_run(args.run_b, args.name_b)
	run_a, run_b, labels = align_common_labels(run_a, run_b)

	summarize_run(run_a, labels)
	summarize_run(run_b, labels)
	compare_summary(run_a, run_b, labels)

	plot_marginals(run_a, run_b, labels, args.output_dir)
	plot_intervals(run_a, run_b, labels, args.output_dir)
	plot_snr(run_a, run_b, args.output_dir)
	plot_corner_per_source(run_a, run_b, labels, args.output_dir)

	print("\nSaved comparison figures:")
	print(f"  {args.output_dir / 'posterior_marginals_compare.png'}")
	print(f"  {args.output_dir / 'posterior_interval_compare.png'}")
	if (args.output_dir / "snr_compare.png").exists():
		print(f"  {args.output_dir / 'snr_compare.png'}")
	if (args.output_dir / "corner_source_1.png").exists():
		print(f"  {args.output_dir / 'corner_source_1.png'}")
	if (args.output_dir / "corner_source_2.png").exists():
		print(f"  {args.output_dir / 'corner_source_2.png'}")


if __name__ == "__main__":
	main()