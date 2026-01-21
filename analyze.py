# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import operator
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _extract_latency_metric(latency_data: Dict[str, Any], metric_name: str, convert_to_ms: bool = False) -> float | None:
    """Helper to extract a metric's mean value from latency data."""
    metric_data = latency_data.get(metric_name)
    if isinstance(metric_data, dict):
        mean_val = metric_data.get("mean")
        if isinstance(mean_val, (int, float)):
            return mean_val * 1000 if convert_to_ms else mean_val
    return None

def _extract_latency_metric_p90(latency_data: Dict[str, Any], metric_name: str, convert_to_ms: bool = False) -> float | None:
    """Helper to extract a metric's mean value from latency data."""
    metric_data = latency_data.get(metric_name)
    if isinstance(metric_data, dict):
        mean_val = metric_data.get("p90")
        if isinstance(mean_val, (int, float)):
            return mean_val * 1000 if convert_to_ms else mean_val
    return None


def _extract_throughput_metric(throughput_data: Dict[str, Any], metric_name: str) -> float | None:
    """Helper to extract a throughput metric's value."""
    metric_value = throughput_data.get(metric_name)
    if isinstance(metric_value, (int, float)):
        return float(metric_value)
    return None


def _generate_plot(charts_to_generate: List[Dict[str, Any]], suptitle: str, output_path: Path) -> None:
    """Generates and saves a plot with multiple subplots."""
    import matplotlib.pyplot as plt

    if not charts_to_generate:
        logger.warning(f"No data available to generate chart: {output_path.name}")
        return

    num_charts = len(charts_to_generate)
    fig, axes = plt.subplots(1, num_charts, figsize=(7 * num_charts, 6), squeeze=False)
    fig.suptitle(suptitle, fontsize=16)

    for i, chart_info in enumerate(charts_to_generate):
        ax = axes[0, i]
        for dataset in chart_info["datasets"]:
            data = dataset["data"]
            if not data:
                continue
            label = dataset["name"]
            x_values = [x[0] for x in data]
            y_values = [x[1] for x in data]
            ax.plot(x_values, y_values, marker="o", linestyle="-", label=label)

        ax.set_title(chart_info["title"])
        ax.set_xlabel(chart_info.get("xlabel", "QPS (requested rate)"))
        ax.set_ylabel(chart_info["ylabel"])
        ax.grid(True)
        ax.legend()

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(output_path)
    logger.info(f"Chart saved to {output_path}")
    plt.close(fig)


def analyze_reports(report_dirs: List[str], output_dir: str) -> None:
    """
    Analyzes performance reports to generate charts.

    Args:
        report_dirs: A list of directories containing the report files.
        output_dir: The directory to save the generated charts.
    """
    try:
        # Check for matplotlib and provide a helpful error message if it's not installed.
        import matplotlib  # noqa: F401
    except ImportError:
        logger.error(
            "matplotlib is not installed. Please install it to use the --analyze feature.\n"
            "You can install it via 'pip install .[analysis]'"
        )
        return

    logger.info(f"Analyzing reports in {report_dirs}")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Latency data
    qps_vs_ttft: Dict[str, List[Tuple[float, float]]] = {}
    qps_vs_ttftp90: Dict[str, List[Tuple[float, float]]] = {}
    qps_vs_ntpot: Dict[str, List[Tuple[float, float]]] = {}
    qps_vs_itl: Dict[str, List[Tuple[float, float]]] = {}
    # Throughput data
    qps_vs_itps: Dict[str, List[Tuple[float, float]]] = {}
    qps_vs_otps: Dict[str, List[Tuple[float, float]]] = {}
    qps_vs_ttps: Dict[str, List[Tuple[float, float]]] = {}
    # Throughput vs Latency data
    ttft_vs_otps: Dict[str, List[Tuple[float, float]]] = {}
    ntpot_vs_otps: Dict[str, List[Tuple[float, float]]] = {}
    itl_vs_otps: Dict[str, List[Tuple[float, float]]] = {}
    # Success ratio data
    qps_vs_success_ratio: Dict[str, List[Tuple[float, float]]] = {}

    for report_dir in report_dirs:
        report_name = Path(report_dir).name
        qps_vs_ttft[report_name] = []
        qps_vs_ttftp90[report_name] = []
        qps_vs_ntpot[report_name] = []
        qps_vs_itl[report_name] = []
        qps_vs_itps[report_name] = []
        qps_vs_otps[report_name] = []
        qps_vs_ttps[report_name] = []
        ttft_vs_otps[report_name] = []
        ntpot_vs_otps[report_name] = []
        itl_vs_otps[report_name] = []
        qps_vs_success_ratio[report_name] = []

        report_path = Path(report_dir)
        stage_files = list(report_path.glob("stage_*_lifecycle_metrics.json"))

        if not stage_files:
            logger.warning(f"No stage lifecycle metrics files found in {report_dir}")
            continue

        for stage_file in stage_files:
            # warmup skip
            # if "stage_0" in stage_file.name:
                # continue
            try:
                with open(stage_file, "r") as f:
                    report_data = json.load(f)

                # Get QPS from report file
                load_summary = report_data.get("load_summary", {})
                qps = load_summary.get("achieved_rate")
                if qps is None:
                    logger.warning(f"Could not find requested_rate in {stage_file.name}. Skipping.")
                    continue

                success_data = report_data.get("successes", {})

                # Calculate success ratio
                num_requests = load_summary.get("count")
                num_successes = success_data.get("count", 0)

                if num_requests is not None and num_requests > 0:
                    success_ratio = (num_successes / num_requests) * 100
                    qps_vs_success_ratio[report_name].append((qps, success_ratio))
                else:
                    logger.warning(
                        f"Could not determine success ratio from {stage_file.name} as num_requests is missing or zero."
                    )

                if not success_data:
                    logger.warning(f"No success data in {stage_file.name}. Skipping.")
                    continue

                # Extract latency metrics if they exist
                ttft, ntpot, itl = None, None, None
                latency_data = success_data.get("latency", {})
                if latency_data:
                    ttft = _extract_latency_metric(latency_data, "time_to_first_token", convert_to_ms=True)
                    if ttft is not None:
                        qps_vs_ttft[report_name].append((qps, ttft))
                    ttftp90 = _extract_latency_metric_p90(latency_data, "time_to_first_token", convert_to_ms=True)
                    if ttftp90 is not None:
                        qps_vs_ttftp90[report_name].append((qps, ttftp90))

                    ntpot = _extract_latency_metric(
                        latency_data, "normalized_time_per_output_token", convert_to_ms=True
                    )
                    if ntpot is not None:
                        qps_vs_ntpot[report_name].append((qps, ntpot))

                    itl = _extract_latency_metric(latency_data, "inter_token_latency", convert_to_ms=True)
                    if itl is not None:
                        qps_vs_itl[report_name].append((qps, itl))

                # Extract throughput metrics if they exist
                otps = None
                throughput_data = success_data.get("throughput", {})
                if throughput_data:
                    itps = _extract_throughput_metric(throughput_data, "input_tokens_per_sec")
                    if itps is not None:
                        qps_vs_itps[report_name].append((qps, itps))

                    otps = _extract_throughput_metric(throughput_data, "output_tokens_per_sec")
                    if otps is not None:
                        qps_vs_otps[report_name].append((qps, otps))

                    ttps = _extract_throughput_metric(throughput_data, "total_tokens_per_sec")
                    if ttps is not None:
                        qps_vs_ttps[report_name].append((qps, ttps))

                # Populate latency vs throughput data
                if otps is not None:
                    if ttft is not None:
                        ttft_vs_otps[report_name].append((ttft, otps))
                    if ntpot is not None:
                        ntpot_vs_otps[report_name].append((ntpot, otps))
                    if itl is not None:
                        itl_vs_otps[report_name].append((itl, otps))

            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {stage_file.name}")
                continue
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing {stage_file.name}: {e}")
                continue
    

    cared_qps = {3.0, 10.0, 15.0, 20.0, 30.0, 35.0, 43.0, 55.0, 60.0}
    for key in qps_vs_ttps:
        print(f"For {key}")
        qps_vs_ttps[key].sort()
        print(f"QPS {qps_vs_ttps[key]}")
        # numbers = []
        # for tup in qps_vs_ttftp90[key]:
        #     if tup[0] in cared_qps:
        #         numbers.append(tup)
        # print(f"cared qps is {numbers}")
    # --- Generate Latency Plot ---
    latency_charts_to_generate = []
    if any(qps_vs_ttft.values()):
        datasets = []
        for report_name, data in qps_vs_ttft.items():
            datasets.append({"name": report_name, "data": sorted(data, key=operator.itemgetter(0))})
        latency_charts_to_generate.append(
            {
                "title": "Time to First Token vs. QPS",
                "ylabel": "Mean TTFT (ms)",
                "datasets": datasets,
            }
        )
    if any(qps_vs_ttftp90.values()):
        datasets = []
        for report_name, data in qps_vs_ttftp90.items():
            datasets.append({"name": report_name, "data": sorted(data, key=operator.itemgetter(0))})
        latency_charts_to_generate.append(
            {
                "title": "Time to First Token P90 vs. QPS",
                "ylabel": "P90 TTFT (ms)",
                "datasets": datasets,
            }
        )
        
    if any(qps_vs_ntpot.values()):
        datasets = []
        for report_name, data in qps_vs_ntpot.items():
            datasets.append({"name": report_name, "data": sorted(data, key=operator.itemgetter(0))})
        latency_charts_to_generate.append(
            {
                "title": "Norm. Time per Output Token vs. QPS",
                "ylabel": "Mean Norm. Time (ms/token)",
                "datasets": datasets,
            }
        )
    if any(qps_vs_itl.values()):
        datasets = []
        for report_name, data in qps_vs_itl.items():
            datasets.append({"name": report_name, "data": sorted(data, key=operator.itemgetter(0))})
        latency_charts_to_generate.append(
            {
                "title": "Inter-Token Latency vs. QPS",
                "ylabel": "Mean ITL (ms)",
                "datasets": datasets,
            }
        )

    _generate_plot(
        latency_charts_to_generate,
        "Latency vs Request Rate",
        output_path / "latency_vs_qps.png",
    )

    # --- Generate Throughput Plot ---
    throughput_charts_to_generate = []
    if any(qps_vs_itps.values()):
        datasets = []
        for report_name, data in qps_vs_itps.items():
            datasets.append({"name": report_name, "data": sorted(data, key=operator.itemgetter(0))})
        throughput_charts_to_generate.append(
            {
                "title": "Input Tokens/sec vs. QPS",
                "ylabel": "Tokens/sec",
                "datasets": datasets,
            }
        )
    if any(qps_vs_otps.values()):
        datasets = []
        for report_name, data in qps_vs_otps.items():
            datasets.append({"name": report_name, "data": sorted(data, key=operator.itemgetter(0))})
        throughput_charts_to_generate.append(
            {
                "title": "Output Tokens/sec vs. QPS",
                "ylabel": "Tokens/sec",
                "datasets": datasets,
            }
        )
    if any(qps_vs_ttps.values()):
        datasets = []
        for report_name, data in qps_vs_ttps.items():
            datasets.append({"name": report_name, "data": sorted(data, key=operator.itemgetter(0))})
        throughput_charts_to_generate.append(
            {
                "title": "Total Tokens/sec vs. QPS",
                "ylabel": "Tokens/sec",
                "datasets": datasets,
            }
        )

    _generate_plot(
        throughput_charts_to_generate,
        "Throughput vs Request Rate",
        output_path / "throughput_vs_qps.png",
    )

    # --- Generate Throughput vs Latency Curve Plot ---
    throughput_latency_charts_to_generate = []
    if any(ntpot_vs_otps.values()):
        datasets = []
        for report_name, data in ntpot_vs_otps.items():
            datasets.append({"name": report_name, "data": sorted(data, key=operator.itemgetter(0))})
        throughput_latency_charts_to_generate.append(
            {
                "title": "Throughput vs. Norm. Time per Output Token",
                "xlabel": "Mean Norm. Time (ms/token)",
                "ylabel": "Output Tokens/sec",
                "datasets": datasets,
            }
        )
    if any(ttft_vs_otps.values()):
        datasets = []
        for report_name, data in ttft_vs_otps.items():
            datasets.append({"name": report_name, "data": sorted(data, key=operator.itemgetter(0))})
        throughput_latency_charts_to_generate.append(
            {
                "title": "Throughput vs. Time to First Token",
                "xlabel": "Mean TTFT (ms)",
                "ylabel": "Output Tokens/sec",
                "datasets": datasets,
            }
        )
    if any(itl_vs_otps.values()):
        datasets = []
        for report_name, data in itl_vs_otps.items():
            datasets.append({"name": report_name, "data": sorted(data, key=operator.itemgetter(0))})
        throughput_latency_charts_to_generate.append(
            {
                "title": "Throughput vs. Inter-Token Latency",
                "xlabel": "Mean ITL (ms)",
                "ylabel": "Output Tokens/sec",
                "datasets": datasets,
            }
        )

    _generate_plot(
        throughput_latency_charts_to_generate,
        "Latency vs Throughput",
        output_path / "throughput_vs_latency.png",
    )

    # --- Generate Success Ratio Plot ---
    success_ratio_charts_to_generate = []
    if any(qps_vs_success_ratio.values()):
        datasets = []
        for report_name, data in qps_vs_success_ratio.items():
            datasets.append({"name": report_name, "data": sorted(data, key=operator.itemgetter(0))})
        success_ratio_charts_to_generate.append(
            {
                "title": "Success Ratio vs. QPS",
                "ylabel": "Success Ratio (%)",
                "datasets": datasets,
            }
        )

    _generate_plot(
        success_ratio_charts_to_generate,
        "Success Ratio vs Request Rate",
        output_path / "success_ratio_vs_qps.png",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Analyze and compare performance reports.")
    parser.add_argument(
        "report_dirs",
        nargs="+",
        help="One or more report directories to analyze and compare.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="The directory to save the generated charts. Defaults to the current directory.",
    )
    args = parser.parse_args()
    parent_folder = Path(args.report_dirs[0])
    subfolders = [str(p) for p in parent_folder.iterdir() if p.is_dir()]
    print("analyzing subfolders:", subfolders)
    analyze_reports(subfolders, parent_folder)