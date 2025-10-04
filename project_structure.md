# Inference Perf Project Structure

This document provides a high-level overview of the `inference-perf` project, its architecture, and its code structure.

## 1. Overview

`Inference Perf` is a performance benchmarking tool for Generative AI inference. It is designed to be highly scalable and supports benchmarking large-scale production deployments. The tool can generate various load patterns, simulate different real-world and synthetic datasets, and report key performance metrics for LLM inference.

## 2. Architecture

The architecture of `inference-perf` is modular, with distinct components for handling different stages of the benchmarking process.

![Architecture Diagram](docs/images/architecture.png)

The main components are:

*   **Dataset Preprocessor**: Prepares datasets for benchmarking. It can handle real-world datasets like ShareGPT as well as generate synthetic data with specific characteristics.
*   **Load Generator**: Generates traffic patterns to simulate different load scenarios, such as constant QPS, bursts, or gradually increasing load.
*   **Request Processor**: Formats and sends requests to the model server. It's designed to be model-server agnostic, supporting different APIs (e.g., OpenAI chat/completion) and protocols.
*   **Response Processor / Data Collector**: Processes the responses from the model server, collects performance data, and measures key metrics like latency and throughput.
*   **Report Generator / Metrics Exporter**: Generates summary reports in JSON format and can export metrics to monitoring systems like Prometheus.

## 3. Code Structure

The core logic of the application is located in the `inference_perf/` directory. Here is a breakdown of the key modules and their responsibilities, mapping them to the architectural components described above.

*   **`inference_perf/main.py`**: The main entry point for the `inference-perf` command-line tool. It parses arguments, loads the configuration, and orchestrates the benchmarking process.

*   **`inference_perf/config.py`**: Responsible for parsing and validating the main configuration file (`config.yml`).

### Component Implementations:

#### Dataset Preprocessor

*   **`inference_perf/datagen/`**: This package is responsible for generating the request data used in the benchmark.
    *   `base.py`: Defines the base class for data generators.
    *   `hf_sharegpt_datagen.py`: Generates data from the ShareGPT dataset.
    *   `synthetic_datagen.py`: Creates synthetic data based on specified input/output distributions.
    *   `random_datagen.py`: Generates random data.
    *   `shared_prefix_datagen.py`: Generates data with shared prefixes to test caching scenarios.
    *   `mock_datagen.py`: A mock generator for testing purposes.

#### Load Generator

*   **`inference_perf/loadgen/`**: This package contains the core load generation engine.
    *   `load_generator.py`: The main class that generates the specified load, sends requests, and collects results.
    *   `load_timer.py`: Manages the timing and rate of request generation (e.g., constant or Poisson distribution).

#### Request & Response Processing

*   **`inference_perf/apis/`**: Defines the API clients for different inference tasks.
    *   `base.py`: A base class for API clients.
    *   `chat.py`: Implements the client for chat completion APIs.
    *   `completion.py`: Implements the client for standard completion APIs.

*   **`inference_perf/client/`**: Contains various clients for interacting with external services.
    *   **`modelserver/`**: Clients for different model serving backends.
        *   `vllm_client.py`: A client specifically for vLLM.
    *   **`metricsclient/`**: Clients for collecting metrics from monitoring systems.
        *   `prometheus_client/`: A client for querying Prometheus.
    *   **`requestdatacollector/`**: Manages the collection of raw request/response data.
    *   **`filestorage/`**: Clients for storing results in different file storage systems (e.g., local, GCS, S3).

#### Report Generator & Analysis

*   **`inference_perf/reportgen/`**: This package is responsible for generating the final JSON reports from the collected data.

*   **`inference_perf/analysis/`**: Contains scripts to analyze the generated reports and create visualizations.
    *   `analyze.py`: The main script for performing analysis and generating plots (e.g., QPS vs. Latency).

## 4. Configuration

The behavior of the `inference-perf` tool is controlled by a central YAML configuration file (e.g., `config.yml`). This file defines all aspects of the benchmark, including:

*   **Data Generation**: The type of data to use (e.g., `shareGPT`, `synthetic`).
*   **Load Profile**: The load pattern to generate (e.g., `constant` QPS, stages).
*   **API and Server**: The target API type and model server endpoint.
*   **Metrics and Reporting**: The metrics to collect and the format of the reports.

For a detailed guide on all configuration options, refer to the [CONFIG.md](./CONFIG.md) file.

## 5. Usage and Examples

The tool is run from the command line, pointing to a configuration file:

```bash
inference-perf --config_file config.yml
```

The `examples/` directory contains various sample configuration files for different scenarios. The tool can be run locally, in a Docker container, or on a Kubernetes cluster. For more details, see the main [README.md](./README.md).
