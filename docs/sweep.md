# Sweep Configuration and Behavior

The `sweep` feature in `inference-perf` automates the process of finding a target model server's maximum request rate (saturation point) and then automatically generates a series of load stages to benchmark the server up to that saturation point.

## Overview

Rather than manually guessing and specifying request rates (`rate`) for load stages, you can configure `sweep` to let `inference-perf` discover the server's capabilities. It does this by running an initial short burst of high traffic to determine the "burn down rate" (the rate at which the server completes requests), estimating the saturation point from this data, and finally generating `num_stages` standard load stages that progress up to the discovered limit.

## Configuration Parameters

Sweep is configured within the `load.sweep` section of the YAML configuration file.

```yaml
load:
  type: constant  # Must be 'constant' or 'poisson'
  sweep:
    type: linear                # Progression type: 'linear' or 'geometric'
    num_requests: 2000          # Total number of requests for the preprocessing burst
    timeout: 60                 # Max duration in seconds for the preprocessing burst
    num_stages: 5               # Number of load stages to generate after finding saturation
    stage_duration: 180         # Duration in seconds for each generated stage
    saturation_percentile: 95   # Percentile of sampled completion rates to use as the saturation point
```

- **`type`**: The progression strategy for the generated stages. `linear` will generate evenly spaced request rates from 1.0 QPS up to the saturation point. `geometric` will generate rates clustered closer to the saturation point.
- **`num_requests`**: The number of requests injected during the preprocessing stage to saturate the server.
- **`timeout`**: The maximum time (in seconds) the preprocessing stage will run before it cancels remaining requests and calculates saturation based on completed requests.
- **`num_stages`**: The number of standard load stages to automatically generate.
- **`stage_duration`**: The length of time (in seconds) that each generated load stage will run.
- **`saturation_percentile`**: The percentile of the sampled "burn down rates" to use as the estimated saturation point. Higher values represent peak throughput, while lower values are more conservative.

## How Sweep Works

When `sweep` is configured, `inference-perf` performs the following steps in `load_generator.py`:

1.  **Preprocessing Burst:** A preliminary load stage (identified internally as `stage_id = -1`) is executed. It sends `num_requests` divided by a 5-second duration, capped at `timeout`.
2.  **Sampling:** While the burst is running, a background aggregator task continually samples the number of active requests in flight every 0.5 seconds.
3.  **Calculation:** The tool calculates the "burn down rate" (requests completed per second) between each sample.
4.  **Saturation Estimation:** It takes the configured `saturation_percentile` (e.g., 95th percentile) of these burn down rates to determine the estimated maximum QPS the server can sustain.
5.  **Stage Generation:** Finally, it generates `num_stages` new load stages using either a `linear` or `geometric` distribution, scaling up to the estimated saturation point. Each generated stage will run for `stage_duration` seconds.

## Interaction with Other Features

Sweep is fully integrated with other `inference-perf` features:

-   **Load Types:** Sweep is **only supported** when `load.type` is set to `constant` or `poisson`. If you configure `sweep` with the `concurrent` load type, configuration validation will fail.
-   **MultiLoRA Traffic Splitting:** Sweep fully respects the `lora_traffic_split` configuration. During both the preprocessing burst and the generated load stages, requests will be assigned LoRA adapters according to your defined weights. This ensures that the measured saturation point accurately reflects the overhead of adapter context switching.
-   **Circuit Breakers:** Circuit breakers are actively monitored during sweep operations. If a circuit breaker trips (opens) during the preprocessing burst or any of the automatically generated stages, `inference-perf` will log a warning and exit the stage early to prevent overwhelming the server.
