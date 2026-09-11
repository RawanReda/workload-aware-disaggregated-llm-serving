# Workload-Aware Disaggregated LLM Serving

This repository benchmarks and evaluates workload-aware disaggregated prefill/decode serving for Qwen LLMs across different GPU split layouts.

The workflow starts from the GPU configuration in `gpu_config.yaml`. The main experiment driver, `vllm_runner.py`, starts the vLLM servers by running the bash script `disaggregated_prefill_example.sh`, then launches the benchmark process through `run_benchmarks.py`. During that benchmark run, `run_benchmarks.py` also starts `gpu_metrics_monitor.py` to collect GPU utilization and memory usage traces while the workload is executing.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the benchmark/test workflow from the project root:

```bash
python3 vllm_runner.py
```

The generated benchmark artifacts are stored under the `results/` directory.

## File overview

- `gpu_config.yaml` — defines the GPU partition layouts and tensor/pipeline parallel settings used for each experiment.
- `vllm_runner.py` — main experiment driver that loops through models and GPU split configurations and launches the benchmark workflow.
- `disaggregated_prefill_example.sh` — example shell script that starts the prefill and decode vLLM servers and sets up the disaggregated serving environment.
- `run_benchmarks.py` — runs the benchmark requests and writes result files for each GPU split scenario.
- `gpu_metrics_monitor.py` — monitors per-GPU utilization and memory usage during benchmarking.
- `proxy.py` — proxy component used to route or exchange requests and KV-cache-related traffic between prefill and decode stages.
- `requirements.txt` — Python package dependencies needed for the benchmark environment.
- `run_script_*.txt` — generated experiment logs showing the full command history and runtime output for each evaluated configuration.
- `results/` — contains benchmark outputs and collected metrics for each model and GPU split setup.
