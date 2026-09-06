import csv
import os
import subprocess
import threading
import time


class GPUMonitor:
    def __init__(self, csv_file="gpu_metrics.csv", interval=1):
        self.csv_file = csv_file
        self.interval = interval
        self.measurements = []
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None

    def query_gpu(self):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        for line in result.stdout.strip().splitlines():
            gpu_index, gpu_utilization, memory_used, memory_total = map(int, line.split(","))
            # Grafana can use Unix epoch timestamps in seconds.
            timestamp = time.time()

            measurement = {
                "timestamp": timestamp,
                "gpu_index": gpu_index,
                "gpu_utilization": gpu_utilization,
                "memory_used": memory_used,
                "memory_total": memory_total,
            }
            self.measurements.append(measurement)

    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                self.query_gpu()
            except Exception as exc:
                self.error = exc
                self.stop_event.set()
                break
            self.stop_event.wait(self.interval)

    def start(self, csv_file=None):
        self.measurements = []
        self.error = None
        self.stop_event.clear()

        if csv_file is not None:
            self.csv_file = csv_file

        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def write_gpu_monitoring_csv(self, start_time=None, end_time=None, output_path=None):
        if output_path is None:
            output_path = self.csv_file

        filtered_measurements = [
            measurement for measurement in self.measurements
            if (start_time is None or measurement["timestamp"] >= start_time)
            and (end_time is None or measurement["timestamp"] <= end_time)
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gpu_index", "gpu_utilization", "memory_used", "memory_total"])
            for measurement in filtered_measurements:
                writer.writerow([
                    measurement["timestamp"],
                    measurement["gpu_index"],
                    measurement["gpu_utilization"],
                    measurement["memory_used"],
                    measurement["memory_total"],
                ])

        return filtered_measurements

    def stop(self):
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
        if self.error is not None:
            raise RuntimeError("GPU monitoring failed") from self.error
        return self.measurements