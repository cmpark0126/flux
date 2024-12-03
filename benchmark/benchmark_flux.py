import time
from dataclasses import dataclass
from pathlib import Path
import json
import os

import torch
import torch._inductor.config as inductor_config
import typer
from rich.console import Console

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)

app = typer.Typer()
console = Console()


@dataclass
class BenchmarkOptions:
    name: str = "flux-schnell"
    width: int = 1360
    height: int = 768
    prompt: str = "a photo of a forest with mist swirling around the tree trunks"
    torch_compile: bool = False
    use_custom_triton_kernels: bool = False
    attention_method: str = "torch_sdpa"
    offload: bool = False
    num_steps: int | None = None
    guidance: float = 3.5
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    save_results: bool = True
    output_dir: str = "benchmark_results"
    output_file_base_name: str = "benchmark_results"


class FluxBenchmark:
    def __init__(self, options: BenchmarkOptions):
        self.options = options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []

        # CLI와 동일한 방식으로 steps 설정
        if self.options.num_steps is None:
            self.options.num_steps = 4 if options.name == "flux-schnell" else 50

        # 디렉토리 생성
        Path(self.options.output_dir).mkdir(parents=True, exist_ok=True)

    def setup_model(self):
        console.print("Loading models...")
        model_device = "cpu" if self.options.offload else self.device

        self.t5 = load_t5(
            self.device, max_length=256 if self.options.name == "flux-schnell" else 512
        )
        self.clip = load_clip(self.device)
        self.model = load_flow_model(
            self.options.name,
            device=model_device,
            use_custom_triton_kernels=self.options.use_custom_triton_kernels,
            attention_method=self.options.attention_method,
        )
        self.ae = load_ae(self.options.name, device=model_device)

        if self.options.torch_compile:
            console.print("Compiling model...")
            inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
            inductor_config.benchmark_kernel = True
            inductor_config.cuda.compile_opt_level = "-O3"
            inductor_config.cuda.use_fast_math = True
            self.model = torch.compile(
                self.model,
                fullgraph=True,
                backend="inductor",
                mode="max-autotune",
            )

    @torch.inference_mode()
    def _single_run(self, seed: int | None = None) -> float:
        """Measure the single inference time"""
        rng = torch.Generator(device="cpu")
        if seed is None:
            seed = rng.seed()

        t0 = time.perf_counter()

        # Get noise and prepare input
        x = get_noise(
            1,
            self.options.height,
            self.options.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=seed,
        )

        # Offload handling for input preparation
        if self.options.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        inp = prepare(self.t5, self.clip, x, prompt=self.options.prompt)
        assert self.options.num_steps is not None, "num_steps must be set"
        timesteps = get_schedule(
            self.options.num_steps,
            inp["img"].shape[1],
            shift=(self.options.name != "flux-schnell"),
        )

        # Offload handling for model inference
        if self.options.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        x = denoise(
            self.model, **inp, timesteps=timesteps, guidance=self.options.guidance  # type: ignore
        )

        # Offload handling for decoding
        if self.options.offload:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            self.ae = self.ae.to(self.device)

        x = unpack(x.float(), self.options.height, self.options.width)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        torch.cuda.synchronize()

        t1 = time.perf_counter()
        return t1 - t0

    def run_benchmark(self):
        console.print(
            f"\nRunning {self.options.warmup_iterations} warmup iterations..."
        )
        for i in range(self.options.warmup_iterations):
            elapsed_time = self._single_run()
            console.print(f"Iteration {i+1}: {elapsed_time:.3f}s")

        # 실제 벤치마크
        console.print(
            f"\nRunning {self.options.benchmark_iterations} benchmark iterations..."
        )
        for i in range(self.options.benchmark_iterations):
            elapsed_time = self._single_run()
            self.results.append(elapsed_time)
            console.print(f"Iteration {i+1}: {elapsed_time:.3f}s")

        # 결과 분석
        p99_time = torch.tensor(self.results).quantile(0.99).item()
        p90_time = torch.tensor(self.results).quantile(0.90).item()
        p50_time = torch.tensor(self.results).median().item()
        mean_time = sum(self.results) / len(self.results)
        std_time = torch.tensor(self.results).std().item()

        console.print("\n[bold green]Benchmark Results:")
        console.print(f"99th percentile inference time: {p99_time:.3f}s")
        console.print(f"90th percentile inference time: {p90_time:.3f}s")
        console.print(f"50th inference time: {p50_time:.3f}s")
        console.print(f"Mean inference time: {mean_time:.3f}s")
        console.print(f"Std deviation: {std_time:.3f}s")

        # 결과 저장
        if self.options.save_results:
            results_data = {
                "config": self.options.__dict__,
                "times": self.results,
                "p99_time": p99_time,
                "p90_time": p90_time,
                "p50_time": p50_time,
                "mean_time": mean_time,
                "std_time": std_time,
            }

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            result_file = (
                Path(self.options.output_dir) / f"{self.options.output_file_base_name}_{timestamp}.json"
            )

            with open(result_file, "w") as f:
                json.dump(results_data, f, indent=2)
            console.print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    base_dir_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir_path, "benchmark_opts.json"), "r") as f:
        opts = json.load(f)

    sub_dir_name = f"{time.strftime('%Y%m%d-%H%M%S')}"
    for opt_name, opt in opts.items():
        console.print(
            f"\n[bold cyan]Running benchmark for configuration: {opt_name}[/bold cyan]"
        )
        console.print("=" * 80)

        benchmark_opt = BenchmarkOptions(**opt)
        benchmark_opt.output_dir = os.path.join(
            base_dir_path, benchmark_opt.output_dir, sub_dir_name
        )
        benchmark_opt.output_file_base_name = opt_name
        console.print(f"Options: {benchmark_opt}")

        try:
            benchmark = FluxBenchmark(benchmark_opt)
            benchmark.setup_model()
            benchmark.run_benchmark()
        except Exception as e:
            console.print(f"[bold red]Error running benchmark: {e}[/bold red]")

            if benchmark_opt.save_results:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                result_file = (
                    Path(benchmark_opt.output_dir)
                    / f"{benchmark_opt.output_file_base_name}_{timestamp}.json"
                )
                with open(result_file, "w") as f:
                    json.dump(
                        {"error": str(e), "config": benchmark_opt.__dict__}, f, indent=2
                    )
                console.print(f"\nResults saved to {result_file}")

            continue
