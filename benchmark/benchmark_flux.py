import os
import time
from dataclasses import dataclass
from pathlib import Path
import json

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

# CLI와 동일한 기본 설정 유지
TORCH_COMPILE = os.getenv("TORCH_COMPILE", "0") == "1"

app = typer.Typer()
console = Console()


@dataclass
class BenchmarkOptions:
    """CLI의 SamplingOptions를 벤치마크용으로 확장"""

    name: str = "flux-schnell"
    width: int = 1360
    height: int = 768
    prompt: str = "a photo of a forest with mist swirling around the tree trunks"
    num_steps: int | None = None
    guidance: float = 3.5
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    save_results: bool = True
    output_dir: str = "benchmark_results"


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
        """CLI와 동일한 방식으로 모델 초기화"""
        console.print("Loading models...")
        self.t5 = load_t5(
            self.device, max_length=256 if self.options.name == "flux-schnell" else 512
        )
        self.clip = load_clip(self.device)
        self.model = load_flow_model(self.options.name, device=self.device)
        self.ae = load_ae(self.options.name, device=self.device)

        if TORCH_COMPILE:
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

    def _single_run(self, seed: int | None = None) -> float:
        """단일 실행에 대한 시간 측정"""
        rng = torch.Generator(device="cpu")
        if seed is None:
            seed = rng.seed()

        t0 = time.perf_counter()

        with torch.inference_mode():
            # CLI와 동일한 이미지 생성 프로세스
            x = get_noise(
                1,
                self.options.height,
                self.options.width,
                device=self.device,
                dtype=torch.bfloat16,
                seed=seed,
            )

            inp = prepare(self.t5, self.clip, x, prompt=self.options.prompt)
            assert self.options.num_steps is not None, "num_steps must be set"
            timesteps = get_schedule(
                self.options.num_steps,
                inp["img"].shape[1],
                shift=(self.options.name != "flux-schnell"),
            )

            x = denoise(
                self.model, **inp, timesteps=timesteps, guidance=self.options.guidance  # type: ignore
            )
            x = unpack(x.float(), self.options.height, self.options.width)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                x = self.ae.decode(x)

            torch.cuda.synchronize()

        t1 = time.perf_counter()
        return t1 - t0

    def run_benchmark(self):
        """벤치마크 실행"""
        # 워밍업
        console.print(
            f"\nRunning {self.options.warmup_iterations} warmup iterations..."
        )
        for _ in range(self.options.warmup_iterations):
            self._single_run()

        # 실제 벤치마크
        console.print(
            f"\nRunning {self.options.benchmark_iterations} benchmark iterations..."
        )
        for i in range(self.options.benchmark_iterations):
            elapsed_time = self._single_run()
            self.results.append(elapsed_time)
            console.print(f"Iteration {i+1}: {elapsed_time:.3f}s")

        # 결과 분석
        mean_time = sum(self.results) / len(self.results)
        std_time = torch.tensor(self.results).std().item()

        console.print("\n[bold green]Benchmark Results:")
        console.print(f"Mean inference time: {mean_time:.3f}s")
        console.print(f"Std deviation: {std_time:.3f}s")

        # 결과 저장
        if self.options.save_results:
            results_data = {
                "config": self.options.__dict__,
                "times": self.results,
                "mean_time": mean_time,
                "std_time": std_time,
            }

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            result_file = (
                Path(self.options.output_dir) / f"benchmark_results_{timestamp}.json"
            )

            with open(result_file, "w") as f:
                json.dump(results_data, f, indent=2)
            console.print(f"\nResults saved to {result_file}")


@app.command()
def run(
    name: str = "flux-schnell",
    width: int = 1360,
    height: int = 768,
    prompt: str = "a photo of a forest with mist swirling around the tree trunks",
    num_steps: int | None = None,
    guidance: float = 3.5,
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
    save_results: bool = True,
    output_dir: str = "benchmark_results",
):
    """FLUX 모델 벤치마크 실행"""
    options = BenchmarkOptions(
        name=name,
        width=width,
        height=height,
        prompt=prompt,
        num_steps=num_steps,
        guidance=guidance,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
        save_results=save_results,
        output_dir=output_dir,
    )

    benchmark = FluxBenchmark(options)
    benchmark.setup_model()
    benchmark.run_benchmark()


if __name__ == "__main__":
    app()
