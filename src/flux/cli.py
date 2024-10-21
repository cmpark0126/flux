import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
import torch._inductor.config as inductor_config
from einops import rearrange
from fire import Fire
from PIL import Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85
TORCH_COMPILE = os.getenv("TORCH_COMPILE", "0") == "1"

@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

class CudaTimer:
    """
    A static context manager class for measuring execution time of PyTorch code
    using CUDA events. It synchronizes GPU operations to ensure accurate time measurements.
    """

    def __init__(self, name="", precision=5, display=False):
        self.name = name
        self.precision = precision
        self.display = display

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, *exc):
        self.end_event.record()
        torch.cuda.synchronize()
        # Convert from ms to s
        self.elapsed_time = self.start_event.elapsed_time(self.end_event) * 1e-3

        if self.display:
            print(f"{self.name}: {self.elapsed_time:.{self.precision}f} s")

    def get_elapsed_time(self):
        """Returns the elapsed time in microseconds."""
        return self.elapsed_time


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting seed to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options


@torch.inference_mode()
def main(
    name: str = "flux-schnell",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    prompt: str = "A tree",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 3.5,
    output_dir: str = "output",
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """
    device = torch.device("cuda")

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # init all components
    t5 = load_t5(device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(device=device)
    model = load_flow_model(name, device=device)
    ae = load_ae(name, device=device)

    if TORCH_COMPILE:
        # torch._inductor.list_options()
        inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
        inductor_config.benchmark_kernel = True
        inductor_config.cuda.compile_opt_level = "-O3"  # default: "-O1"
        inductor_config.cuda.use_fast_math = True
        model = torch.compile(model,
                    fullgraph=True,
                    backend="inductor",
                    mode="max-autotune",
                    )

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if loop:
        opts = parse_prompt(opts)
    # warmup
    for _ in range(3):
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        opts.seed = None

        inp = prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))
        # denoise initial noise
        # with torch.no_grad():
        #     with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'), with_stack=True) as prof:
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)
        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        fn = output_name.format(idx=idx)
        print(f"Saving {fn}")
        # bring into PIL format and save
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        img.save(fn, exif=Image.Exif(), quality=95, subsampling=0)
        idx += 1

    with CudaTimer(display=False) as timer:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        opts.seed = None
        inp = prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))
        # denoise initial noise
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)
        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

    print(f"Inference time: {timer.get_elapsed_time()}")


def app():
    Fire(main)


if __name__ == "__main__":
    app()
