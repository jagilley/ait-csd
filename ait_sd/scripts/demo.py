#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import click
import torch

from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.import_path import import_parent
from diffusers import EulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.pipeline_stable_diffusion_ait_cog import StableDiffusionAITSlimPipeline


@click.command()
@click.option(
    "--local-dir",
    default="./weights/diffusers-pipeline/",
    help="the local diffusers pipeline directory",
)
@click.option("--compile-dir", default="./weights", help="root directory for AITemplate compiled binaries")
@click.option("--width", default=None, help="Width of generated image")
@click.option("--height", default=None, help="Height of generated image")
@click.option("--prompt", default="A vision of paradise, Unreal Engine", help="prompt")
@click.option("--negative_prompt", default="blurry, low-res")
@click.option(
    "--benchmark", type=bool, default=False, help="run stable diffusion e2e benchmark"
)
def run(local_dir, compile_dir, width, height, prompt, negative_prompt, benchmark):
    MODEL_CACHE = "weights"
    SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        SAFETY_MODEL_ID,
        cache_dir=MODEL_CACHE,
    )

    pipe = StableDiffusionAITSlimPipeline.from_pretrained(
        local_dir,
        scheduler=EulerDiscreteScheduler.from_pretrained(
            local_dir, subfolder="scheduler"
        ),
        safety_checker=safety_checker,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe.init_modules(workdir=compile_dir)
    pipe.to("cuda")

    with torch.autocast("cuda"):
        image = pipe(prompt, height, width, negative_prompt=negative_prompt).images[0]
        if benchmark:
            t = benchmark_torch_function(10, pipe, prompt, height=height, width=width)
            print(f"sd e2e: {t} ms")

    image.save("example_ait.png")


if __name__ == "__main__":
    run()
