import os
from typing import List

import torch
from aitemplate.utils.import_path import import_parent
from cog import BasePredictor, Input, Path
from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       LMSDiscreteScheduler, PNDMScheduler)

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

import_parent(filepath=__file__, level=1)

from src.pipeline_stable_diffusion_ait_cog import StableDiffusionAITSlimPipeline

# Configure on compilation
SD_DIR = "./weights_sd21/diffusers-pipeline/"

# dims to compiled AIT model dir:
FIVE_TWELVE = "512x512"
SEVEN_SIXTY_EIGHT = "768x768"
DIM_TO_DIR = {
    FIVE_TWELVE: "./weights_sd21_512",
    SEVEN_SIXTY_EIGHT: "./weights_sd21_768"
}

DIMS = sorted(list(DIM_TO_DIR.keys()))

SAFETY_CACHE = "weights"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"


class Predictor(BasePredictor):
    def _setup_sd_15(self, model_dir):
        pipe = StableDiffusionAITSlimPipeline.from_pretrained(
            SD_DIR,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(
                SD_DIR, subfolder="scheduler"
            ),
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe.init_modules(model_dir)
        pipe.to("cuda")
        return pipe
    
    def _setup_sd_21(self, model_dir):
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=SAFETY_CACHE,
        )
        pipe = StableDiffusionAITSlimPipeline.from_pretrained(
            SD_DIR,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(
                SD_DIR, subfolder="scheduler"
            ),
            safety_checker=safety_checker,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe.init_modules(model_dir)
        pipe.to("cuda") 
        return pipe

    def setup(self):
        self.pipes = {}
        for dim, dir in DIM_TO_DIR.items():
            self.pipes[dim] = self._setup_sd_21(dir)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a vision of paradise. unreal engine",
        ),
        image_dimensions: str = Input(
            description=f"pixel dimensions of output image",
            default=SEVEN_SIXTY_EIGHT,
            choices=DIMS
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        pipe = self.pipes[image_dimensions]

        pipe.scheduler = make_scheduler(scheduler)

        generator = torch.Generator("cuda").manual_seed(seed)

        with torch.autocast("cuda"):
            outputs = []
            for _ in range(num_outputs):
                output = pipe(
                    prompt if prompt is not None else None,
                    negative_prompt=negative_prompt
                    if negative_prompt is not None
                    else None,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                )
                outputs.append(output)

        output_paths = []
        for i, sample in enumerate(outputs):
            if sample.nsfw_content_detected[0]:
                print("nsfw detected")
                continue

            output_path = f"out-{i}.png"
            sample.images[0].save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name):
    return {
        "PNDM": PNDMScheduler.from_pretrained(SD_DIR, subfolder="scheduler"),
        "KLMS": LMSDiscreteScheduler.from_pretrained(SD_DIR, subfolder="scheduler"),
        "DDIM": DDIMScheduler.from_pretrained(SD_DIR, subfolder="scheduler"),
        "K_EULER": EulerDiscreteScheduler.from_pretrained(
            SD_DIR, subfolder="scheduler"
        ),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_pretrained(
            SD_DIR, subfolder="scheduler"
        ),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_pretrained(
            SD_DIR, subfolder="scheduler"
        ),
    }[name]
