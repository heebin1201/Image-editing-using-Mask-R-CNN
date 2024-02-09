import os
import hashlib

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import imghdr
import io
import logging
import multiprocessing
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger

from lama_cleaner.const import SD15_MODELS
from lama_cleaner.file_manager import FileManager
from lama_cleaner.model.utils import torch_gc
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.plugins import (
    InteractiveSeg,
    RemoveBG,
    RealESRGANUpscaler,
    MakeGIF,
    GFPGANPlugin,
    RestoreFormerPlugin,
    AnimeSeg,
)
from lama_cleaner.schema import Config

from lama_cleaner.helper import (
    load_img,
    numpy_to_bytes,
    resize_max_size,
    pil_to_bytes,
)
global image_quality
image_quality = 95

def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w

def save_image(request:dict ,output_dir:str, image_quality: int = 95):
    if output_dir is None:
        return "--output-dir is None", 500

    input = request
    filename = input["filename"]
    origin_image_bytes = input["image"].read()  # RGB
    ext = get_image_ext(origin_image_bytes)
    image, alpha_channel, exif_infos = load_img(origin_image_bytes, return_exif=True)
    save_path = os.path.join(output_dir, filename)

    if alpha_channel is not None:
        if alpha_channel.shape[:2] != image.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(image.shape[1], image.shape[0])
            )
        image = np.concatenate((image, alpha_channel[:, :, np.newaxis]), axis=-1)

    pil_image = Image.fromarray(image)

    img_bytes = pil_to_bytes(
        pil_image,
        ext,
        quality=image_quality,
        exif_infos=exif_infos,
    )
    with open(save_path, "wb") as fw:
        fw.write(img_bytes)


def process(request: dict, model: ModelManager = None):
    """inpainting 적용 input 과 마스크를 parm 안에 dict 형태로 전달할 것

    Args:
        request (dict): {"image": (bytes), "mask": (bytes), "paintByExampleImage": (bytes | None)}

    Returns:
        None: None
    """
    input = request.files
    # RGB
    origin_image_bytes = input["image"].read()
    image, alpha_channel, exif_infos = load_img(origin_image_bytes, return_exif=True)

    mask, _ = load_img(input["mask"].read(), gray=True)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    if image.shape[:2] != mask.shape[:2]:
        return (
            f"Mask shape{mask.shape[:2]} not queal to Image shape{image.shape[:2]}",
            400,
        )

    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC

    form = request.form
    size_limit = max(image.shape)

    if "paintByExampleImage" in input:
        paint_by_example_example_image, _ = load_img(
            input["paintByExampleImage"].read()
        )
        paint_by_example_example_image = Image.fromarray(paint_by_example_example_image)
    else:
        paint_by_example_example_image = None

    config = Config(
        ldm_steps=form["ldmSteps"],
        ldm_sampler=form["ldmSampler"],
        hd_strategy=form["hdStrategy"],
        zits_wireframe=form["zitsWireframe"],
        hd_strategy_crop_margin=form["hdStrategyCropMargin"],
        hd_strategy_crop_trigger_size=form["hdStrategyCropTrigerSize"],
        hd_strategy_resize_limit=form["hdStrategyResizeLimit"],
        prompt=form["prompt"],
        negative_prompt=form["negativePrompt"],
        use_croper=form["useCroper"],
        croper_x=form["croperX"],
        croper_y=form["croperY"],
        croper_height=form["croperHeight"],
        croper_width=form["croperWidth"],
        sd_scale=form["sdScale"],
        sd_mask_blur=form["sdMaskBlur"],
        sd_strength=form["sdStrength"],
        sd_steps=form["sdSteps"],
        sd_guidance_scale=form["sdGuidanceScale"],
        sd_sampler=form["sdSampler"],
        sd_seed=form["sdSeed"],
        sd_match_histograms=form["sdMatchHistograms"],
        cv2_flag=form["cv2Flag"],
        cv2_radius=form["cv2Radius"],
        paint_by_example_steps=form["paintByExampleSteps"],
        paint_by_example_guidance_scale=form["paintByExampleGuidanceScale"],
        paint_by_example_mask_blur=form["paintByExampleMaskBlur"],
        paint_by_example_seed=form["paintByExampleSeed"],
        paint_by_example_match_histograms=form["paintByExampleMatchHistograms"],
        paint_by_example_example_image=paint_by_example_example_image,
        p2p_steps=form["p2pSteps"],
        p2p_image_guidance_scale=form["p2pImageGuidanceScale"],
        p2p_guidance_scale=form["p2pGuidanceScale"],
        controlnet_conditioning_scale=form["controlnet_conditioning_scale"],
        controlnet_method=form["controlnet_method"],
    )

    if config.sd_seed == -1:
        config.sd_seed = random.randint(1, 999999999)
    if config.paint_by_example_seed == -1:
        config.paint_by_example_seed = random.randint(1, 999999999)

    logger.info(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)

    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

    start = time.time()
    try:
        res_np_img = model(image, mask, config)
    except RuntimeError as e:
        if "CUDA out of memory. " in str(e):
            # NOTE: the string may change?
            return "CUDA out of memory", 500
        else:
            logger.exception(e)
            return f"{str(e)}", 500
    finally:
        logger.info(f"process time: {(time.time() - start) * 1000}ms")
        torch_gc()

    res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
            )
        res_np_img = np.concatenate(
            (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )

    ext = get_image_ext(origin_image_bytes)

    bytes_io = io.BytesIO(
        pil_to_bytes(
            Image.fromarray(res_np_img),
            ext,
            quality=image_quality,
            exif_infos=exif_infos,
        )
    )
    
    return bytes_io
