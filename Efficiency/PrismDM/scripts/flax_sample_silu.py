import argparse
from contextlib import contextmanager
import json
import logging
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax
from flax.linen.linear import Array
from flax import linen as nn
from flax.serialization import to_bytes, from_bytes
from typing import Any, Callable, Dict, Optional, Tuple, Union
from flax.training import train_state
import pickle
import os
import optax
import tqdm
from tools import file_utils
from tools import cv2_utils
import time

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

from flax_ddpm.script_utils import get_args
from flax_ddpm.script_utils import get_diffusion_from_args
from flax_ddpm.script_utils import initialize_diffusion_from_args
from flax_ddpm.approximation_methods import *
from flax_ddpm.mish_function import mish

logger = logging.getLogger(__name__)
# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def get_sample_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str, default="./model_eval")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("-c", "--config", default="3pc.json")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--ddim-steps", type=int, choices=[2, 4, 5, 10, 20, 25, 50, 100], default=50)
    parser.add_argument("--backend", choices=["all", "cpu", "vanilla", "prismdm"], default="all")
    return parser

def measure_runtime(func):
    """
    A decorator to measure the runtime of a function.

    Parameters:
    func (callable): The function whose runtime you want to measure.

    Returns:
    callable: A wrapper function that will measure the runtime of 'func'.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Function '{func.__name__}' took {runtime:.4f} seconds to complete.")
        return result
    return wrapper

@measure_runtime
def run_inference_on_cpu(params,y,save_dir):
    output = sample_image(params, y)
    jax.block_until_ready(output)


@measure_runtime
def run_inference_on_spu(params, y,save_dir):
    output = ppd.device("SPU")(sample_image, copts=copts)(params, y)
    output = ppd.get(output)

def normalize_to_0_1(array):
    min_val = array.min()
    max_val = array.max()
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def sample_image(params, y=None):
    diffusion = get_diffusion_from_args(args)
    rng = jax.random.PRNGKey(0)
    sampler = diffusion.sample_ddim if args.sampler == 'ddim' else diffusion.sample
    sample_options = {'ddim_timesteps': args.ddim_steps} if args.sampler == 'ddim' else {}
    samples = diffusion.apply({'params': params}, batch_size=1, y=y, use_ema=False, rng_key=rng, method=sampler, **sample_options)
    image = normalize_to_0_1(samples[0])
    image = (image * 255).astype(jnp.uint8)  # Convert to [0, 255]
    return image

def initialize_diffusion(model_path):
    #assert model_path existence & load params
    assert model_path is not None
    assert os.path.exists(model_path), f"model file not exist: {model_path}"

    #initialize parameters
    rng = jax.random.PRNGKey(0)
    diffusion = initialize_diffusion_from_args(args)
    x = jax.random.normal(rng, (128,28,28,1))
    y = jnp.ones(128, dtype=jnp.int32)
    variables = diffusion.init(rngs=rng,rng=rng, x=x,y=y)
    optimizer = optax.adam(learning_rate=args.learning_rate)

    state = train_state.TrainState.create(
            apply_fn=diffusion.apply,
            params=variables['params'],
            tx=optimizer,
        )

    # load params
    pkl_file=pickle.load(open(model_path,"rb"))
    restored_state=flax.serialization.from_state_dict(target=state, state=pkl_file)

    #get_unlinear_list through single forward calculation
    unlinear_list = diffusion.apply({'params': restored_state.params}, batch_size=128, y=y, use_ema= False, rng_key=rng, method=diffusion.get_unlinear_list) #***Important*** here batch size is set to 128 cause x, y's size is 128
    unlinear_keys = list(unlinear_list.keys())

    return restored_state, unlinear_keys

def public_evaluation(restored_state, num_each_label, save_dir):
    params = restored_state.params
    y = jnp.ones(num_each_label, dtype=jnp.int32)
    run_inference_on_cpu(params,y,save_dir)

def secure_evaluation(restored_state, num_each_label, save_dir):
    params = restored_state.params
    y = jnp.ones(num_each_label, dtype=jnp.int32)
    y_secret = ppd.device("INPUT")(lambda x: x)(y)
    params_secret = ppd.device("INPUT")(lambda x: x)(params)
    run_inference_on_spu(params_secret, y_secret,save_dir)

def main(args: argparse.Namespace):
    #Global Parameters
    num_each_label = 1
    batch_size =1

    #Now unlinear_function_dict included in args
    #Load the Model Checkpoint
    model_path = args.model_path or "./checkpoints/ddpm-20000.msgpack"
    save_dir = args.save_dir
    assert model_path is not None
    assert os.path.exists(model_path), f"model file not exist: {model_path}"
    file_utils.mkdir(save_dir)

    #Diffusion Initialization & Unlinear_dict Generate
    restored_state, unlinear_keys=initialize_diffusion(model_path)
    #print(unlinear_keys)

    #Prepare Convert Json File STRING->CALLABLE FUNCTION
    convert_dict={
        "nn.silu":nn.silu,
        "nn.softmax":nn.softmax,
        "approximate_silu_PDM":approximate_silu_PDM,
        "approximate_mish_PDM":approximate_mish_PDM,
        "approximate_softmax_PDM":approximate_softmax_PDM
        }

    logger.info(f"===== Starting CPU Eval=====")
    #CPU Json Load & Evaluation
    with open('./unlinear_dicts_silu/unlinear_dict_CPU.json', 'r', encoding='utf-8') as file:
        name_of_unlinear_dict = json.load(file)
    for key in name_of_unlinear_dict.keys():
        args.unlinear_function_dict[key]=convert_dict[name_of_unlinear_dict[key]]
    if args.backend in ("all", "cpu"):
        public_evaluation(restored_state, num_each_label, save_dir)

    logger.info(f"CPU Eval Completed! Start SPU Eval:----")
    #SPU Json Load & Evaluation
    with open('./unlinear_dicts_silu/unlinear_dict_SPU.json', 'r', encoding='utf-8') as file:
        name_of_unlinear_dict = json.load(file)
    for key in name_of_unlinear_dict.keys():
        args.unlinear_function_dict[key]=convert_dict[name_of_unlinear_dict[key]]
    if args.backend in ("all", "vanilla"):
        secure_evaluation(restored_state, num_each_label, save_dir)

    logger.info(f"Public Eval Completed! Start SPU Eval:PDM")
    #CTA Json Load & Evaluation
    with open('./unlinear_dicts_silu/unlinear_dict_PDM.json', 'r', encoding='utf-8') as file:
        name_of_unlinear_dict = json.load(file)
    for key in name_of_unlinear_dict.keys():
        args.unlinear_function_dict[key]=convert_dict[name_of_unlinear_dict[key]]
    if args.backend in ("all", "prismdm"):
        secure_evaluation(restored_state, num_each_label, save_dir)

    print("end")




if __name__ == "__main__":
    #SPU Compiler Setup
    copts = spu_pb2.CompilerOptions()
    copts.enable_pretty_print = False
    copts.xla_pp_kind = 2
    copts.enable_optimize_denominator_with_broadcast = True

    parser = get_sample_arg_parser()
    args = get_args(parser)
    if args.backend != "cpu":
        with open(args.config, "r") as file:
            conf = json.load(file)
        ppd.init(conf["nodes"], conf["devices"])

    args.activation="silu"
    for k, v in args.__dict__.items():
        print(f"===> {k}: {v}")

    #Start Testing
    main(args)
