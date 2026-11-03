import os
import jax
import flax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from flax import linen as nn
from flax.training import train_state
from flax.training.common_utils import shard
from flax.training import checkpoints
from flax.serialization import to_bytes, from_bytes
import optax
import pickle
import time

from flax.core import FrozenDict
from jax.tree_util import tree_leaves
from flax_ddpm import script_utils
from datasets.flax_tiny_mnist import MnistDataset
from flax_ddpm.script_utils import get_args
from flax_ddpm.script_utils import get_diffusion_from_args
from flax_ddpm.approximation_methods import *
from flax_ddpm.mish_function import mish
from tools import file_utils
import struct
import numpy as np
from torch.utils.data import DataLoader
import logging
import sys
import json
import cv2
import tqdm
import argparse

def get_sample_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str, default="./model_eval")
    parser.add_argument("--num_samples", type=int, default=10)
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

def load_customized_diffusion_from_args(selected_method, customized_json_file_path, args, convert_dict, batch_size, model_path):
    #different with get_customized_diffusion_from_args()
    selected_method_unlinear_function_dict={}
    #Customized Json Load
    with open(customized_json_file_path, 'r', encoding='utf-8') as file:
        name_of_unlinear_dict = json.load(file)
    #Low-Level dict
    for key in name_of_unlinear_dict.keys():
        selected_method_unlinear_function_dict[key]=convert_dict[name_of_unlinear_dict[key]]
    #High-Level dict
    args.unlinear_function_dict[selected_method]=selected_method_unlinear_function_dict

    diffusion = get_diffusion_from_args(args, selected_method)

    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (batch_size,28,28,1))
    y = jnp.ones(batch_size, dtype=jnp.int32)*1
    variables = diffusion.init(rngs=rng,rng=rng, x=x,y=y)# Initialize the parameters
    optimizer = optax.adam(learning_rate=args.learning_rate)
    state = train_state.TrainState.create(
            apply_fn=diffusion.apply,
            params=variables['params'],
            tx=optimizer,
            )
    # load params
    pkl_file=pickle.load(open(model_path,"rb"))
    restored_state=flax.serialization.from_state_dict(target=state, state=pkl_file)
    params = restored_state.params

    return diffusion, params, args

def normalize_to_0_255(array):
    min_val = array.min()
    max_val = array.max()
    normalized_array = (array - min_val) *255/ (max_val - min_val)
    return normalized_array.astype(jnp.uint8)

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

@jax.jit
def sample_image(params, rng, y):
    samples = diffusion_customized.apply({'params': params}, batch_size=1, y=y, use_ema= False, rng_key=rng, ddim_timesteps=50, method=diffusion_customized.sample_ddim)
    image = normalize_to_0_255(samples[0])
    return image

def main(args, model_path, selected_method, model_version, selected_nonlinear_function):
    #Define global parameters
    num_each_label = 1
    batch_size =1

    #Load the model checkpoint
    assert model_path is not None
    assert os.path.exists(model_path), f"model file not exist: {model_path}"

    #Define customized json file paths

    diffusion_customized_json_path=f"./unlinear_dicts_{selected_nonlinear_function}/unlinear_dict_{selected_method}.json" #SPU->No Approximation || CDM/CTA->Different Approximation Methods

    #Prepare Convert Json File STRING->CALLABLE FUNCTION
    convert_dict={
        "nn.silu":nn.silu,
        "mish":mish,
        "nn.softmax":nn.softmax,
        "approximate_silu_CDM":approximate_silu_CDM,
        "approximate_silu_CTA":approximate_silu_CTA,
        "approximate_mish_CDM":approximate_mish_CDM,
        "approximate_mish_CTA":approximate_mish_CTA,
        "approximate_softmax_CDM":approximate_softmax_CDM,
        "approximate_softmax_CTA":approximate_softmax_CTA,
        }

    diffusion_customized, params, args = load_customized_diffusion_from_args(selected_method, diffusion_customized_json_path, args, convert_dict, batch_size, model_path)

    #Generated Img Save Begin
    save_generated_method_dataset_dir = f"./FID_dataset/generated_method_{selected_method}_{model_version}_{selected_nonlinear_function}"
    file_utils.mkdir(save_generated_method_dataset_dir)

    already_generated_number=[] #Avoid Regenerate Already-Generated Labels/Numbers

    for class_number in range(args.num_classes):
        if class_number in already_generated_number:
            continue

        train_dataset = MnistDataset(is_train=True, target_labels=[class_number])
        test_dataset = MnistDataset(is_train=False, target_labels=[class_number])

        print(f"Processing Class Number:{class_number} || Image Total Number:{len(train_dataset)+len(test_dataset)}")

        specific_save_generated_method_dataset_dir=os.path.join(save_generated_method_dataset_dir, "Class_"+str(class_number))

        file_utils.mkdir(specific_save_generated_method_dataset_dir)
        #Read Figures in MNIST Dataset refer to Labels
        for counter in tqdm.tqdm(range(len(train_dataset))):
            #set figure generate-random number
            rng = jax.random.PRNGKey(counter)
            #sample
            y = jnp.ones(num_each_label, dtype=jnp.int32)*class_number
            image = sample_image(params, rng, y)
            #get image & save
            result=jax.device_get(image)
            cv2.imwrite(f"{specific_save_generated_method_dataset_dir}/Class_{class_number}_count_{counter}.png", result)

        base_counter=len(train_dataset)

        for counter in tqdm.tqdm(range(len(test_dataset))):
            #set figure generate-random number
            rng = jax.random.PRNGKey(counter+base_counter)
            #sample
            y = jnp.ones(num_each_label, dtype=jnp.int32)*class_number
            image = sample_image(params, rng, y)
            #get image & save
            result=jax.device_get(image)
            cv2.imwrite(f"{specific_save_generated_method_dataset_dir}/Class_{class_number}_count_{base_counter+counter}.png", result)

if __name__ == "__main__":
    parser = get_sample_arg_parser()
    args = get_args(parser)
    args.num_classes=2
    args.num_timesteps=1000

    for k, v in args.__dict__.items():
        print(f"===> {k}: {v}")

    model_path = "./checkpoints/FA-CTA-PlanA-silu-epoch-50-model.msgpack"
    model_version="trained" #Options: "trained" "untrained"
    selected_method="CTA" #Options: "CPU" "SPU" "CDM" "CTA" "CDM_github_code_version" "CDM_paper_version"
    selected_nonlinear_function="silu" # Options: "relu" "silu" "mish"

    #main(args, model_path, selected_method, model_version, selected_nonlinear_function)
    #Define global parameters
    num_each_label = 1
    batch_size =1

    #Load the model checkpoint
    assert model_path is not None
    assert os.path.exists(model_path), f"model file not exist: {model_path}"

    #Define customized json file paths

    diffusion_customized_json_path=f"./unlinear_dicts_{selected_nonlinear_function}/unlinear_dict_{selected_method}.json" #SPU->No Approximation || CDM/CTA->Different Approximation Methods

    #Prepare Convert Json File STRING->CALLABLE FUNCTION
    convert_dict={
        "nn.silu":nn.silu,
        "mish":mish,
        "nn.softmax":nn.softmax,
        "approximate_silu_CDM":approximate_silu_CDM,
        "approximate_silu_CTA":approximate_silu_CTA,
        "approximate_mish_CDM":approximate_mish_CDM,
        "approximate_mish_CTA":approximate_mish_CTA,
        "approximate_softmax_CDM":approximate_softmax_CDM,
        "approximate_softmax_CTA":approximate_softmax_CTA,
        }

    diffusion_customized, params, args = load_customized_diffusion_from_args(selected_method, diffusion_customized_json_path, args, convert_dict, batch_size, model_path)

    #Generated Img Save Begin
    save_generated_method_dataset_dir = f"./FID_dataset/generated_method_{selected_method}_{model_version}_{selected_nonlinear_function}"
    file_utils.mkdir(save_generated_method_dataset_dir)

    already_generated_number=[] #Avoid Regenerate Already-Generated Labels/Numbers

    for class_number in range(args.num_classes):
        if class_number in already_generated_number:
            continue

        train_dataset = MnistDataset(is_train=True, target_labels=[class_number])
        test_dataset = MnistDataset(is_train=False, target_labels=[class_number])

        print(f"Processing Class Number:{class_number} || Image Total Number:{len(train_dataset)+len(test_dataset)}")

        specific_save_generated_method_dataset_dir=os.path.join(save_generated_method_dataset_dir, "Class_"+str(class_number))

        file_utils.mkdir(specific_save_generated_method_dataset_dir)
        #Read Figures in MNIST Dataset refer to Labels
        for counter in tqdm.tqdm(range(len(train_dataset))):
            #set figure generate-random number
            rng = jax.random.PRNGKey(counter)
            #sample
            y = jnp.ones(num_each_label, dtype=jnp.int32)*class_number
            image = sample_image(params, rng, y)
            #get image & save
            result=jax.device_get(image)
            cv2.imwrite(f"{specific_save_generated_method_dataset_dir}/Class_{class_number}_count_{counter}.png", result)

        base_counter=len(train_dataset)

        for counter in tqdm.tqdm(range(len(test_dataset))):
            #set figure generate-random number
            rng = jax.random.PRNGKey(counter+base_counter)
            #sample
            y = jnp.ones(num_each_label, dtype=jnp.int32)*class_number
            image = sample_image(params, rng, y)
            #get image & save
            result=jax.device_get(image)
            cv2.imwrite(f"{specific_save_generated_method_dataset_dir}/Class_{class_number}_count_{base_counter+counter}.png", result)
