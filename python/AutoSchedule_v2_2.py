#!/usr/bin/env python
# coding: utf-8
import onnx

import glob
import os
import numpy as np
import time

import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
from tvm.auto_scheduler.utils import request_remote
from tvm.contrib import utils, ndk

from tvm import te
from tvm import relay
from tvm.relay.testing.darknet import __darknetffi__


# Also replace this with the device key in your tracker
device_key = "v9h"
rpc_host = "192.168.105.70"
rpc_port = 9190
print("device:", device_key)
print("rpc_host: %s:%s" % (rpc_host, rpc_port))


# Define the tune 
turn_trials = 4000
turn_enable = False
turn_build = False

preload_log_file = False
load_local_log_file = False
# Set this to True if you use ndk tools for cross compiling
use_ndk = False
# Path to cross compiler
# os.environ["TVM_NDK_CC"] = "/usr/bin/aarch64-linux-gnu-g++"

# Define the neural network and compilation target.
# network = "op9_dla"
network = "mobilenet"
#network = "yolov2"
#network = "yolov3"

batch_size = 1
dtype = "float32"
layout = "NCHW"
# layout = "NHWC"

#target_type = "aarch64"
target_type = "opencl"

if network == "mobilenet":
    tune_model = onnx.load('../models/mobilenet/onnx/mobilenetv2-7.onnx')
    input_name = "input"
    input_shape = (batch_size, 3, 244, 244)
    shape_dict = {input_name: input_shape}
    print("shape_dict: ", shape_dict)
    relay_model, params = relay.frontend.from_onnx(tune_model, shape_dict)
elif network == "resnet50":
    tune_model = onnx.load('../models/resnet50/resnet50.onnx')
    input_name = "input"
    input_shape = (batch_size, 3, 244, 244)
    shape_dict = {input_name: input_shape}
    print("shape_dict: ", shape_dict)
    relay_model, params = relay.frontend.from_onnx(tune_model, shape_dict)
elif network == "op9_dla":
    tune_model = onnx.load('../models/op9_dla/20210622_320_192_op9_dla.onnx')
    input_name = "input.1"
    input_shape = (batch_size, 3, 320, 192)
    shape_dict = {input_name: input_shape}
    print("shape_dict: ", shape_dict)
    print("Converting onnx model to relay function...")
    relay_model, params = relay.frontend.from_onnx(tune_model, shape_dict)
elif network == "yolov2":
    cfg_path = "../models/yolov2/darknet/yolov2.cfg" # cfg path
    weights_path = "../models/yolov2/darknet/yolov2.weights"# weights path
    lib_path = "../models/yolov2/darknet/libdarknet2.0.so" # lib path
    DARKNET_LIB = __darknetffi__.dlopen(lib_path)
    print(DARKNET_LIB)
    net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
    print(net)
    data = np.empty([batch_size, net.c, net.h, net.w], dtype)
    input_name = "data"
    input_shape = data.shape
    shape_dict = {input_name: input_shape}
    print(shape_dict)
    print(net.layers[net.n - 1].classes)
    print("Converting darknet to relay function...")
    relay_model, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
elif network == "yolov3":
    cfg_path = "../models/yolov3/darknet/yolov3.cfg" # cfg path
    weights_path = "../models/yolov3/darknet/yolov3.weights"# weights path
    lib_path = "../models/yolov3/darknet/libdarknet2.0.so" # lib path
    DARKNET_LIB = __darknetffi__.dlopen(lib_path)
    print(DARKNET_LIB)
    net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
    print(net)
    data = np.empty([batch_size, net.c, net.h, net.w], dtype)
    input_name = "data"
    input_shape = data.shape
    shape_dict = {input_name: input_shape}
    print(shape_dict)
    print(net.layers[net.n - 1].classes)
    print("Converting darknet to relay function...")
    relay_model, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

if layout == 'NHWC':
    # convert from NCHW to NHWC
    desired_layouts = {'nn.conv2d': ['NHWC', 'default']}

    # Convert the layout to NHWC
    # RemoveUnunsedFunctions is used to clean up the graph.
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts)])

    with tvm.transform.PassContext(opt_level=3):
        relay_model = seq(relay_model)
    
    print(relay_model)


if target_type == "aarch64":
    target = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
elif target_type == "opencl":
    if turn_build:
        target = tvm.target.Target("opencl", host="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
    else:
        #target = tvm.target.Target("opencl -device=mali", host="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
        target = tvm.target.Target("opencl -device=powervr -model=v9h", host="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")

log_file = "%s-%s-B%d-%s-C%s-T%s.json" % (network, layout, batch_size, target.kind.name, turn_trials, time.strftime('%y-%m-%d-%H-%M',time.localtime(time.time())))
print("log file:", log_file)


# remote = request_remote(device_key, rpc_host, rpc_port)
# dev = remote.cl()
# print("device_name:", dev.device_name)
# print("compute_version:", dev.compute_version)
# print("max_clock_rate:", dev.max_clock_rate)
# print("multi_processor_count:", dev.multi_processor_count)
# print("max_thread_dimensions:", dev.max_thread_dimensions)
# max_shared_memory_per_block = dev.max_shared_memory_per_block
# print("max_shared_memory_per_block:", max_shared_memory_per_block)
# max_threads_per_block = dev.max_threads_per_block
# print("max_threads_per_block:", max_threads_per_block)
# warp_size = dev.warp_size
# print("warp_size: ", warp_size)

if turn_enable:
    if target_type == "opencl":
        max_shared_memory_per_block = 4096
        print("max_shared_memory_per_block:", max_shared_memory_per_block)
        max_threads_per_block = 512
        print("max_threads_per_block:", max_threads_per_block)
        warp_size = 2
        print("warp_size: ", warp_size)
        # There is no explicit local memory limition
        # so we can use INT32_MAX to disable the check on local_memory.
        max_local_memory_per_block = 4096000 # INT32_MAX
        print("max_local_memory_per_block:", max_local_memory_per_block)
        max_vthread_extent = 2 #int(dev.warp_size / 4) if int(dev.warp_size / 4) > 1 else dev.warp_size
        print("max_vthread_extent:", max_vthread_extent)
        num_cores = 2
        print("number of cores:", num_cores)
        vector_unit_bytes = 16
        print("vector unit bytes:", vector_unit_bytes)
        cache_line_bytes = 64
        print("cache line bytes:", cache_line_bytes)
        hardware_params = auto_scheduler.HardwareParams(num_cores,
                                                        vector_unit_bytes,
                                                        cache_line_bytes,
                                                        max_shared_memory_per_block,
                                                        max_local_memory_per_block,
                                                        max_threads_per_block,
                                                        max_vthread_extent, 
                                                        warp_size)

        tasks, task_weights = auto_scheduler.extract_tasks(relay_model["main"], params, target,
                                                           hardware_params=hardware_params)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials = turn_trials,  # change this to 20000 to achieve the best performance
            builder = auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
            runner  = auto_scheduler.RPCRunner(
                    device_key,
                    host=rpc_host,
                    port=rpc_port, 
                    repeat=1, 
                    timeout=30,
                    min_repeat_ms = 200),
            measure_callbacks = [auto_scheduler.RecordToFile(log_file)],
        )
    elif target_type == "aarch64":
        tasks, task_weights = auto_scheduler.extract_tasks(relay_model["main"], params, target)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials = turn_trials,  # change this to 20000 to achieve the best performance
            builder = auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
            runner  = auto_scheduler.RPCRunner(
                    device_key,
                    host=rpc_host,
                    port=rpc_port, 
                    repeat=1, 
                    timeout=30,
                    min_repeat_ms = 200,
                    enable_cpu_cache_flush=True),
            measure_callbacks = [auto_scheduler.RecordToFile(log_file)],
        )
        
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    # generate the tune object 
    if preload_log_file:
        load_log_file = "mobilenet-NCHW-B1-opencl-C3000-T21-08-11-21-39.json"
        print("preload file:", load_log_file)
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=load_log_file)
    else:
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)

    try:
        print("Begin tuning...")
        tuner.tune(tune_option)
    except KeyboardInterrupt:
        print('What a rude awakening!')


# Compile the whole network
if turn_build:
    print("Compile for tune...")
    if load_local_log_file:
        log_file = "./tune_log_file/yolov3-NCHW-B1-llvm-C4000-T21-08-17-08-50.json"
        #log_file = "mobilenet-NCHW-B1-opencl-C20000-T21-08-13-20-16.json"
        # log_file = "mobilenet-NCHW-B1-opencl-C3000-T21-08-11-21-39.json" # 76ms -> opencl
        print("Load Local File:", log_file)
    else:
        print("Load Tune File:", log_file)

    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            model_lib = relay.build(relay_model, target, params=params)
else:        
    print("Compile without tune...")
    with tvm.transform.PassContext(opt_level=3):
        model_lib = relay.build(relay_model, target=target, params=params)


# Export lib
temp = utils.tempdir()
filename = target_type + "_deploy_lib.tar"
path_lib = temp.relpath(filename)
model_lib.export_library(path_lib)
# lib.export_library(path_lib, ndk.create_shared)

# upload module to device
print("Upload...")
remote = request_remote(device_key, rpc_host, rpc_port, timeout = 10000)
remote.upload(path_lib)
loaded_lib = remote.load_module(filename)


# Create graph executor
if target_type == "aarch64":
    dev = remote.cpu()
elif target_type == "opencl":
    dev = remote.cl()
module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype)))


# Evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", dev, number=50, repeat=3, min_repeat_ms=50)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))


# ## Evaluate inference time cost
# |device|log file | Mean inference time(ms)|
# |------|---------| -----------------------|
# | opencl | op9_dla-NCHW-B1-opencl-C4000-T21-08-16-15-38.json | 139 |
# | aarch64 | yolov3-NCHW-B1-llvm-C4000-T21-08-17-08-50.json | 2582|

