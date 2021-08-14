#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import sys
import numpy as np
import onnx
import os
import glob
from onnx import numpy_helper
import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, graph_executor as runtime
from tvm.contrib.download import download_testdata

# get the target device type 
target_type = sys.argv[1]
#model_name = "resnet50"
model_name = "googlenet"
print("model name:", model_name)
# load the nn model
if model_name == "mobilenet":
    mnist_model = onnx.load('mobilenet/mobilenetv2-7.onnx')
    # configure the model input shape
    input_name = "input"
    input_shape = (1, 3, 244, 244);
    dtype = "float32"
    shape_dict = {input_name: input_shape}
elif model_name == "resnet50":
    mnist_model = onnx.load('resnet50/resnet50.onnx')
    # configure the model input shape
    input_name = "input"
    input_shape = (1, 3, 244, 244);
    dtype = "float32"
    shape_dict = {input_name: input_shape}
elif model_name == "googlenet":
    mnist_model = onnx.load('googlenet/googlenet-9.onnx')
    # configure the model input shape
    input_name = "data_0"
    input_shape = (1, 3, 244, 244);
    dtype = "float32"
    shape_dict = {input_name: input_shape}


# The following is my environment, change this to the IP address of
# your target device
host = "192.168.104.240"
port = 9090
# convert the model to ir
model, params = relay.frontend.from_onnx(mnist_model, shape_dict, freeze_params=True)
if target_type == 'x86_64':
    target = tvm.target.Target('llvm')
    target_host = tvm.target.Target('llvm')
elif target_type == 'aarch64':
    target      = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
    target_host = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
elif target_type == 'opencl':
    target      = tvm.target.Target("opencl -device=mali")
    target_host = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon")
print("device:", target)
print("host:", target_host)

with tvm.transform.PassContext(opt_level=3):
    model_lib = relay.build(model, target=target, target_host=target_host, params=params, mod_name='mnist')
print(model_lib)

# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.tar")
model_lib.export_library(lib_fname)

if target_type == 'x86_64':
    remote = rpc.LocalSession()
    dev = remote.cpu(0)
elif target_type == 'aarch64':
    remote = rpc.connect(host, port)
    dev = remote.cpu(0)
elif target_type == 'opencl':
    remote = rpc.connect(host, port)
    dev = remote.cl(0)

print("Target type:", target_type)
# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module("net.tar")
module = runtime.GraphModule(rlib["mnist"](dev))

# set input data
module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))) 

# run
module.run()

# get output
out = module.get_output(0)

# get the result
number = np.argmax(out.numpy())
print("TVM prediction number: ", number)

# evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", dev, number=1, repeat=50)
prof_res = np.array(ftimer().results) * 1000
# convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
