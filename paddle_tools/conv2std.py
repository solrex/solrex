#!/usr/bin/env python
from paddle import fluid
import sys
from os import path

# Convert paddle models to standard format: "std_model_path/__model__, std_model_path/__params__"

# Input arg: model path
model_path = sys.argv[1]
# Output arg: save_inference model path
std_model_path = sys.argv[2]

exe = fluid.Executor(fluid.CPUPlace())

if path.isfile(path.join(model_path, '__params__')):
    r = fluid.io.load_inference_model(model_path, exe, '__model__', '__params__')
elif path.isfile(path.join(model_path, 'params.pdparams')):
    r = fluid.io.load_inference_model(model_path, exe, 'model.pdmodel', 'params.pdparams')
elif path.isfile(path.join(model_path, '__model__')):
    r = fluid.io.load_inference_model(model_path, exe, '__model__')
elif path.isfile(path.join(model_path, 'model.pdmodel')):
    r = fluid.io.load_inference_model(model_path, exe, 'model.pdmodel')
else:
    print("Invalid paddle model path: %s" % model_path)
    exit(1)

# Save the old model to a new location
fluid.io.save_inference_model(dirname=std_model_path, feeded_var_names=r[1], target_vars=r[2], executor=exe,
        model_filename='__model__', params_filename="__params__", main_program=r[0])
