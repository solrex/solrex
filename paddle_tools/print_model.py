#!/usr/bin/env python
from paddle import fluid
import sys
from os import path

# Print paddle model to string

# Input arg: model path
model_path = sys.argv[1]

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

print("##### Model Input Tags #####")
print(r[1])
print("##### Model Output Target #####")
print(r[2])
print("##### Model Program #####")
print(r[0])
