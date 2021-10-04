import argparse
import json
import os
import pickle
import shutil

import numpy as np
import torch

def parse_variable_name(variable_name):
    variable_list = variable_name.split("$")
    layer_number = int(variable_list[1].split(":")[-1])
    head_number = int(variable_list[2].split(":")[-1])
    LOC_left = int(variable_list[3].split(":")[0].strip("["))
    LOC_right = int(variable_list[3].split(":")[1].strip("]"))
    LOC = np.s_[LOC_left:LOC_right]
    return layer_number, head_number, LOC

def get_activation_at(
    model, input_ids, attention_mask, 
    variable_names
):
    outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask
    )
    if not isinstance(model, torch.nn.DataParallel):
        head_dimension = model.config.hidden_size // model.config.num_attention_heads
    else:
        head_dimension = model.module.config.hidden_size // model.module.config.num_attention_heads
    activations = []
    for variable in variable_names:
        layer_index, head_index, LOC = parse_variable_name(
            variable_name=variable
        )
        head_hidden_states = outputs["hidden_states"][layer_index][
            :,:,(head_index*head_dimension):((head_index+1)*head_dimension)
        ]
        head_slice = head_hidden_states[:,:,LOC]
        activations += [head_slice]
    return activations

def interchange_hook(interchanged_variable, LOC):
    # the hook signature
    def hook(model, input, output):
        # interchange inplace.
        output[:,:,LOC] = interchanged_variable
    return hook

def interchange_with_activation_at(
    model, input_ids, attention_mask, 
    interchanged_variables, 
    variable_names
):
    if not isinstance(model, torch.nn.DataParallel):
        head_dimension = model.config.hidden_size // model.config.num_attention_heads
    else:
        head_dimension = model.module.config.hidden_size // model.module.config.num_attention_heads
    # interchange hook.
    hooks = []
    for i in range(len(variable_names)):
        layer_index, head_index, LOC = parse_variable_name(
            variable_name=variable_names[i]
        )
        start_index = head_index*head_dimension + LOC.start
        stop_index = start_index + LOC.stop
        assert LOC.stop <= head_dimension
        # this is a design todo item.
        # we need to avoid using try catch as a conditioned logic.
        
        # TODO: remove hard coded module.
        try:
            if not isinstance(model, torch.nn.DataParallel):
                hook = model.bert.encoder.layer[layer_index].output.register_forward_hook(
                    interchange_hook(interchanged_variables[i], np.s_[start_index:stop_index]),
                )
            else:
                hook = model.module.bert.encoder.layer[layer_index].output.register_forward_hook(
                    interchange_hook(interchanged_variables[i], np.s_[start_index:stop_index]),
                )
        except:
            if not isinstance(model, torch.nn.DataParallel):
                # this is a special distilled bert class.
                hook = model.distilbert.transformer.layer[layer_index].output_layer_norm.register_forward_hook(
                    interchange_hook(interchanged_variables[i], np.s_[start_index:stop_index]),
                )
            else:
                # this is a special distilled bert class.
                hook = model.module.distilbert.transformer.layer[layer_index].output_layer_norm.register_forward_hook(
                    interchange_hook(interchanged_variables[i], np.s_[start_index:stop_index]),
                )
        hooks += [hook]
    # forward.
    interchanged_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    # clean up hooks.
    for hook in hooks:
        hook.remove()
    return interchanged_outputs