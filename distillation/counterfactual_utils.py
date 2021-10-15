import argparse
import json
import os
import pickle
import shutil

import numpy as np
import torch


def deserialize_variable_name(variable_name):
    deserialized_variables = []
    variable_list = variable_name.split("$")
    if "[" in variable_list[1]:
        left_l = int(variable_list[1].split(":")[1].strip("["))
        right_l = int(variable_list[1].split(":")[2].strip("]"))
    else:
        left_l = int(variable_list[1].split(":")[-1])
        right_l = int(variable_list[1].split(":")[-1])+1
    if "[" in variable_list[2]:
        left_h = int(variable_list[2].split(":")[1].strip("["))
        right_h = int(variable_list[2].split(":")[2].strip("]"))
    else:
        left_h = int(variable_list[2].split(":")[-1])
        right_h = int(variable_list[2].split(":")[-1])+1

    left_d = int(variable_list[3].split(":")[0].strip("["))
    right_d = int(variable_list[3].split(":")[1].strip("]"))
    
    for i in range(left_l, right_l):
        for j in range(left_h, right_h):
            deserialized_variable = f"$L:{i}$H:{j}$[{left_d}:{right_d}]"
            deserialized_variables += [deserialized_variable]
    return deserialized_variables


def parse_variable_name(variable_name, model_config=None):
    if model_config == None:
        variable_list = variable_name.split("$")
        layer_number = int(variable_list[1].split(":")[-1])
        head_number = int(variable_list[2].split(":")[-1])
        LOC_left = int(variable_list[3].split(":")[0].strip("["))
        LOC_right = int(variable_list[3].split(":")[1].strip("]"))
        LOC = np.s_[LOC_left:LOC_right]
        return layer_number, head_number, LOC
    else:
        # to be supported.
        pass
    

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


def interchange_hook(interchanged_variable, sampled_interchange_position, LOC):
    # the hook signature
    def hook(model, input, output):
        # interchange inplace.
        # TODO: consider the position here.
        batch_size = output.shape[0]
        for i in range(batch_size):
            s = sampled_interchange_position[i][0]
            e = sampled_interchange_position[i][1]
            d_s = sampled_interchange_position[i][2]
            d_e = sampled_interchange_position[i][3]
            output[i,s:e,LOC] = interchanged_variable[i,d_s:d_e,:]
    return hook


def interchange_with_activation_at(
    model, input_ids, attention_mask, 
    interchanged_variables, 
    variable_names,
    sampled_interchange_position,
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
                    interchange_hook(interchanged_variables[i], sampled_interchange_position, np.s_[start_index:stop_index]),
                )
            else:
                hook = model.module.bert.encoder.layer[layer_index].output.register_forward_hook(
                    interchange_hook(interchanged_variables[i], sampled_interchange_position, np.s_[start_index:stop_index]),
                )
        except:
            if not isinstance(model, torch.nn.DataParallel):
                # this is a special distilled bert class.
                hook = model.distilbert.transformer.layer[layer_index].output_layer_norm.register_forward_hook(
                    interchange_hook(interchanged_variables[i], sampled_interchange_position, np.s_[start_index:stop_index]),
                )
            else:
                # this is a special distilled bert class.
                hook = model.module.distilbert.transformer.layer[layer_index].output_layer_norm.register_forward_hook(
                    interchange_hook(interchanged_variables[i], sampled_interchange_position, np.s_[start_index:stop_index]),
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