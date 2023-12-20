from envs import create_env
import os
import dill
from torchviz import make_dot
import click
import torch
from typing import Tuple, List, Iterable
from pydot import Dot, graph_from_dot_data, Edge
from graphviz.graphs import BaseGraph
from graphviz import Source
import networkx as nx
import torch.nn as nn
import hiddenlayer as hl
import numpy as np
from .torchview import draw_graph
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy import Policy
import networkx as nx
import pydotplus
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from copy import deepcopy
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT, STATE_OUT
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch

from utils.utils import flatten
MAX_DEPTH=4
activations = {
    "tanh",
    "relu",
    "relu6",
    "elu",
    "sigmoid",
    "silu"
}
activations_names = {
    "Tanh",
    "ReLU",
    "ReLU6",
    "ELU",
    "Sigmoid",
    "SiLU"
}
activation_pattern = "|".join([act for act in list(activations_names)])

def edge_to_node_ids(edge: Edge) -> Tuple[str, str]:
    """Returns the node id pair for the edge object"""
    return (edge.get_source(), edge.get_destination())


def get_graph_dot_obj(graph_spec) -> List[Dot]:
    """Get a dot (graphs) object list from a variety of possible sources (postelizing inputs here)"""
    _original_graph_spec = graph_spec
    if isinstance(graph_spec, (BaseGraph, Source)):
        # get the source (str) from a graph object
        graph_spec = graph_spec.source
    if isinstance(graph_spec, str):
        # get a dot-graph from dot string data
        graph_spec = graph_from_dot_data(graph_spec)
    # make sure we have a list of Dot objects now
    assert isinstance(graph_spec, list) and all(
        isinstance(x, Dot) for x in graph_spec
    ), (
        f"Couldn't get a proper dot object list from: {_original_graph_spec}. "
        f"At this point, we should have a list of Dot objects, but was: {graph_spec}"
    )
    return graph_spec


def get_edges(graph_spec, postprocess_edges=edge_to_node_ids):
    """Get a list of edges for a given graph (or list of lists thereof).
    If ``postprocess_edges`` is ``None`` the function will return ``pydot.Edge`` objects from
    which you can extract any information you want.
    By default though, it is set to extract the node pairs for the edges, and you can
    replace with any function that takes ``pydot.Edge`` as an input.
    """
    graphs = get_graph_dot_obj(graph_spec)
    n_graphs = len(graphs)

    if n_graphs > 1:
        return [get_edges(graph, postprocess_edges) for graph in graphs]
    elif n_graphs == 0:
        raise ValueError(f"Your input had no graphs")
    else:
        graph = graphs[0]
        edges = graph.get_edges()
        if callable(postprocess_edges):
            edges = list(map(postprocess_edges, edges))
        return edges
    
def create_forward(model):
    def forward(self, inputs):
        input_dict = inputs["input_dict"]
        state = inputs["state"]
        seqlens = inputs["seqlens"]
        
        return model(input_dict, state=state, seq_lens=seqlens)[0]
    return forward

def set_display_name(state):
    state = nested_transform_nparray_to_tensor(state, copy_tensor=True)
    state_items = list(state.items())
    state_names_list = []
    for i, (k, s) in enumerate(state_items):
        if k not in ("actor", "critic"):
            hidden_for = " (Actor/Critic)"
        else:
            K = k.capitalize()
            hidden_for = f" ({K})"
        hidden_for = ""
        if isinstance(s, dict):
            s_items = list(s.items())
            for kst, st in s_items:
                st.display_name = f"Hidden<br/>state{hidden_for}"
                state_names_list.append(st.display_name)
        else:
            s.display_name = f"Hidden<br/>state{hidden_for}"
            state_names_list.append(s.display_name)
    return state, state_names_list

class TorchWrapper(nn.Module):

    def __init__(self, model: PPOTorchRLModule):
        super(TorchWrapper, self).__init__()
        self._model = model

    def forward(self, *args, **kwargs):
        inputs, state = args
        batch = {
            "obs": inputs,
            "state_in": state
        }



        output = self._model.forward_exploration(batch)
        #print(dir(self._model))
        #exit()
        state_out = output["state_out"]
        value_function = output["vf_preds"]
        action_distribution = output["action_dist_inputs"]
        state_out = set_display_name(state_out)[0]
        if "actor" in list(state_out.keys()) and "critic" in list(state_out.keys()):
            return set_display_name(state_out["actor"]), action_distribution, set_display_name(state_out["critic"]), value_function
        else:
            return (value_function, action_distribution, set_display_name(state_out)[0])
def nested_transform_nparray_to_tensor(d, copy_tensor=False, apply_op=lambda t: t):
    if torch.is_tensor(d):
        return apply_op(d if not copy_tensor else torch.clone())
    elif type(d) == np.ndarray:
        return apply_op(torch.from_numpy(d))
    elif type(d) == list:
        return [nested_transform_nparray_to_tensor(v, apply_op=apply_op) for v in d]
    else:
        return {
            k:nested_transform_nparray_to_tensor(v, apply_op=apply_op) for k,v in d.items()
        }
    
def remove_depth(node_content):
    node_content_copy = node_content.replace("depth:0", "")
    for i in range(1, MAX_DEPTH+1):
        node_content_copy = node_content_copy.replace(f"depth:{i}", "")
    return node_content_copy

def color_activation_function(node_content):
    for activation in activations:

        if activation in node_content.lower():
            return node_content.replace("darkseagreen1", "aliceblue")

    return node_content

def color_output(node_content):

    if "output-tensor" in node_content.lower():
        return node_content.replace("lightyellow", "coral")

    return node_content

def process_raw_edge(edge):
    edges = edge.split("->")
    return [edges[0].replace("\n", "").replace("\t", "").rstrip().lstrip(), edges[1].replace("\n", "").replace("\t", "").rstrip().lstrip()]

import re
def convert_to_scalar(match_shape):
    shape_replaced_by_scalar = match_shape.group(0).replace("(", "").replace(")", "").split(",")
    shape_replaced_by_scalar = str(np.prod([int(i) for i in shape_replaced_by_scalar]))
    return shape_replaced_by_scalar

def convert_to_scalar_2(match_shape):
    shape_replaced_by_scalar = match_shape.group(0).split(",")
    shape_replaced_by_scalar = str(np.sum([int(i) for i in shape_replaced_by_scalar]))
    return shape_replaced_by_scalar

def shape_to_scalar(txt):
    return re.sub(r'\(([1-9][0-9]*,\s)*[1-9][0-9]*\)', convert_to_scalar, txt)   

def commas_to_sum(txt):
    return re.sub(r'([1-9][0-9]*,\s)*[1-9][0-9]*', convert_to_scalar_2, txt)

def remove_mult(txt):
    return re.sub(r'[1-9][0-9]*\sx\s', "", txt) 

def add_linear_to_raw_hidden_layer(txt):
    return txt.replace("layer<BR/>", "layer<BR/>(Linear)")

def remove_space_before_td(txt):
    return txt.replace(" </TD>", "</TD>")

def create_multiply_value(m):
    def multiply_value(v):
        print(v)
        return "["+str(int(np.round(float(v.group(0))*float(m), 0)))+"]"
    return multiply_value

def remove_brackets(v):
    return v.group(0).replace("[", "").replace("]", "")

def multiply_value_by_fct(txt, value_to_multiply):
    new_txt = txt
    for v, m in value_to_multiply:
        pattern = r'(?<!\[)[1-9][0-9]*(?!\])'
        pattern = pattern.replace("[1-9][0-9]*", str(v))
        new_txt = re.sub(pattern, create_multiply_value(m), txt)
    pattern = r"\[[1-9][0-9]*\]"
    return re.sub(pattern, remove_brackets, new_txt)
    


@click.command()
@click.option('--path-to-model-files', "path_to_model_file", help='Path to model + additional stuff files needed for generating.', required=True)
@click.option('--output-filename', "output_filename", help='Output file name.', default="out")
@click.option('--start', "start", help='Start attribute.', default="auto")
@click.option('--replace-txt', "replace_txt", type=(str, str), help='Replace txt by another in the graph. Case sensitive.', default=[], multiple=True)
@click.option('--font-size', "font_size", type=int, help='Font size', default=10)
@click.option('--graph-dims', "graph_dims", type=(str, str), help='Graph dimensions.', default=None)
@click.option('--multiply-value-by', "multiply_value_by", type=(str, str), help='Multiply value by', multiple=True, default=[])
def plot_model(path_to_model_file, output_filename, start, replace_txt, font_size, graph_dims, multiply_value_by):
    policy = torch.load(path_to_model_file+"/model.m")
    inputs = dill.load(open(path_to_model_file+"/inputs.dat", "rb"))
    state = dill.load(open(path_to_model_file+"/state.dat", "rb"))

    """
    out, state = model(input_dict, state=state, seq_lens=seqlens)
    out_critic = model.value_function()
    out_unified = torch.cat([out, out_critic.unsqueeze(1)], dim=-1)
    dot = make_dot((out, out_critic), params=dict(model.named_parameters()), show_attrs=True, show_saved=False)
    dot_nodes = [
        b for b in dot.body if "->" not in b
    ]
    dot_edges = [
        b.replace("\n", "") for b in dot.body if "->" in b
    ]
    G = nx.Graph()


    dot_body = [
       b for b in dot.body if "fillcolor" in b
    ]
    print("\n".join(dot_edges))
    #dot.body = dot_body
    dot.render(path_to_model_file+"/"+output_filename, format="pdf")
    """
    
    """
    input = nested_transform_nparray_to_tensor({
        "input_dict": input_dict,
        "state": state,
        "seqlens": seqlens
    })
    """
    inputs.display_name = "Observation"
    j = 0
    state, state_names_list = set_display_name(state)
    state_names = list(state_names_list)
    next_state_names = [str(s) for s in state_names]
    
    if len(next_state_names) == 2:
        outputs = [next_state_names[0]] + [next_state_names[1]] + ["Critic<br/>value"] + ["Action<br/>distribution"]
    else:
        outputs = next_state_names + ["Critic<br/>value"] + ["Action<br/>distribution"]
    
    len_state_names_list = len(state_names_list)
    input_names = ["Observation"] + state_names_list
    model_graph = draw_graph(TorchWrapper(policy), input_data=(inputs, state), device='cpu', depth=MAX_DEPTH, graph_dir="TB", font_size=str(font_size))
    dot = model_graph.visual_graph

    
    
    nodes_to_delete = [
        n.split(" ")[0].replace("\t", "").replace("\n", "")  for n in dot.body if "aliceblue" in n or "LayerNorm" in n
    ]
    all_edges = [
        tuple(process_raw_edge(edge)) for edge in dot.body if "->" in edge
    ]
    import time
    
    while len(set(flatten(all_edges)).intersection(set(nodes_to_delete))) > 0:
        
        
        for node_to_delete in nodes_to_delete:
            edges_to_remove = []
            edges_to_add = []
            edges_0 = [edge for edge in all_edges if edge[1] == node_to_delete]
            edges_1 = [edge for edge in all_edges if edge[0] == node_to_delete]
            edges_to_remove += [edge for edge in all_edges if node_to_delete in edge]
            for edge_0 in edges_0:
                for edge_1 in edges_1:
                    edges_to_add += [(edge_0[0], edge_1[1])]
            all_edges = [e for e in all_edges if e not in edges_to_remove]
            all_edges += edges_to_add
    parsed_node_to_node = {
        n.split(" ")[0].replace("\t", "").replace("\n", ""):n.split(" ")[0]  for n in dot.body
    }
    all_edges = list(set(all_edges))
    #print(nodes_to_delete)
    #all_edges = [e for e in all_edges if e[0] not in nodes_to_delete and e[1] not in nodes_to_delete]
    #exit()
    dot.body = [
        color_output(color_activation_function(remove_depth(b))).replace("input:", "In").replace("output: ", "Out").replace("Linear", "Hidden<br />layer") for b in dot.body if ("->" not in b and b.split(" ")[0].replace("\t", "").replace("\n", "") not in nodes_to_delete)
    ]
    dot.body = [
        (b.replace("<TD>(", "<TD>").replace(")</TD>", "</TD>").replace("1, ", "") if "input-tensor" in b else b) for b in dot.body
    ]
    dot.body = [
        (b.replace("<TD>(", "<TD>").replace(")</TD>", "</TD>").replace("1, ", "") if "output-tensor"  else b) for b in dot.body
    ]
    i = 0
    j = 0
    while i < len(outputs) and j < len(dot.body):
        if "output-tensor" in dot.body[j]:
            dot.body[j] = dot.body[j].replace("output-tensor", outputs[i])
            i += 1
        j += 1

    i = 0
    j = 0
    
    while i < len(input_names) and j < len(dot.body):
        if "input-tensor" in dot.body[j]:
            dot.body[j] = dot.body[j].replace("input-tensor", input_names[i])
            i += 1
        j += 1
    
    #print(edges_to_add)
    
    nodes_to_delete = []
    nodes_to_replace = []
    i = 0
    while i < len(all_edges):
        node_1, node_2  = all_edges[i]
        node_1_text = next((b for b in dot.body if f"{node_1} [label" in b), None)
        node_2_text = next((b for b in dot.body if f"{node_2} [label" in b), None)
        if node_1_text is not None and node_2_text is not None and "Hidden<br />layer" in node_1_text and "aliceblue" in node_2_text:
            m = re.search(activation_pattern, node_2_text)
            #print(m.group(0))
            node_1_modified = node_1_text.replace("Hidden<br />layer", f"Hidden<br />layer<br />({m.group(0)})")
            #print(node_1_modified)
            nodes_to_delete += [node_2]
            #nodes_to_replace = [(node_1_text, node_1_modified)]
            dot.body = [
               (node_1_modified if node_1_text == b else b) for b in dot.body if b != node_2_text
            ]
            edges = [(node_1, e[1]) for e in all_edges if e[0] == node_2]
            all_edges = [all_edges[j] for j in range(len(all_edges)) if j != i]
            all_edges.extend(edges)
            i -= 1
        i += 1
    all_edges = [e for e in all_edges if e[0] not in nodes_to_delete and e[1] not in nodes_to_delete]
    
    dot.body += [
        f"{e[0]} -> {e[1]}\n" for e in all_edges
    ]
    
    
    dot.body = [
        shape_to_scalar(b) for b in dot.body
    ]
    dot.body = [
        multiply_value_by_fct(b, multiply_value_by) for b in dot.body
    ]
    dot.body = [
        commas_to_sum(b) for b in dot.body
    ]
    dot.body = [
        (remove_mult(b) if "GRU" in b or "LSTM" in b or "BRC" in b or "nBRC" in b else b) for b in dot.body 
    ]
    dot.body = [
        (add_linear_to_raw_hidden_layer(b) if "Hidden<br />layer" in b and re.search(activation_pattern, b) is None else b) for b in dot.body 
    ]
    dot.body = [
        remove_space_before_td(b) for b in dot.body
    ]
    for txt_to_replace, subst in replace_txt:
        dot.body = [
            b.replace(txt_to_replace, subst) for b in dot.body
        ]
    
    
    #exit()
    """
    #Get IDs of nodes to be deleted
    parsed_node_to_node = {
        n.split(" ")[0].replace("\t", "").replace("\n", ""):n.split(" ")[0]  for n in dot.body
    }
    nodes_to_delete = [
        n.split(" ")[0].replace("\t", "").replace("\n", "")  for n in dot.body if "aliceblue" in n
    ]
    #outputs = [
    #    b for b in dot.body if "output-tensor" in b
    #]
    #exit()
    edges_to_add = []
    for node_to_delete in nodes_to_delete:
        edges = []
        for try_edge in dot.body:
            if "->" in try_edge:
                edge = try_edge.split(" ")
                edge = [edge[0].replace("\n", "").replace("\t", ""), edge[2].replace("\n", "").replace("\t", "")]
                if node_to_delete in edge:
                    edges.append(edge)
        edges_0 = [edge for edge in edges if edge[1] == node_to_delete]
        edges_1 = [edge for edge in edges if edge[0] == node_to_delete]
        for edge_0 in edges_0:
            for edge_1 in edges_1:
                edges_to_add += [[edge_0[0], edge_1[1]]]

    
            

    dot.body = [
        color_output(color_activation_function(remove_depth(b))).replace("input:", "In size:").replace("output:", "Out size:").replace("Linear", "Hidden<br />layer") for b in dot.body if (("->" in b and b.split(" ")[0].replace("\t", "").replace("\n", "") not in nodes_to_delete and b.split(" ")[2].replace("\t", "").replace("\n", "") not in nodes_to_delete) 
                                or ("->" not in b and b.split(" ")[0].replace("\t", "").replace("\n", "") not in nodes_to_delete))
    ]
    dot.body += [
        f"{parsed_node_to_node[e[0]].rstrip()} -> {parsed_node_to_node[e[1]].rstrip().lstrip()}\n" for e in edges_to_add
    ]

    nodes_to_delete = set()
    dot_body = []
    
        
    for b in dot.body:
        add_to_body = True
        b_copy = str(b)
        for state_name in state_names:
            state_name_in_node = state_name in b
            if "->" not in b and state_name_in_node:
                node_already_in = len([b2 for b2 in dot_body if state_name in b2]) > 0
                if node_already_in:
                    add_to_body = False
                    break
            
        if add_to_body:
            dot_body.append(b_copy)
        else:
            nodes_to_delete.add(b.split(" ")[0].replace("\t", "").replace("\n", ""))

    j = 0
    old_dot_body = list(dot_body)
    dot_body = []
    already_employed_output = set()
    for b in old_dot_body:
        if "->" not in b:
            if "output-tensor" in b:
                print(outputs[j], already_employed_output)
                if outputs[j] not in already_employed_output:
                    new_b = b.replace("output-tensor", outputs[j])
                    dot_body.append(new_b.replace("Hidden", "Next hidden"))
                    already_employed_output.add(outputs[j])
                else:
                    nodes_to_delete.add(b.split(" ")[0].replace("\t", "").replace("\n", ""))
                j += 1
            else:
                dot_body.append(b)
        else:
            dot_body.append(b)

    dot.body = [
        b for b in dot_body if ("->" not in b or (b.split(" ")[0].replace("\t", "").replace("\n", "") not in nodes_to_delete and b.split(" ")[2].replace("\t", "").replace("\n", "") not in nodes_to_delete))
    ]
    print([
        b.replace("\t", "").replace("\n", "") for b in dot_body if ("->" in b and (b.split(" ")[0].replace("\t", "").replace("\n", "") not in nodes_to_delete and b.split(" ")[2].replace("\t", "").replace("\n", "") not in nodes_to_delete))
    ])
    """
    
    #dot.body = dot_body
    if start == "auto":
        dot.graph_attr["start"] = str(np.random.randint(1,100000))
    else:
        dot.graph_attr["start"] = start
    dot.graph_attr["ordering"] = "out"
    dot.graph_attr["splines"] = "ortho"
    if graph_dims is not None:
        dot.graph_attr["fixedsize"] = "true"
        dot.graph_attr["width"] = graph_dims[0]
        dot.graph_attr["height"] = graph_dims[1]
    dot.render(path_to_model_file+"/"+output_filename+".dot", format="pdf", outfile=path_to_model_file+"/"+output_filename+".pdf")
    
    
    """

    input_names = ['Observation']
    output_names = ['Action', "Value function", "Hidden state"]
    torch.onnx.export(TorchWrapper(model, input), input, path_to_model_file+"/"+output_filename, input_names=input_names, output_names=output_names)
    #model_graph = draw_graph(TorchWrapper(model), input_data=input, device='meta', depth=6)
    #dot = model_graph.visual_graph
    #dot.render(path_to_model_file+"/"+output_filename, format="pdf")
    """
if __name__ == "__main__":
    plot_model()