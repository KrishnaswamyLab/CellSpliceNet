from argparse import Namespace
import configparser
import os
import pathlib
import numpy as np
import argparse
import ast
import re


def init_base_config(path_to_config_dir):

    config = configparser.ConfigParser()

    # files
    config.add_section("input_files")

    # files
    config.add_section("processed_files")

    # params
    config.add_section("params")

    # info
    config.add_section("info")

    fname = pathlib.Path(path_to_config_dir).joinpath("data_config.ini")
 
    with open(fname, "w") as f:
        config.write(f)


def check_path_dtype(path):

    if type(path) == pathlib.Path:
        pass
    elif type(path) == str:
        path = pathlib.Path(path)

    return path


def update_config(path_to_config_dir, level, key, value):

    path_to_config_dir = check_path_dtype(path_to_config_dir)

    fname = path_to_config_dir.joinpath("data_config.ini")

    config = configparser.ConfigParser()
    config.read(fname)

    value = str(value)
    config.set(level, key, value)

    with open(fname, "w") as f:
        config.write(f)


def get_config_val(path_to_config_dir, level, key):

    path_to_config_dir = check_path_dtype(path_to_config_dir)

    fname = path_to_config_dir.joinpath("data_config.ini")

    config = configparser.ConfigParser()
    config.read(fname)
    return config[level][key]



def get_config_val_from_file(path_to_config_file, level, key):

    path_to_config_file = check_path_dtype(path_to_config_file)

    config = configparser.ConfigParser()
    config.read(path_to_config_file)
    return config[level][key]


def str2list(v):
    if v == None:
        return v
    else:
        if type(eval(v)) == list:
            return eval(v)
        else:
            raise argparse.ArgumentTypeError("target list unable to be parsed")


def load_target_stats(string):

    try:
        stats_list = eval(string)

    except:
        stats_list = ast.literal_eval(re.sub(r"\bnan\s*", "-10000.0", string))

    return stats_list


def load_config_as_namespace(path_to_config_dir):

    config = configparser.ConfigParser()
    config.read(path_to_config_dir)

    files_ns = Namespace(**dict(config.items("files")))
    params_ns = Namespace(**dict(config.items("params")))
    info_ns = Namespace(**dict(config.items("info")))

    params_ns.max_seq_len = int(params_ns.max_seq_len)
    info_ns.vocab_size = int(info_ns.vocab_size)
    info_ns.target_cols = eval(info_ns.target_cols)

    info_ns.target_mean = np.array(load_target_stats(info_ns.target_mean))
    info_ns.target_std = np.array(load_target_stats(info_ns.target_std))
    info_ns.target_max = np.array(load_target_stats(info_ns.target_max))

    config_ns = {"files": files_ns, "params": params_ns, "info": info_ns}
    config_ns = Namespace(**config_ns)
 
    return config_ns
