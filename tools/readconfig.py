"""
@ File: generate_data.py
@ Author: Cunliang Ma and Wei Wang
@ Email: mcl@jxust.edu.cn
@ Brief: This is the main script that generate the data.
@ Copyright(C): 2023 Jiangxi University of Science and Technology. All rights reserved.
"""
import yaml

def readconfig(yaml_name):
    file = open(yaml_name)
    conf = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    return conf