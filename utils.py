import os
from models.fc import *
from models.wide_resnet import wide_resnet_t
from models.wide_resnet_1 import WideResNet
from models.all_cnn import allcnn_t
from models.lenet import lenet

def create_path(model_name, args, num_true, num_random, dataset):

    path = model_name
    for item in args:
        path += "_" + str(item)

    path += "_" + str(num_true) + "_" + str(num_random) + "_" + dataset

    path = os.path.join("./", path, "")

    return path


def mkdir(path):

    isfolder = os.path.exists(path)
    if not isfolder:
        os.makedirs(path)



def get_model_class(model_name):
    
    if model_name == "fc":
        mc = Network
    if model_name == "wr":
        mc = WideResNet
    if model_name == "all_cnn":
        mc = allcnn_t
    if model_name == "lenet":
        mc = lenet

    return mc
