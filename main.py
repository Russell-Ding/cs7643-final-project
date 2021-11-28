import torch
import pickle

def load_data(device, *args):
    '''

    :param device: cpu or gpu
    :param args: arguments to determine which relationship matrix to load
    :return:
    '''
    x_numerical = pickle.load(open('./Data/relations_author_source/x_numerical.pkl','rb'))
    y = pickle.load(open('./Data/relations_author_source/y_.pkl','rb'))

    #### load features
    if len(args)>0:
        relation = pickle.load(open(f"./Data/relations_author_source/{args[0]}_relation.pkl",'rb'))
        return x_numerical, y, relation
    else:
        return x_numerical, y, None
