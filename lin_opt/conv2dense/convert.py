import torch
import torch.nn as nn
from network import SmallDenseNet, SmallConvNet
from torch_conv_layer_to_fully_connected import torch_conv_layer_to_affine
from max_pooling_to_fully_connected import ListOfNetworks

def load_network(network_class, network_path):
    net = network_class()
    net.load_state_dict(torch.load(network_path))
    net.eval()
    return  net.double()

def get_network_conv():
    NETWORK = "mnist_conv_net.pt"
    MODEL = SmallConvNet
    LAYERS = 5
    INPUT_SIZE = (1, 28, 28) 

    net = load_network(MODEL, NETWORK)
    return net

def to_dense(net, input_size):

    seq = next(net.children())
    assert isinstance(seq, nn.Sequential)

    list_of_layers = []
    input_ = torch.rand(input_size, dtype=torch.float64)
    for layer in seq:
        if isinstance(layer, nn.Linear):
            list_of_layers.append(layer)
        elif isinstance(layer, nn.ReLU):
            list_of_layers.append(layer)
        elif isinstance(layer, nn.Flatten):
            input_ = layer(input_)
#            list_of_layers.append(layer)
            pass # must be after max pooling! TODO fix
        elif isinstance(layer, nn.MaxPool2d):
            print("max pooling")
            input_ = layer(input_)
            print("loading")
            new_layer = torch.load("tmp/max_sparse_layer.pt")
            print("loaded")
            assert isinstance(new_layer, nn.Sequential)
            new_layers = list(new_layer.children())
            list_of_layers.extend(new_layers)
        elif isinstance(layer, nn.Dropout):
            pass
        elif isinstance(layer, nn.Conv2d):
            input_size = input_.shape
            input_ = layer(input_)
            print("****", input_size, "****")
            new_layer = torch_conv_layer_to_affine(layer.cpu(), input_size[-2:])
            print("we are back", flush=True)
            list_of_layers.append(new_layer)
            print("here we are", flush=True)
        else:
            print(type(layer))
            
    return nn.Sequential(*list_of_layers)
    #return list_of_layers

input_ = torch.rand((1,1,28,28), dtype=torch.float64)

net = get_network_conv()
net2 = to_dense(net, input_size=(1,28,28))



output1 = net(input_)
output2 = net2(input_)


#output2 = input_
#for layer in net2:
#    print(layer.__class__.__name__)
#    print(output2.shape)
#    output2 = layer(output2)
    
    
print(output1)
print(output2)

print("saving network")
torch.save(net2, "tmp/pa_conv_net.pt")
print("saved")
print("finished")
