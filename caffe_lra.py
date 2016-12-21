import os, sys, copy, numpy as np
CAFFE_ROOT = os.environ['CAFFE_ROOT']
os.chdir(CAFFE_ROOT)
sys.path.insert(0, CAFFE_ROOT + '/python')

from google import protobuf
from caffe.proto import caffe_pb2
import caffe
from hosvd import conv_svd, ip_svd

def proto_lra(proto_input, proto_output, lra_map):
    net = caffe_pb2.NetParameter()
    with open(proto_input, 'r') as f:
        protobuf.text_format.Merge(f.read(), net)
    new_layers = []
    for layer in net.layer:
        modified = False
        if layer.name in lra_map:
            if layer.type == 'Convolution':
                args = lra_map[layer.name]
                layers = [copy.deepcopy(layer) for i in xrange(3)]

                layers[0].name = layer.name + '_lra_a'
                layers[0].top[0] = layer.name + '_lra_a'
                layers[0].convolution_param.num_output = args[0]
                layers[0].convolution_param.pad[0] = 0
                layers[0].convolution_param.kernel_size[0] = 1
                layers[0].convolution_param.stride[0] = 1

                layers[1].name = layer.name + '_lra_b'
                layers[1].bottom[0] = layer.name + '_lra_a'
                layers[1].top[0] = layer.name + '_lra_b'
                layers[1].convolution_param.num_output = args[1]

                layers[2].name = layer.name + '_lra_c'
                layers[2].bottom[0] = layer.name + '_lra_b'
                layers[2].convolution_param.pad[0] = 0
                layers[2].convolution_param.kernel_size[0] = 1
                layers[2].convolution_param.stride[0] = 1

                new_layers.extend(layers)
                modified = True
            elif layer.type == 'InnerProduct':
                arg = lra_map[layer.name]
                layers = [copy.deepcopy(layer) for i in xrange(2)]

                layers[0].name = layer.name + '_svd_a'
                layers[0].top[0] = layer.name + '_svd_a'
                layers[0].convolution_param.num_output = arg

                layers[1].name = layer.name + '_svd_b'
                layers[1].bottom[0] = layer.name + '_svd_a'

                new_layers.extend(layers)
                modified = True
        if not modified:
            new_layers.append(layer)
    for i in xrange(len(net.layer)):
        net.layer.pop()
    net.layer.extend(new_layers)
    with open(proto_output, 'w') as f:
        f.write(str(net))

def weight_lra(proto_input, weight_input, proto_output, weight_output, lra_map):
    net_input = caffe_pb2.NetParameter()
    with open(weight_input, 'rb') as f:
        net_input.ParseFromString(f.read())
    net_output = caffe.Net(proto_output, caffe.TEST)
    for layer in net_input.layer:
        modified = False
        if layer.name in lra_map:
            if layer.type == 'Convolution':
                args = lra_map[layer.name]
                A = np.array(layer.blobs[0].data).reshape(layer.blobs[0].shape.dim)
                w = [None] * 3
                w[0], w[1], w[2] = conv_svd(A, args[0], args[1])
                layer_names = [layer.name + '_lra_' + c for c in 'abc']
                for i in xrange(3):
                    net_output.params[layer_names[i]][0].data[:] = w[i]
                net_output.params[layer_names[2]][1].data[:] = layer.blobs[1].data
                modified = True
            elif layer.type == 'InnerProduct':
                arg = lra_map[layer.name]
                A = np.array(layer.blobs[0].data).reshape(layer.blobs[0].shape.dim)
                w = [None] * 2
                w[0], w[1] = ip_svd(A, arg)
                layer_names = [layer.name + '_svd_' + c for c in 'ab']
                for i in xrange(2):
                    net_output.params[layer_names[i]][0].data[:] = w[i]
                net_output.params[layer_names[1]][1].data[:] = layer.blobs[1].data
                modified = True
        if not modified:
            if layer.name in net_output.params:
                for i, d in enumerate(net_output.params[layer.name]):
                    d.data = np.array(layer.blobs[i].data).reshape(layer.blobs[i].shape.dim)
    net_output.save(weight_output)

