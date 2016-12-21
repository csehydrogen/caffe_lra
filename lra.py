from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

proto_input = "/opt/caffe/examples/cifar10/cifar10_quick_train_test.prototxt"
weight_input = "/opt/caffe/examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5"

proto_output = "/opt/caffe/examples/cifar10/cifar10_quick_lra_train_text.prototxt"
weight_output = "/opt/caffe/examples/cifar10/cifar10_quick_lra.caffemodel"

from proto_lra import proto_lra
from weight_lra import weight_lra

#lra_map = {'conv1': (3,32), 'conv2': (32,32), 'conv3': (32, 64), 'ip1': 64, 'ip2': 10}
lra_map = {'conv1': (3,32), 'conv2': (16,32), 'conv3': (32, 64), 'ip1': 64, 'ip2': 10}

if __name__ == "__main__":
	proto_lra(proto_input, proto_output, lra_map)
	weight_lra(proto_input, weight_input, proto_output, weight_output, lra_map)	
