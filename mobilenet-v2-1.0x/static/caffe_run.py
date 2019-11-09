import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
from caffe2.python import core, net_drawer, net_printer, visualize, workspace, utils
import numpy as np
import time


if __name__ == '__main__':

    model = onnx.load('model.onnx')
    prepared_backend = onnx_caffe2_backend.prepare(model) # bug here??? why ?!!

    # hyper-param
    batch_size = 1
    number_of_iterations = 1000

    # Test the speed
    start_time = time.time()

    for i in range(number_of_iterations):
        inp = np.random.randn(batch_size,3,224,224)

        a = prepared_backend.run(inp.astype(np.float32))
        #pred = a[0]
        #print(pred.shape)

    end_time = time.time()

    print('Num of iterations:{} BatchSize:{} Total Time:{}'.format(number_of_iterations,
                                                                   batch_size, end_time - start_time))
