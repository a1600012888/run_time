## Requirements
pytorch == 1.3.0

onnx == 1.6.0

caffe2


## How to run


``` bash run.sh ```

## hyper-params

There are two hyper-params in ```caffe_run.py``` 
* batch_size: the batch_size for each iteration. (Default: 1)
* num_of_iterations: how many iterations we need to run to test the speed. (Default: 1000)


## Note:
I did not change the inference device.(Default: CPU)
