import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

if __name__ == '__main__':

    model = onnx.load('model.onnx')
    prepared_backend = onnx_caffe2_backend.prepare(model) # bug here??? why ?!!

    c2_workspace = prepared_backend.workspace
    c2_model = prepared_backend.predict_net

    from caffe2.python.predictor import mobile_exporter

    with open('init_net.pb', "wb") as fopen:
        fopen.write(init_net.SerializeToString())
    with open('predict_net.pb', "wb") as fopen:
        fopen.write(predict_net.SerializeToString())

