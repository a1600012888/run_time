from torch_model import create_network
import torch

if __name__ == "__main__":

    net = create_network() # random initialized

    batch_size = 1

    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    vec = torch.randn(batch_size, 2048, requires_grad=True)

    torch_out = torch.onnx._export(net, (x, vec), 'model.onnx', export_params=True,
            keep_initializers_as_inputs=True)



