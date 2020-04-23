import torch.onnx
import torch
from torch.autograd import Variable
from models import Darknet

# # Load the trained model from file
# trained_model = Darknet(
#     '../webservice/pretrain_models/water_meter/yolo-obj-608-water.cfg')
# trained_model.load_state_dict(torch.load(
#     '../webservice/pretrain_models/water_meter/yolo-obj-608-water_15000.pt'))

# # Export the trained model to ONNX
# # one black and white 28 x 28 picture will be the input to the model
# dummy_input = Variable(torch.randn(1, 3, 608, 608))
# torch.onnx.export(trained_model, dummy_input,
#                   '../webservice/pretrain_models/water_meter/yolo-obj-608-water_15000_new.onnx')


# Export an ONNX model from a PyTorch .pt model
# Loading the input PyTorch model and mapping the tensors to CPU
device = torch.device('cpu')
model = torch.load(
    '../webservice/pretrain_models/water_meter/yolo-obj-608-water_15000.pt', map_location=device)

# Generate a dummy input that is consistent with the network's arhitecture
dummy_input = torch.randn(1, 3, 608, 608)

# Export into an ONNX model using the PyTorch model and the dummy input
torch.onnx.export(model, dummy_input,
                  '../webservice/pretrain_models/water_meter/yolo-obj-608-water_15000_new_2.onnx')
