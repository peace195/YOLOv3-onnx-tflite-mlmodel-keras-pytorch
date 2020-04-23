import os
import coremltools
import onnxmltools

input_coreml_model = '../webservice/pretrain_models/water_meter/yolo-obj-608-water_15000.mlmodel'

output_onnx_model = '../webservice/pretrain_models/water_meter/yolo-obj-608-water_15000.onnx'

coreml_model = coremltools.utils.load_spec(input_coreml_model)

onnx_model = onnxmltools.convert_coreml(coreml_model)

onnxmltools.utils.save_model(onnx_model, output_onnx_model)

print(os.path.getsize(output_onnx_model))
