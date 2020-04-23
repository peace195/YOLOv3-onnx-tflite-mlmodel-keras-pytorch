import onnx

# Load the ONNX model
onnx_model = onnx.load("yolo-obj-608-water_15000.onnx")

# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(onnx_model.graph)


import onnxruntime as ort

ort_session = ort.InferenceSession(onnx_model)

outputs = ort_session.run(None, {'actual_input_1': np.random.randn(1, 3, 608, 608).astype(np.float32)})

print(outputs[0])
