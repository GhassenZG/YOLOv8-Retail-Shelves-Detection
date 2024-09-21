import torch

# Load your trained PyTorch model
model = torch.hub.load('ultralytics/yolov8', 'yolov8s')
model.load_state_dict(torch.load('models\ best.pt'))
model.eval()

# Example input
dummy_input = torch.randn(1, 3, 640, 640)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, 'best.onnx', opset_version=11)
