import onnx

# 加载ONNX模型
onnx_model_path = '/data/MobileFM/yx/AudioCaption/dcase_2020_T6/model.onnx'
model = onnx.load(onnx_model_path)

# 获取模型的计算图
graph = model.graph

# 统计不同算子的种类
operator_types = set()
for node in graph.node:
    operator_types.add(node.op_type)

print("Number of unique operator types used in the ONNX model:", len(operator_types))
print("List of unique operator types:")
print(operator_types)
