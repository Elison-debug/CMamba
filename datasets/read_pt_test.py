import torch

# 读入权重文件
ckpt = torch.load("./ckpt/grid/result/best1_epe.pt", map_location="cpu")

# 看整体结构
print(type(ckpt))
print(ckpt.keys() if isinstance(ckpt, dict) else "not a dict")

# 如果是 state_dict
state_dict = ckpt["model"]


# 打印所有参数名
for name, param in state_dict.items():
    print(name, param.shape)

