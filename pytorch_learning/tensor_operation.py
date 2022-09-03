import torch
import numpy as np
# 基本操作
# x = torch.empty(4, 4, 3)  # 创建一个4x4x3的tensor数组，未经初始化
# print(x)
# x = torch.rand(4, 4, 3)  # 随机初始化
# print(x)
# x = torch.zeros(4, 4, 3, dtype=torch.float)  # 生成一个全为零的数组
# print(x)
# print(x.size())  # 查看x的维度
# x = torch.rand(4, 4, 3)
# y = torch.rand(4, 4, 3)
# z = x+y
# z2 = torch.zeros(4, 4, 3)
# torch.add(x, y, out=z2)  # 两种加法操作
# y.add_(x)
# print(z, z2, y)
# print(z == z2)  # 判断两种方法得到的值是否相同
# print(z == y)
# x = torch.tensor(np.array([[[1, 2, 3]]]))
# print(x)  #  ndarray和tensor的转换
# x = torch.rand(4, 4, 3)
# x1 = x.view(16, 3)
# x2 = x.view(2, 8, 3)
# print(x1, x2)

# 自动求导

# x = torch.ones(2, 2, requires_grad=True)  # 追踪对x的操作
# y = x + 2
# print(y)  # y新增了grad_fn 属性
# z = y*y*3
# out = z.mean()
# print(out)
# out.backward()  # 将最后的运算返回
# print(x.grad)  # 求导
# x = torch.rand(3, requires_grad=True)
# y = x * 2
# while y.data.norm() < 1000:
#     y = y*2
# print(y)
# v = torch.ones(3, dtype=torch.float)
# y.backward(v)  # y这里不是标量，所以还需要传入一个向量
# print(x.grad)
# print((x ** 2).requires_grad)
# with torch.no_grad():
#     print((x ** 2).requires_grad)  # 停止追踪






