# pytorch_basic
pytorch入门项目：利用CNN网络实现MNIST数据集数字识别

在边学习边总结，因此分阶段写下自己犯过的一些错误

2022-09-02

下载mnist数据集，并成功读取数据

今天主要的收获是，了解了下载并读取数据的函数

```python
train_dataset = mnist.MNIST('./data', train=True, transform=transform)
```
这个函数的作用是导入数据集，第一个参数是指定数据集存放的位置（在同文件夹中创建一个名为"data"的文件夹）本地没有数据集的情况下可以加参数`download = True`，作用是从MNIST官网上下载数据集并解压在raw文件夹中，再处理得到能用的数据在processed文件夹里，后缀为.pt 这里我就遇到了问题，下载到后面会报错，提示被终止了，可能是因为官网的原因，因此建议直接把data文件夹放在里面，这样可以避免错误

还有不足的地方是，代码中涉及了batch_size,transform,标准化等问题,涉及到CNN的理论部分
```python
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
```
以及load后的数据类型,我查看后发现是tensor类型，和我预想中的RGB值得到的矩阵不一样，以及下面这句代码`plt.imshow(example_data[i][0], cmap='gray', interpolation='none')`为什么example是二维数组，下标为0的那个维度含义是什么,因为我刚接触这个框架，还需要具体地了解，学习pytorch中的数据结构

后续我会跟莫凡的课进行学习，回来更新
