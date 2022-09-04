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

后续我会对pytorch的框架继续进行学习，回来更新

2022-09-03

先回答昨天的几个问题
tensor是神经网络中的numpy，两者区别在于前者可以在GPU上运算，而后者只能在CPU上运算

调试过程中查看变量发现example_data是一个128x1x28x28的多维数组，128对应的就是batch_size,长度为1的这个维度暂时不知道用处是什么，28x28储存了一张图片各个像素点的灰度值

然后就是对tensor基础操作的学习，包括创建，加法，矩阵乘法，重组等

最核心的是pytorch的自动求导机制的学习，这个会专门写一篇博客

今天尝试在pycharm中上传项目到Github，遇到了一个问题，提交成功后推送时，出现了报错，发现超时，可能是因为我用Git时输了乱七八糟的指令，最后我直接git init 完美解决

2022-09-04
今天尝试搭建神经网络进行回归，理清了神经网络的基本步骤

搭建过程中遇到了两个问题：输入的特征只有1维的时候，输入向量必须是列向量，创建神经网络类的时候涉及到python类继承方面的基础知识，还没熟悉。

搭建时自己将例子延伸到多层隐藏层的情况，并成功搭建，算是巩固一下所学，后面会去学习神经网络分类，毕竟这个项目是个分类任务
