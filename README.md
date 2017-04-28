# stanford深度学习编程练习
 
 包括深度学习的编程练习部分原材料和代码。代码在octave4.21上调试通过。
 
 深度学习网上教程地址： http://ufldl.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B
 
### 稀疏自编码器
     x --> lr(N1x) --> lr(N2x2) --> X_head
     通过两层神经网络，把 X 和 X_head的交叉熵损失降为最小
     这部分代码里实现了 矢量化编程：主要是提高了性能，矩阵计算并行化。要不然效率非常低
     
     [sparseae_exercise](sparseae_exercise)
     
### 主成分分析与白化（预处理的算法）
     共两个练习。使用了eig/svd来计算特征值和特征向量。可以参考 linalg 里的eig/svd算法
     通过对已有特征的线性组合，得到新的特征。新特征维度更低，而且正交
     [pca_2d](pca_wd)
     [pca_exercise](pca_exercise)

### Softmax回归
    使用softmax进行回归预测
    [softmax_exercise](softmax_exercise)

### 自学习
    全称为 自我学习与无监督特征学习
    和稀疏自编码一样，只是取了输入层至隐层的参数，对输入参数进行非线性变换，以此来做为新的特征lr(Ax)(一般是50维或100维）
    [stl_exercise](stl_exercise)

### 建立分类用深度网络
    通过栈式自编码，对每层进行自编码，然后使用反向传播法进行微调（输出层为softmax)
    stackedae_exercise

### 自编码线性解码器
    稀疏自编码器包含3层神经元，分别是输入层，隐含层以及输出层。线性解码器，指输入到隐层还是logic/tanh函数，隐层到输出是线性函数。这样得到的模型更容易应用，而且模型对参数的变化也更为鲁棒。（word2vec, 输入层是线性的，输出是logic函数，与这个正好相反）
    [linear_decoder_exercise](linear_decoder_exercise)

### cnn(卷积神经网）
    卷积神经网用来处理大型图像，通过卷积提取特征，通过池化降维(平移不变性？(translation invariant))
    [cnn_exercise](cnn_exercise)

 

 
 
 
 
 
