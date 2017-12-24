## 论文中的per-pixel multinomial sigmoid loss是什么
[author implente code](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/a581e48846b81f880e165b98c8d89897586fef2e/siftflow-fcn32s/net.py)    
从源代码中可以看到作者使用的是caffe的[softmaxwithloss](http://caffe.berkeleyvision.org/tutorial/layers/softmaxwithloss.html)

即先对输入计算softmax，然后使用[MultinomialLogisticLossLayer](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1MultinomialLogisticLossLayer.html#details)  
MultinomialLogisticLossLayer的输入要是概率分布，计算公式
> $$E=-1/N \sum_{i=1}^{N}\log(\hat p_n,l_n)$$
这里$N$指batch_size，$\forall n, \sum_{k=1}^{K}{\hat p_n}=1$，$l_n\in\{0,1,……,K-1\}$是真实label

## 为什么pad 100
100 padding for 2 reasons:
    1) support very small input size
    2) allow cropping in order to match size of different layers' feature maps
https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/models/fcn8s.py

## 为什么分割label的边界都那么明显  
边界表示VOC数据集的标签中周围都是白色，并且图片中有的部分也是白色，像素值是255。
0表示背景，255表示不计入loss，带上背景，共有$21$类  
注意不能使用cv2读label，只有使用PIL读出的才是序号label

## 对loss求均值？
对图像数目求均值还是计入loss的像素数目？
per-pixel 应该是计入loss的像素数目

## 图像是否缩放到一致大小？
label的缩放，只能使用近邻插值，不能引入新的类别  
图片要按长宽比等比例缩放  
因为batch_size=1，不需要缩放

## 训练时loss一直是3.0445?
math.exp(-3.0445) = 0.0476 = 1./21, 即每一个预测是1./21的正确率，完全随机  
开始全部预测错误是正常的，使用VGG16_from_caffe的预训练模型初始化网络，在训练几张图片时，loss就会下降，不过也在小数点后第5位变动。

## 为什么网络权重初始化为0
init_weight函数只是用于设置反卷积层的参数，卷积层置0后，又被pretrained覆盖

## SGD 0.1学习率，batch_size=20，第二epoch时出现nan损失，不是inf??
[nan loss的出现原因](http://www.jianshu.com/p/9018d08773e6)
可能是梯度爆炸，因为损失稍微上升之后，突然Nan   
使用Adam优化器更消耗显存，batch_size=10，学习率改为0.001。  
学习率为0.01时，经过一个batch后损失就升到百万级后才开始降。

## pytorch 的vgg16预训练模型
预训练模型的输入要均值为0，方差为1

## 训练后，预测输出完全乱的？
是因为保存的模型有问题或者加载模型时有问题，因为不训练的模型的损失为3.0445，而训练后模型的损失是3.455，更高了？   
忘记model.eval()， 补上后不起作用   
不同大小的图片loss不一样？？  
去掉数据增强，resize，使用batch_size=1  不起作用，观察多个开源FCN，batch_size确实为1
原论文的weight_decay 是$5e-4$，不是$5^{-4}$？  不起作用  
发现训练集并不是PASCAL VOC，而是SBD(Semantic Boundaries Dataset)     
是loss的原因，不是对图像的像素数目取loss的平均，而是在计算完整幅图像的loss后，对输入的图像数目取平均，现在正常了


# evaluate
FCN8
Accuracy: 88.5370091544
Accuracy Class: 68.73723663
Mean IU: 55.1793869933
FWAV Accuracy: 80.5496839107

Accuracy: 91.3853625064
Accuracy Class: 76.9016720714
Mean IU: 64.6996659807
FWAV Accuracy: 84.8389738198


FCN32
Accuracy: 86.7887748668
Accuracy Class: 67.7846132391
Mean IU: 51.7852606026
FWAV Accuracy: 77.805823855
