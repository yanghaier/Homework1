# Homework1
本项目是DATA620004神经网络和深度学习课程的第一次作业，主要为使用numpy构建一个两层的神经网络分类器。
### 超参选取
主要使用网格法：
+ hidden node size: [100, 300, 500, 1000]
+ learning rate:[1, 0.5, 0.1, 0.05, 0.01, 0.005]
+ regularization parameter:[1, 1e-1, 1e-2, 1e-3, 1e-4]

对以上所有参数通过遍历，每次训练200epoch,以在测试机上的准确率作为评价标准，选出最优的超参数组，具体细节实现在find_hyperparam.py文件中。

结果为：hidden_node_size = 1000, learning_rate = 0.005, regularization_parameter = 0.001
### 模型训练
使用选择出的超参数进行训练，并保存模型，具体见train_moedl.py。
### 测试
导入训练好的模型，对测试集数据进行预测，并计算准确率，具体见test.py。
