import tqdm
from train_model import *


## 读取数据
np.random.seed(1204)
train_image, train_label = load_mnist('MNIST')
test_image, test_label = load_mnist('MNIST', 't10k')
# 参数查找
hidden_nodes_list = [100, 300, 500, 1000]  # 隐藏层大小
mu_list = [1, 1e-1, 1e-2, 1e-3, 1e-4]  # 正则化参数
learning_rate_list = [1, 0.5, 0.1, 0.05, 0.01, 0.005]  # 学习率
iteration = 200  # 训练次数

optim_test_accuracy = 0
optim_hidden_nodes = 100
optim_mu = 1
optim_learning_rate = 1

for hidden_nodes in tqdm.tqdm(hidden_nodes_list):
    for mu in mu_list:
        for learning_rate in learning_rate_list:
            model = MyNet(784, hidden_nodes, 10)
            _, _, _ = model.train(train_image, train_label, test_image, test_label, learning_rate=learning_rate,
                                  regulariaztion=mu, iteration=iteration, test_flag=False, print_flag=False)
            test_accuracy = (model.predict(test_image) == test_label).mean()
            if test_accuracy > optim_test_accuracy:
                optim_test_accuracy = test_accuracy
                print(optim_test_accuracy)
                optim_hidden_nodes = hidden_nodes
                optim_mu = mu
                optim_learning_rate = learning_rate
            else:
                pass


print("最终学习率为", optim_learning_rate, "；最终隐藏层大小为", optim_hidden_nodes, "；最终正则化强度为", optim_mu)

