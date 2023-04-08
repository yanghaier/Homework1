from train_model import *


# 读取数据
np.random.seed(1204)
test_image, test_label = load_mnist('MNIST', 't10k')
# 导入模型
model = MyNet(784, 1000, 10)
model.load_model('./result/model.npz')

print('test accuracy:', (model.predict(test_image) == test_label).mean())
