import matplotlib.pyplot as plt
import re

from matplotlib.ticker import MaxNLocator

# 从日志文件中读取数据
log_file = "outputs/SparseBEV/deep-supervisored-v1/train.log"

# 使用正则表达式提取 loss 值
with open(log_file, "r") as f:
    log_data = f.read()

loss_values = re.findall(r"loss: (\d+\.\d+)", log_data)

# 将 loss 值转换为浮点数
loss_values = [float(value) for value in loss_values]

# 每个 epoch 包含 3517 个 loss 值
epoch_size = 3517

# 将所有的 loss 值分组
loss_values_grouped = [
    loss_values[i : i + epoch_size] for i in range(0, len(loss_values), epoch_size)
]

# 对每组的 loss 值求平均
average_loss_per_epoch = [sum(group) / len(group) for group in loss_values_grouped]

fig = plt.figure(figsize=(10, 5))

# 创建第一个子图
ax1 = fig.add_subplot(121)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.plot(loss_values)
ax1.set_title("Loss Curve per Iteration")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss")

# 创建第二个子图
ax2 = fig.add_subplot(122)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.plot(average_loss_per_epoch)
ax2.set_title("Average Loss Curve per Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Average Loss")

# 调整子图之间的间距
plt.tight_layout()

# 保存图像为 png 文件
plt.savefig("loss_curves.png")
plt.close()
