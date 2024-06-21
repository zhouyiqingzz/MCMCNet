import matplotlib.pyplot as plt

# 示例数据
x = [0.7, 0.8, 0.9]
#CHN6数据集上超参数消融
y1 = [0.479, 0.485, 0.489]#5%标签率
y2 = [0.518, 0.521, 0.526]#10%标签率
y3 = [0.548, 0.553, 0.557]#20%标签率

# 绘制折线图
plt.plot(x, y1, marker='o', label='5% labeled ratio')
plt.plot(x, y2, marker='o', label='10% labeled ratio')
plt.plot(x, y3, marker='o', label='20% labeled ratio')

# 添加标题和标签
plt.title('The Ablation Study of alpha')
plt.xlabel('The value of alpha')
plt.ylabel('IoU')

# 保存和显示图形
plt.legend()
plt.savefig('/home/arsc/tmp/pycharm_project_698/DA_Road/test_output/figures/ablation_alpha.png')
plt.show()
