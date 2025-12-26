import numpy as np


def generate_strict_star_weights(num_points=500):
    """
    生成严格位于星形辐射线上的权重组合。
    点均匀分布在从中心到边界的6条射线上。
    """
    center = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    # 6个主要辐射方向（已归一化）
    directions = np.array([
        [2, -1, -1],  # 指向 (1,0,0)
        [-1, 2, -1],  # 指向 (0,1,0)
        [-1, -1, 2],  # 指向 (0,0,1)
        [1, 1, -2],  # 指向 (0.5,0.5,0)
        [1, -2, 1],  # 指向 (0.5,0,0.5)
        [-2, 1, 1]  # 指向 (0,0.5,0.5)
    ]) / 3.0

    all_weights = []
    points_per_direction = int(np.ceil(num_points / len(directions)))

    for d in directions:
        # 为每条射线生成点，强度从0.05到0.95（避开中心和边界极值）
        lambdas = np.linspace(0.05, 0.95, points_per_direction)

        for lam in lambdas:
            # 核心公式：沿方向移动
            point = center + lam * d

            # 修正可能出现的极小负值（数学上应为非负）
            point = np.maximum(point, 1e-10)
            # 重新归一化确保和为1
            point = point / point.sum()

            all_weights.append(point.tolist())

    # 截取所需数量
    all_weights = all_weights[:num_points]

    # 转换为Python原生float（保留4位小数）
    return [[round(x, 6) for x in w] for w in all_weights]


# ============ 执行生成与保存 ============
if __name__ == "__main__":
    # 生成500个严格星形辐射权重
    weights_list = generate_strict_star_weights(500)

    # 保存为Python文件（可直接导入）
    with open('star_weights_500.py', 'w', encoding='utf-8') as f:
        f.write('star_weights_500 = ')
        f.write(repr(weights_list))

    # 同时保存为纯文本文件方便查看
    with open('star_weights_500.txt', 'w', encoding='utf-8') as f:
        f.write('[')
        for i, w in enumerate(weights_list):
            if i > 0:
                f.write(', ')
            f.write(f'{w}')
        f.write(']')

    print("生成完成。已保存文件：")
    print("1. star_weights_500.py （Python列表格式，可直接导入使用）")
    print("2. star_weights_500.txt （文本格式，方便查看）")