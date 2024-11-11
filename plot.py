import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_predictions_vs_target(predictions, target, stock_code, mode='ALL', dataset='csi300'):
    """
    绘制指定股票代码的预测值和目标值的折线图。

    参数:
    predictions (pd.Series): 预测值的 Series，索引为 (股票代码, 日期)
    target (pd.Series): 目标值的 Series，索引为 (股票代码, 日期)
    stock_code (str): 指定的股票代码
    """
    pred_scale = 2
    target_scale = 0.5
    # 提取对应股票的数据
    dir = f'figure/{stock_code}'
    if not os.path.exists(dir): os.makedirs(dir)
    filename = f'{dir}/{dataset}-{mode}.png'
    pred_values = predictions[predictions.index.get_level_values(0) == stock_code] * pred_scale
    target_values = target[target.index.get_level_values(0) == stock_code] * target_scale
    # 检查是否有数据可绘制
    if pred_values.empty or target_values.empty:
        print(f"No data available for stock code: {stock_code}")
        return

    # 绘制折线图
    plt.figure(figsize=(18, 6))
    plt.plot(pred_values.index.get_level_values(1), pred_values.values, label='Predictions', color='blue')
    plt.plot(target_values.index.get_level_values(1), target_values.values, label='Target', color='red')
    plt.ylim(-1.5, 1.5)
    # 添加标题和标签
    plt.title(f'Predictions vs Target for {stock_code}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()

    # 显示图形
    plt.xticks(rotation=45)  # 使日期标签倾斜，便于阅读
    plt.tight_layout()  # 自动调整布局
    plt.savefig(filename)  # 保存文件，使用指定的文件名
    plt.show()
    plt.close()
# 示例调用
# plot_predictions_vs_target(predictions, target, 'SH600000')

def plot_multiple_stocks(predictions, target, stock_codes, mode='ALL', dataset='csi300'):
    """
    绘制指定股票代码的预测值和目标值的折线图，支持同时绘制四个股票。

    参数:
    predictions (pd.Series): 预测值的 Series，索引为 (股票代码, 日期)
    target (pd.Series): 目标值的 Series，索引为 (股票代码, 日期)
    stock_codes (list): 指定的股票代码列表，最多支持 4 个股票
    """
    dir = f'figure/{mode}-{dataset}'
    if not os.path.exists(dir): os.makedirs(dir)
    filename = f'{dir}/2x2-{stock_codes[0]}.png'
    n_stocks = len(stock_codes)

    # 确保最多只绘制 4 个股票
    if n_stocks > 4:
        print("最多支持 4 个股票。")
        return

    # 创建 2x2 网格
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()  # 将 2D 数组展平，方便索引

    for i, stock_code in enumerate(stock_codes):
        # 提取对应股票的数据
        pred_values = predictions[predictions.index.get_level_values(0) == stock_code]
        target_values = target[target.index.get_level_values(0) == stock_code]

        # 检查是否有数据可绘制
        if pred_values.empty or target_values.empty:
            axs[i].set_title(f'No data for {stock_code}')
            continue

        # 绘制折线图
        axs[i].plot(pred_values.index.get_level_values(1), pred_values.values, label='Predictions', color='blue')
        axs[i].plot(target_values.index.get_level_values(1), target_values.values, label='Target', color='red')

        # 添加标题和标签
        axs[i].set_title(f'Predictions vs Target for {stock_code}')
        axs[i].set_xlabel('Date')
        axs[i].set_ylabel('Values')
        axs[i].legend()
        axs[i].grid()

    # 自动调整布局
    plt.tight_layout()
    plt.savefig(filename)  # 保存文件，使用指定的文件名
    plt.show()
    plt.close()

# 示例调用
# plot_multiple_stocks(predictions, target, ['SH600000', 'SH600001', 'SH600002', 'SH600003'])