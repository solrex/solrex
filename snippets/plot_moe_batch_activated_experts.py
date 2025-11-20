import matplotlib.pyplot as plt
import numpy as np

def plot_moe_batch_activated_experts(model_name, total_experts, topk, max_batch_size):
    """
    绘制不同 Batch Size 下激活专家数量的期望曲线。
    
    参数:
    model_name (str): 模型名称，显示在标题中
    total_experts (int): 专家总数 (N)
    topk (int): 每个 Token 选择的专家数 (k)
    max_batch_size (int): X轴的最大 Batch Size (绘制 0 到此数值)
    """
    
    # --- 1. 数据计算 ---
    # 生成 X 轴数据 (Batch Size: 0 ~ max_batch_size)
    x = np.arange(0, max_batch_size + 1)
    
    # 计算公式: E = N * (1 - ((N-k)/N)^x)
    # 单个 Token 未选中某专家的概率
    prob_not_picked = (total_experts - topk) / total_experts
    # x 个 Token 后激活的专家数期望
    y = total_experts * (1 - prob_not_picked ** x)

    # --- 2. 绘图设置 ---
    plt.figure(figsize=(12, 7), dpi=100)
    
    # 绘制主曲线
    plt.plot(x, y, label=f'Acitivated Experts (TopK={topk})', color='#1f77b4', linewidth=2)
    
    # 绘制参考线 (总专家数)
    plt.axhline(y=total_experts, color='red', linestyle='--', alpha=0.5, label=f'Total Experts ({total_experts})')
    
    # --- 3. 智能标注逻辑 ---
    # 自动计算步长：保证图上大约有 32 个左右的密集标签，避免太拥挤
    # 例如：max_batch=128 -> step=4; max_batch=1024 -> step=32
    step = max(4, int(max_batch_size / 32))
    
    # 构造标注点索引：强制包含 1，然后从 step 开始每隔 step 取一个
    label_indices = [1] + list(range(step, max_batch_size + 1, step))
    
    for i, x_val in enumerate(label_indices):
        # 防止索引越界
        if x_val >= len(y): continue
            
        y_val = y[x_val]
        label_text = f"{y_val:.1f}"
        
        # 上下交替显示，防止重叠
        if i % 2 == 0:
            offset = (0, 15)   # 文字在点上方
            va = 'bottom'
        else:
            offset = (0, -20)  # 文字在点下方
            va = 'top'

        # 画点
        plt.scatter(x_val, y_val, color='black', zorder=5, s=25)
        
        # 写字
        plt.annotate(
            label_text, 
            xy=(x_val, y_val), 
            xytext=offset,
            textcoords='offset points',
            ha='center', 
            va=va,
            fontsize=9,
            fontweight='bold',
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.8)
        )

    # --- 4. 图表装饰 ---
    plt.title(f'MoE Acitivated Distinct Experts (Expected) vs Batch Size - {model_name}', fontsize=14)
    plt.xlabel('Batch Size (Parallel Decoding Tokens)', fontsize=12)
    plt.ylabel('Activated Experts (Expected)', fontsize=12)
    
    # 设置 X 轴刻度：每隔 2 个步长显示一个刻度数值，保持由于
    tick_step = step * 2 if step < 10 else step
    plt.xticks(list(range(0, max_batch_size + 1, tick_step)))
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    # 字体设置 (根据系统自动调整，优先使用常见中文字体)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.tight_layout()
    plt.show()

# --- 调用示例 ---
# 示例 1: 您的原始配置
plot_moe_batch_activated_experts(
    model_name="DeepSeek-V3", 
    total_experts=256, 
    topk=8, 
    max_batch_size=128
)

plot_moe_batch_activated_experts(
    model_name="Qwen-A3B", 
    total_experts=128, 
    topk=8, 
    max_batch_size=128
)
