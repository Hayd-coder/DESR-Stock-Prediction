# -*- coding: utf-8 -*-
import gradio as gr
import pandas as pd
import torch
import os
from PIL import Image
import warnings
from datetime import datetime
import random
import traceback
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# --- 导入模型和项目模块 ---
try:
    from model import EGRU
except ImportError as e:
    print(f"错误：无法导入 EGRU 模型类: {e}")
    EGRU = None

try:
    from process_stock_data import get_stock_data, clean_csv_files
    from stock_prediction_egru import run_stock_prediction_egru
except ImportError as e:
    print(f"错误：无法导入必要的项目模块。")
    print(f"请确保 process_stock_data.py 和 stock_prediction_egru.py 文件存在。")
    print(f"详细错误: {e}")
    # exit()

# --- 全局设置 ---
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'tmp/gradio_egru_final_run'  # 新目录

# --- 创建必要的目录 ---
os.makedirs(os.path.join(SAVE_DIR, 'ticker'), exist_ok=True)


# --- 数据获取函数 ---
def get_data(ticker, start_date, end_date, progress=gr.Progress()):
    """根据股票代码和日期范围获取本地股票数据或调用函数下载"""
    data_folder = os.path.join(SAVE_DIR, 'ticker')
    os.makedirs(data_folder, exist_ok=True)
    temp_path = os.path.join(data_folder, f'{ticker}.csv')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_data_path = os.path.join(current_dir, 'data', f'{ticker}.csv')
    try:
        progress(0, desc="检查本地文件...")

        if os.path.exists(local_data_path):
            print(f"发现本地数据文件: {local_data_path}")
            stock_data = pd.read_csv(local_data_path, index_col='Date', parse_dates=True)
            stock_data = stock_data.loc[(stock_data.index >= start_date) & (stock_data.index <= end_date)]
            progress(0.5, desc="保存数据副本...")
            stock_data.to_csv(temp_path)
            print(f"数据副本已保存到: {temp_path}")
            progress(1.0, desc="数据加载完成（本地）")
            return temp_path, "成功从本地加载数据"

        if 'get_stock_data' not in globals() or 'clean_csv_files' not in globals():
            error_msg = "错误: 缺少 get_stock_data 或 clean_csv_files 函数，且本地数据未找到。"
            print(error_msg)
            return None, error_msg

        progress(0.2, desc="本地未找到，正在尝试联网获取...")
        stock_data = get_stock_data(ticker, start_date, end_date)
        if stock_data is None or stock_data.empty:
            raise ValueError(f"get_stock_data 未能成功获取 {ticker} 的数据。")
        print(f"get_stock_data 返回了 {len(stock_data)} 行数据。")

        progress(0.5, desc="保存原始数据...")
        stock_data.to_csv(temp_path)
        progress(0.7, desc="清理数据...")
        clean_csv_files(temp_path)
        progress(1.0, desc="数据获取和处理完成")
        return temp_path, "数据在线获取成功"

    except Exception as e:
        print(f"获取数据出错 (ticker: {ticker}, start: {start_date}, end: {end_date}):")
        traceback.print_exc()
        return None, f"获取数据出错: {str(e)}"


# 生成AI文本摘要
def generate_ai_summary(metrics):
    """
    生成基于预测指标的 AI 文本摘要。
    这是一个占位符函数，实际应用中您需要将其替换为调用大型语言模型的逻辑。
    """
    if not metrics or not isinstance(metrics, dict):
        return "无法生成摘要：指标数据缺失或格式不正确。"

    ticker = metrics.get('Ticker', '该股票')  # 尝试从metrics获取股票代码，若无则使用默认
    rmse = metrics.get('RMSE')
    mae = metrics.get('MAE')
    dir_acc = metrics.get('Directional Accuracy')
    total_profit = metrics.get('Total Profit ($)')
    return_rate = metrics.get('Return Rate (%)')
    buy_times = metrics.get('Buy Times')
    sell_times = metrics.get('Sell Times')

    summary_parts = [f"**📈 AI 对 {ticker} 的投资分析摘要**\n"]
    summary_parts.append("---")

    # 1. 模型性能评估
    summary_parts.append("\n**模型预测性能评估:**")
    if rmse is not None and mae is not None:
        summary_parts.append(
            f"- **预测精度**: 模型的均方根误差 (RMSE) 为 {rmse:.3f}，平均绝对误差 (MAE) 为 {mae:.3f}。这些指标反映了模型预测价格与实际价格的平均偏离程度。通常情况下，这些值越低，代表模型的预测越贴近实际价格。")
    else:
        summary_parts.append("- 预测精度指标 (RMSE, MAE) 未提供。")

    if dir_acc is not None:
        dir_acc_pct = dir_acc * 100
        summary_parts.append(
            f"- **价格方向预测准确率**: {dir_acc_pct:.2f}%。此指标衡量模型预测股价未来上涨或下跌方向的准确性。高于 50% 通常被视为一个积极信号，表明模型对市场趋势有一定的判断能力。")
    else:
        summary_parts.append("- 价格方向预测准确率未提供。")

    # 2. 模拟交易表现
    summary_parts.append("\n**模拟交易表现分析:**")
    if total_profit is not None and return_rate is not None:
        profit_str = f"{total_profit:,.2f}美元"
        return_str = f"{return_rate:.2f}%"
        if total_profit > 0:
            summary_parts.append(
                f"- **盈利能力**: 基于模型的预测进行模拟交易，实现了 **{profit_str} 的总收益**，对应的投资回报率为 **{return_str}**。这是一个积极的信号，显示策略具有潜在盈利性。")
        elif total_profit == 0:
            summary_parts.append(
                f"- **盈亏平衡**: 模拟交易结果为盈亏平衡，总收益为 {profit_str}，投资回报率为 {return_str}。")
        else:
            summary_parts.append(
                f"- **亏损情况**: 模拟交易显示策略出现亏损，总亏损为 **{abs(total_profit):,.2f}美元**，投资回报率为 **{return_str}**。这提示投资者需要警惕潜在风险。")
    else:
        summary_parts.append("- 模拟交易的盈利和回报率数据未提供。")

    if buy_times is not None and sell_times is not None:
        summary_parts.append(
            f"- **交易活动**: 在模拟期间，策略共执行了 {buy_times} 次买入操作和 {sell_times} 次卖出操作。")
    else:
        summary_parts.append("- 模拟交易次数未提供。")

    # 3. 综合投资决策参考
    summary_parts.append("\n**综合投资决策参考:**")
    positive_signals = 0
    neutral_signals = 0
    negative_signals = 0

    if dir_acc is not None:
        if dir_acc > 0.55:
            positive_signals += 1
        elif dir_acc < 0.45:
            negative_signals += 1
        else:
            neutral_signals += 1

    if return_rate is not None:
        if return_rate > 5:
            positive_signals += 1  # 假设5%以上回报率是好的
        elif return_rate < -2:
            negative_signals += 1  # 假设-2%以下回报率是差的
        else:
            neutral_signals += 1

    if total_profit is not None:
        if total_profit <= 0 and (return_rate is not None and return_rate < 0):  # 明确亏损
            negative_signals += 1  # 亏损也算一个负面信号

    if positive_signals > negative_signals and positive_signals >= 1:
        summary_parts.append(
            f"- **整体观点 (偏积极)**: 模型对 {ticker} 的分析显示出一些积极迹象。例如，方向预测准确率和/或模拟交易回报表现良好。这可能表明该股票具有一定的投资潜力。建议投资者结合当前市场环境和个人风险偏好，进一步关注。")
    elif negative_signals > positive_signals and negative_signals >= 1:
        summary_parts.append(
            f"- **整体观点 (偏谨慎)**: 模型对 {ticker} 的分析提示投资者应保持谨慎。可能的原因包括较低的方向预测准确率、模拟交易亏损或预测精度不理想。建议在做出投资决策前进行更全面的风险评估。")
    else:  # positive_signals == negative_signals or all neutral
        summary_parts.append(
            f"- **整体观点 (中性)**: 模型对 {ticker} 的分析结果表现中性，或各项指标之间存在一些矛盾。目前尚无明确的强力买入或卖出信号。建议投资者保持观望，或结合更多维度的信息（如基本面分析、市场情绪等）进行综合判断。")

    summary_parts.append(
        "\n---\n**⚠️ 重要声明**: 此 AI 摘要是基于算法模型的历史数据预测和模拟交易结果生成的，**仅供学习和参考，不构成任何实际的投资建议或操作指导**。股票市场存在固有风险，历史表现不能保证未来结果。投资者在做出任何投资决策前，应进行独立研究，充分了解相关风险，并咨询专业的财务顾问。请对自己的投资行为负责。")

    return "\n".join(summary_parts)


# --- Gradio 按钮处理函数 ---
def run_desr_pipeline_interface(
        temp_csv_path,
        sequence_length, hidden_dim, epochs, batch_size, learning_rate,
        save_dir_base
):
    """
    Gradio 按钮点击的处理函数，调用 DESR 训练、预测、简单交易模拟及AI摘要流程。
    """
    if not temp_csv_path or not os.path.exists(temp_csv_path):
        print(f"错误: 未找到有效的数据文件路径: {temp_csv_path}")
        # 返回匹配所有输出组件的空值 (10 个)
        return [None, None, None, None, None, None, None, None, pd.DataFrame(), "AI 分析失败：数据文件未找到。"]

    ticker = os.path.basename(temp_csv_path).replace('.csv', '')
    output_dir_for_run = os.path.join(save_dir_base, 'desr_results', ticker)

    try:
        if 'run_stock_prediction_egru' not in globals():
            raise ImportError("缺少 run_stock_prediction_egru 函数")

        print(f"\n--- 开始为 {ticker} 运行 DESR 训练、预测与模拟交易 ---")
        results = run_stock_prediction_egru(  # 此函数应返回包含所有指标的列表或字典
            file_path=temp_csv_path,
            output_dir=output_dir_for_run,
            sequence_length=int(sequence_length),
            hidden_dim=int(hidden_dim),
            num_epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate)
        )
        print(f"--- DESR 流程为 {ticker} 执行完毕 ---")

        if results is None or not isinstance(results, list) or len(results) < 9:  # 确保返回了至少9个指标
            print("错误: DESR 核心流程执行失败或返回结果不完整。")
            # 返回匹配所有输出组件的空值 (10 个)
            return [None, None, None, None, None, None, None, None, pd.DataFrame(), "AI 分析失败：核心流程错误。"]

        # 解包结果 (假设顺序与 stock_prediction_egru.py 中一致)
        # [gallery_paths_list, rmse, mae, dir_acc, total_profit, return_rate, buy_times, sell_times, transaction_log_df]

        metrics_dict = {
            'Ticker': ticker,  # 添加股票代码到字典
            'RMSE': results[1],
            'MAE': results[2],
            'Directional Accuracy': results[3],
            'Total Profit ($)': results[4],
            'Return Rate (%)': results[5],
            'Buy Times': results[6],
            'Sell Times': results[7]
            # 您可以根据需要添加更多从 results 中提取的指标
        }

        ai_summary = generate_ai_summary(metrics_dict)

        # 将 AI 摘要附加到结果列表的末尾
        final_results = results + [ai_summary]
        return final_results  # 现在返回10个元素

    except Exception as e:
        print(f"运行 DESR Pipeline 时出错 (Ticker: {ticker}):")
        traceback.print_exc()
        # 返回匹配所有输出组件的空值 (10 个)
        return [None, None, None, None, None, None, None, None, pd.DataFrame(), f"AI 分析失败：{str(e)}"]


# --- Gradio 界面定义 ---
custom_css = """
/* --- CSS 保持不变 --- */
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; } 
.gradio-container { max-width: 1280px !important; margin: auto; padding: 20px; } 
h1 { text-align: center; color: #2c3e50; font-size: 2.5em; margin-bottom: 25px; font-weight: 600; } 
label > .label-text ,
.gr-form > * > .label {
    font-size: 1.1em !important; 
    font-weight: 600 !important; 
    color: #34495e !important; 
    margin-bottom: 8px !important; 
    display: block; 
}
.gr-form > * > .description {
    font-size: 0.9em !important;
    color: #7f8c8d !important; 
    margin-bottom: 5px;
}
.gr-markdown h3 { 
    font-size: 1.6em !important;
    color: #16a085; 
    border-bottom: 2px solid #1abc9c;
    padding-bottom: 8px;
    margin-top: 25px;
    margin-bottom: 20px;
    font-weight: 500;
}
.gr-markdown h4 { 
    font-size: 1.3em !important;
    color: #2c3e50;
    margin-top: 20px;
    margin-bottom: 10px;
    font-weight: 600;
}
input[type="text"], input[type="number"], textarea {
    font-size: 1.05em !important; 
    padding: 12px !important; 
    border: 1px solid #bdc3c7 !important; 
    border-radius: 6px !important; 
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.075); 
}
input[type="range"] { cursor: pointer; height: 8px; }
.gr-slider span { font-size: 1em !important; font-weight: bold; color: #2980b9; margin-left: 10px; }
.gr-button {
    font-size: 1.15em !important; 
    padding: 12px 28px !important; 
    border-radius: 8px !important; 
    font-weight: 600 !important;
    cursor: pointer;
    transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease; 
    margin: 10px 5px !important; 
    border: none !important; 
}
.gr-button.lg.primary, .gr-button.primary {
    background-color: #3498db !important; 
    color: white !important;
    box-shadow: 0 4px 8px rgba(52, 152, 219, 0.25);
}
.gr-button.lg.primary:hover, .gr-button.primary:hover {
    background-color: #2980b9 !important; 
    transform: translateY(-2px); 
    box-shadow: 0 6px 12px rgba(41, 128, 185, 0.3);
}
.gr-button.lg.primary:disabled, .gr-button.primary:disabled {
    background-color: #bdc3c7 !important; 
    color: #7f8c8d !important;
    cursor: not-allowed;
    box-shadow: none;
    transform: none;
}
#gallery { background-color: #f8f9f9; border: 1px solid #ecf0f1; border-radius: 8px; padding: 20px; margin-top: 20px; min-height: 350px; }
#gallery .thumbnail-item { border-radius: 6px !important; overflow: hidden !important; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: transform 0.2s ease; }
#gallery .thumbnail-item:hover { transform: scale(1.03); }
#gallery .thumbnail-item img { object-fit: contain !important; display: block; }
.gr-dataframe { border: 1px solid #e1e4e8; border-radius: 8px; overflow: hidden; margin-top: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.gr-dataframe table { width: 100%; border-collapse: collapse; }
.gr-dataframe th {
    background-color: #f6f8fa !important; 
    color: #24292e !important; 
    font-weight: 600 !important; 
    padding: 14px 18px !important; 
    text-align: left !important;
    font-size: 1.05em !important;
    border-bottom: 1px solid #dfe2e5 !important; 
}
.gr-dataframe td {
    padding: 12px 18px !important; 
    border-bottom: 1px solid #eaecef !important; 
    font-size: 1em !important;
    color: #444; 
}
.gr-dataframe tr:nth-child(even) { background-color: #fcfcfd !important; } 
.gr-dataframe tr:hover { background-color: #f1f8ff !important; } 
.gr-number, .gr-textbox[aria-label="系统状态"] {
    background-color: #f6f8fa !important; 
    border: 1px solid #e1e4e8 !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    margin-top: 5px; 
}
.gr-number input, .gr-textbox[aria-label="系统状态"] textarea { 
    font-size: 1.1em !important;
    font-weight: bold !important;
    color: #2c3e50 !important;
    background-color: transparent !important; 
    border: none !important; 
    padding: 0 !important; 
}
.gr-textbox[aria-label="系统状态"] textarea { font-weight: normal !important; color: #555 !important; }
.gr-row { margin-bottom: 20px; } 
.gr-column { padding: 0 12px; } 
.gr-group { border: 1px solid #e1e4e8; border-radius: 10px; padding: 25px; margin-bottom: 25px; background-color: #ffffff; box-shadow: 0 3px 6px rgba(0,0,0,0.05); }
.st-tabs, .gr-tabs { margin-top: 20px; border: none; border-radius: 8px; overflow: hidden; }
.st-tabs button, .gr-tabs button { font-size: 1.1em !important; padding: 14px 22px !important; border-bottom: 3px solid transparent !important; transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease; color: #586069 !important; background-color: #f6f8fa !important; margin-right: 2px; }
.st-tabs button.selected, .gr-tabs button.selected { font-weight: 600 !important; color: #0366d6 !important; border-bottom-color: #0366d6 !important; background-color: #ffffff !important; }
.tabitem, .gr-tabitem { padding: 25px !important; background-color: #ffffff; border: 1px solid #e1e4e8; border-top: none; border-radius: 0 0 8px 8px; }
.gr-file { margin-top: 15px; } 
.gr-file a { font-size: 1.1em; color: #0366d6; text-decoration: none; font-weight: 500; padding: 8px 12px; border: 1px solid #e1e4e8; border-radius: 6px; background-color: #f6f8fa; display: inline-block; transition: background-color 0.2s ease; }
.gr-file a:hover { text-decoration: none; background-color: #eef4fc; }
#ai_summary_output .gr-markdown { /* 针对AI摘要的特定样式 */
    background-color: #f0f4f8; /* 淡蓝色背景 */
    border-left: 5px solid #3498db; /* 左侧蓝色边框 */
    padding: 15px;
    border-radius: 5px;
    line-height: 1.6;
}
#ai_summary_output .gr-markdown h4 { /* AI摘要中的小标题 */
    color: #2980b9;
    margin-top: 10px;
    margin-bottom: 5px;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue)) as demo:
    gr.Markdown("# 📈 DESR 可信股票预测与模拟交易系统 (智能AI版)")

    save_dir_state = gr.State(value=SAVE_DIR)
    temp_csv_state = gr.State(value=None)

    with gr.Group():
        gr.Markdown("### 1. 选择股票与日期范围")
        with gr.Row(equal_height=True):
            ticker_input = gr.Textbox(label="股票代码", placeholder="例如: AAPL, GOOGL, MSFT", scale=2)
            start_date_input = gr.Textbox(
                label="开始日期",
                value=(datetime.now().replace(year=datetime.now().year - 4)).strftime('%Y-%m-%d'),
                placeholder="YYYY-MM-DD", scale=2
            )
            end_date_input = gr.Textbox(
                label="结束日期",
                value=datetime.now().strftime('%Y-%m-%d'),
                placeholder="YYYY-MM-DD", scale=2
            )
            fetch_button = gr.Button("📊 获取数据", variant="primary", scale=1)

    with gr.Row():
        status_output = gr.Textbox(label="系统状态", interactive=False, placeholder="等待操作...", lines=1)
    with gr.Row():
        data_file_output = gr.File(label="下载处理后的股票数据", visible=False, interactive=False)

    with gr.Group():
        gr.Markdown("### 2. 配置 DESR 训练与预测参数")
        with gr.Row():
            egru_seq_length = gr.Slider(minimum=10, maximum=120, value=16, step=1,
                                        label="序列长度 (Lookback)", interactive=True)
            egru_hidden_dim = gr.Slider(minimum=16, maximum=256, value=64, step=8,
                                        label="隐藏层维度 (Hidden Dim)", interactive=True)
        with gr.Row():
            egru_epochs = gr.Slider(minimum=5, maximum=200, value=50, step=5,
                                    label="训练轮数 (Epochs)", interactive=True)
            egru_batch_size = gr.Slider(minimum=8, maximum=128, value=32, step=8,
                                        label="批次大小 (Batch Size)", interactive=True)
        with gr.Row():
            egru_learning_rate = gr.Number(minimum=1e-5, maximum=1e-2, value=0.001,
                                           label="学习率 (Learning Rate)", interactive=True)

    with gr.Row(elem_classes="flex justify-center"):
        run_button = gr.Button("🚀 运行 DESR 预测、模拟交易与AI分析", variant="primary", interactive=False)

    with gr.Group():
        gr.Markdown("### 3. 分析结果展示")
        with gr.Tabs():
            with gr.TabItem("图表结果"):
                output_gallery = gr.Gallery(
                    label="可视化结果 (预测图 + 不确定性)",
                    show_label=True, elem_id="gallery", columns=[1],
                    height=500, object_fit="contain", preview=True, allow_preview=True
                )
            with gr.TabItem("指标总结与AI分析"):  # 合并标签页或保持独立
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("#### DESR 预测指标")
                        rmse_output = gr.Number(label="RMSE (价格预测)", precision=4, interactive=False)
                        mae_output = gr.Number(label="MAE (价格预测)", precision=4, interactive=False)
                        dir_acc_output = gr.Number(label="方向准确率 (%)", precision=2, interactive=False)

                        gr.Markdown("#### 简单模拟交易指标")
                        profit_output = gr.Number(label="总收益 ($)", precision=2, interactive=False)
                        return_output = gr.Number(label="投资回报率 (%)", precision=2, interactive=False)
                        buy_trades_output = gr.Number(label="买入次数", precision=0, interactive=False)
                        sell_trades_output = gr.Number(label="卖出次数", precision=0, interactive=False)

                    with gr.Column(scale=2):  # AI摘要区域，给予更多空间
                        gr.Markdown("#### 🤖 AI 投资分析摘要")
                        # 使用 gr.Markdown 来更好地格式化AI输出的文本
                        ai_summary_output = gr.Markdown(label="AI 分析与建议", value="AI 分析结果将在此显示...",
                                                        elem_id="ai_summary_output")

            with gr.TabItem("交易记录 (简单模拟)"):
                transactions_df_output = gr.DataFrame(
                    headers=["Date", "Action", "Price", "Shares", "Balance"],
                    label="交易详细记录", row_count=(10, "dynamic"), col_count=(5, "dynamic"),
                    wrap=True, interactive=False
                )


    def update_interface(csv_path):
        file_exists = bool(csv_path and os.path.exists(csv_path))
        print(f"Update interface: CSV path='{csv_path}', Exists={file_exists}")
        return {
            data_file_output: gr.update(value=csv_path if file_exists else None, visible=file_exists),
            run_button: gr.update(interactive=file_exists),
            status_output: gr.update(value=f"{'数据准备就绪' if file_exists else '请先获取数据'}。")
        }


    fetch_button.click(
        fn=get_data,
        inputs=[ticker_input, start_date_input, end_date_input],
        outputs=[temp_csv_state, status_output]
    ).then(
        fn=update_interface,
        inputs=[temp_csv_state],
        outputs=[data_file_output, run_button, status_output]
    )

    run_button.click(
        fn=run_desr_pipeline_interface,
        inputs=[
            temp_csv_state,
            egru_seq_length,
            egru_hidden_dim,
            egru_epochs,
            egru_batch_size,
            egru_learning_rate,
            save_dir_state
        ],
        outputs=[  # 对应 run_desr_pipeline_interface 返回的 10 个值
            output_gallery,
            rmse_output,
            mae_output,
            dir_acc_output,
            profit_output,
            return_output,
            buy_trades_output,
            sell_trades_output,
            transactions_df_output,
            ai_summary_output  # 新增 AI 摘要输出
        ]
    )

if __name__ == "__main__":
    print("-" * 50)
    print("启动 Gradio 应用 (DESR 预测、模拟交易与AI分析)...")
    print(f"使用的设备: {device}")
    print(f"结果保存目录: {SAVE_DIR}")
    required_funcs = ['get_stock_data', 'clean_csv_files', 'run_stock_prediction_egru']
    missing_funcs = [func for func in required_funcs if func not in globals()]
    if missing_funcs:
        print(f"\n错误：启动前检测到缺少必要的函数: {', '.join(missing_funcs)}")
        print("请确保 process_stock_data.py 和 stock_prediction_egru.py 文件有效且位于正确路径。\n")
    print("-" * 50)

    try:
        demo.launch(server_port=7860, share=False, inbrowser=True, prevent_thread_lock=True)
    except OSError as e:
        if "Cannot find empty port" in str(e) or "address already in use" in str(e).lower():
            print(f"\n错误：端口 7860 已被占用。")
            print(f"请尝试关闭使用该端口的其他程序，或者修改代码中的 server_port 为其他可用端口 (例如 7861)。\n")
        else:
            print(f"启动 Gradio 时发生 OS 错误: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"启动 Gradio 时发生未知错误: {e}")
        traceback.print_exc()
    demo.launch() # 移除了这句重复的launch
