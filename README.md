# DESR 可信股票预测与模拟交易系统 (Stock Trading AI)

基于 Evidence GRU (EGRU) 模型与不确定性量化技术的股票预测与模拟交易系统。该系统不仅能预测股票价格走势，还能评估预测结果的置信度（不确定性），并结合这些指标自动进行模拟交易回测，最终通过直观的 Web 界面呈现所有分析结果。

## 项目特点

* **引入不确定性量化**：采用 EGRU 模型结合 Normal Inverse Gamma (NIG) 证据层，在预测价格的同时输出预测的不确定性区间（Aleatoric Uncertainty）。
* **自动化模拟交易**：基于模型预测结果执行简单的模拟交易策略，自动计算总收益、投资回报率、胜率等关键交易指标。
* **一站式交互界面**：使用 Gradio 打造了用户友好的 Web UI，支持在线获取数据、动态调参、一键训练与可视化分析。
* **AI 智能分析摘要**：接入 Gemini API。系统会根据生成的预测精度和交易回测指标，自动生成结构化的 AI 投资参考摘要。
* **完整的数据流水线**：内置从 Yahoo Finance 自动下载数据、缺失值处理到多维度技术指标（MA, RSI, MACD, Bollinger Bands 等）计算的完整特征工程。

## 环境要求

* Python 3.8+ (推荐 3.10+)
* PyTorch (强烈推荐配置 CUDA 以支持 GPU 加速)
* Gradio (用于构建 Web 交互界面)
* 其它依赖项请根据实际运行环境补充安装

## 安装与运行

1. 克隆项目到本地:

    git clone https://github.com/你的用户名/你的仓库名.git
    cd 你的仓库名

2. 安装所需依赖 (推荐在虚拟环境中进行):

    pip install pandas numpy yfinance scikit-learn torch matplotlib gradio

3. 启动可视化系统:

    python main.py

运行后，控制台会输出一个本地局域网地址（通常为 http://127.0.0.1:7860 ）。在浏览器中打开该地址，即可进入交互界面。

## 项目结构

    DESR可信预测网站/
    ├── data/                       # 存储股票历史数据缓存 (如 AAPL.csv)
    ├── results/                    # 存储模型训练后的权重与结果图表
    ├── main.py                     # Gradio Web 界面主程序 (系统启动入口)
    ├── stock_prediction_egru.py    # EGRU 模型训练、预测评估及模拟交易核心流水线
    ├── model.py                    # EGRU 深度学习网络模型与 NIG 证据层定义
    ├── loss.py                     # 针对 NIG 分布的负对数似然 (NLL) 损失函数
    ├── process_stock_data.py       # 股票数据自动化获取与技术指标特征工程模块
    ├── pyproject.toml              # 项目配置文件
    └── README.md                   # 项目说明文档

## 核心模块解析

1. **数据处理 (process_stock_data.py)**
   * 自动拉取股票日线数据。
   * 自动生成丰富的技术分析特征（如相对强弱指数 RSI、平滑异同移动平均线 MACD、成交量加权平均价 VWAP 等）。
   * 数据归一化及前向/后向缺失值填充。

2. **可信预测网络 (model.py & loss.py)**
   * 构建基于 GRU 架构的时序特征提取器。
   * 通过 NormalInvGamma 自定义层输出目标分布的参数 (μ, ν, α, β)。
   * 使用自定义的 criterion_nig 损失函数优化模型，使得模型在预测错误时具有较高的不确定性输出。

3. **流水线与交易引擎 (stock_prediction_egru.py)**
   * 封装 PyTorch 的 DataLoader 进行批量训练。
   * 计算预测均值，并利用网络输出推导 95% 置信区间。
   * 内置轻量级回测引擎，基于 T+1 预测生成 Buy/Hold/Sell 信号，记录资金曲线与交易流水。

## 备注

* 如果遇到网络问题导致无法下载股票数据，请检查本地网络设置或配置相应的代理，也可以手动将合规的 CSV 文件放入 data 文件夹中。
* EGRU 模型的训练涉及大量的张量运算，建议在配置了 NVIDIA 显卡及对应 CUDA 环境的计算机上运行，以显著缩短训练时间。

## 联系方式

本项目在持续优化中。如有任何想法、建议或发现了 Bug，欢迎在 GitHub 的 Issues 或 Discussions 中提出，也非常欢迎提交 PR 共同改进！