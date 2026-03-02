import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import matplotlib.pyplot as plt
import traceback # 导入 traceback

# 导入你的 EGRU 模型和 NIG 损失函数
try:
    from model import EGRU # 假设 model.py 在同一目录下或 PYTHONPATH 中
    from loss import criterion_nig # 假设 loss.py 在同一目录下或 PYTHONPATH 中
except ImportError as e:
     print(f"错误：无法导入 EGRU 或 criterion_nig: {e}")
     EGRU = None
     criterion_nig = None

# --- 方向准确率计算函数 ---
def calculate_directional_accuracy(actual, predicted):
    """
    计算预测方向的准确率。
    比较实际价格变动方向与预测价格变动方向是否一致。
    Args:
        actual (np.array): 实际价格序列。
        predicted (np.array): 预测价格序列。
    Returns:
        float: 方向准确率 (0 到 1 之间)。
    """
    if len(actual) < 2 or len(predicted) < 2:
        return 0.0 # 数据点不足无法计算方向

    # 计算实际价格变动方向 (与前一天比较)
    actual_diff = np.diff(actual)
    actual_direction = np.sign(actual_diff) # +1 表示上涨, -1 表示下跌, 0 表示不变

    # 计算预测价格变动方向 (与前一天比较)
    predicted_diff = np.diff(predicted)
    predicted_direction = np.sign(predicted_diff)

    # 确保比较的长度一致
    min_len = min(len(actual_direction), len(predicted_direction))
    actual_direction = actual_direction[:min_len]
    predicted_direction = predicted_direction[:min_len]

    # 比较方向是否一致 (忽略方向为 0 的情况，避免除零)
    # 只比较 actual_direction 不为 0 的情况
    valid_comparison_mask = actual_direction != 0
    if np.sum(valid_comparison_mask) == 0:
        return 0.0 # 如果实际价格一直不变，无法计算方向准确率

    correct_direction = np.sum((actual_direction[valid_comparison_mask] == predicted_direction[valid_comparison_mask]))
    total_comparisons = np.sum(valid_comparison_mask)

    accuracy = correct_direction / total_comparisons
    return accuracy

# --- 数据处理函数 (保持你之前的版本) ---
def load_and_preprocess_data(file_path, sequence_length, target_column='Close'):
    """
    加载数据，进行预处理，并创建序列。
    Args:
        file_path (str): CSV 文件路径。
        sequence_length (int): 用于预测的序列长度。
        target_column (str): 目标预测列名。
    Returns:
        tuple: 包含 X_train, y_train, X_test, y_test, scaler, train_df, test_df, feature_columns, target_col_index
               或者如果数据不足/出错，返回 None。
    """
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df.set_index('Date') # 将 Date 设置为索引

        # 选择特征列 (例如: Open, High, Low, Close, Volume)
        # *** 确保这里的特征与 EGRU 的 input_dim 匹配 ***
        feature_columns = ['Open', 'High', 'Low', target_column, 'Volume'] # 保持你的版本
        if not all(col in df.columns for col in feature_columns):
             print(f"警告: 文件 {file_path} 缺少必要的列。跳过。")
             missing = [col for col in feature_columns if col not in df.columns]
             print(f"缺失列: {missing}")
             # 可以选择只使用存在的列，但这需要调整 input_dim
             # feature_columns = [col for col in feature_columns if col in df.columns]
             # if len(feature_columns) < 2: # 至少需要目标列和一个特征
             return None

        data = df[feature_columns].values.astype(float)

        # 处理 NaN 值 (例如，用前一个值填充) - 添加更鲁棒的处理
        if np.isnan(data).any():
            print(f"警告: 文件 {file_path} 包含 NaN 值，将使用前向填充处理。")
            data = pd.DataFrame(data, columns=feature_columns).ffill().bfill().values # 先前向填充，再后向填充处理开头的 NaN
            if np.isnan(data).any():
                 print(f"错误: 文件 {file_path} 在填充后仍包含 NaN 值。跳过。")
                 return None

        # 数据标准化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # 分割训练集和测试集 (例如 80/20)
        training_data_len = int(np.ceil(len(scaled_data) * 0.8))
        if training_data_len <= sequence_length or len(scaled_data) - training_data_len <= sequence_length:
            print(f"警告: 文件 {file_path} 数据量 ({len(scaled_data)}) 不足以创建训练/测试序列 (需要 > {sequence_length} 条)。跳过。")
            return None

        train_data = scaled_data[0:int(training_data_len), :]
        test_data = scaled_data[training_data_len - sequence_length:, :] # 测试集需要包含训练集末尾的序列长度数据

        # 创建训练序列
        X_train, y_train = [], []
        target_col_index = feature_columns.index(target_column)
        for i in range(sequence_length, len(train_data)):
            X_train.append(train_data[i-sequence_length:i, :]) # 使用所有特征作为输入
            y_train.append(train_data[i, target_col_index]) # 预测目标列
        X_train, y_train = np.array(X_train), np.array(y_train)

        # 创建测试序列
        X_test, y_test = [], []
        for i in range(sequence_length, len(test_data)):
            X_test.append(test_data[i-sequence_length:i, :])
            y_test.append(test_data[i, target_col_index])
        X_test, y_test = np.array(X_test), np.array(y_test)

        # 获取对应的原始 DataFrame 部分用于后续分析
        train_df = df[:training_data_len]
        test_df = df[training_data_len:]

        # 确保 test_df 长度足够覆盖 y_test 对应的日期
        # y_test 的第一个点对应 test_df 的第 sequence_length 个点
        if len(test_df) < len(y_test):
             print(f"警告: test_df 长度 ({len(test_df)}) 小于 y_test 长度 ({len(y_test)})。可能导致索引错误。将截断测试数据。")
             y_test = y_test[:len(test_df)]
             X_test = X_test[:len(test_df)]


        return X_train, y_train, X_test, y_test, scaler, train_df, test_df, feature_columns, target_col_index

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"加载和预处理文件 {file_path} 时出错: {e}")
        traceback.print_exc()
        return None

# --- EGRU 模型训练函数 (保持你之前的版本) ---
def train_egru_model(X_train, y_train, input_dim, hidden_dim, output_dim, sequence_length, num_epochs=50, batch_size=32, learning_rate=0.001, device='cpu', model_save_path='egru_model.pth'):
    """训练 EGRU 模型。"""
    if EGRU is None or criterion_nig is None:
         print("错误: EGRU 模型或损失函数未成功导入，无法训练。")
         return None, None
    # 转换数据为 Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device) # 增加一个维度以匹配输出

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = EGRU(hidden_dim=hidden_dim, seq_length=sequence_length, device=device, input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = criterion_nig # 不需要实例化，直接引用函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    history = {'loss': []}
    print("开始训练 DESR 模型...") # 保持你的命名
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            nig_params = model(batch_X)
            nig_params_last_step = nig_params[:, -1, :]
            loss = criterion(nig_params_last_step, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_epoch_loss)
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1: # 在最后也打印
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}')

    print("训练完成。")

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到 {model_save_path}")

    return model, history

# --- EGRU 模型预测函数 (保持你之前的版本) ---
def predict_with_egru(model, X_test, device='cpu'):
    """使用训练好的 EGRU 模型进行预测。"""
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        nig_params_full = model(X_test_tensor)
        nig_params_last_step = nig_params_full[:, -1, :]
    return nig_params_last_step.cpu().numpy()

# --- 结果处理和评估 (修改以包含方向准确率) ---
def evaluate_predictions(predictions_nig, y_test, scaler, target_col_index):
    """
    评估预测结果并进行反标准化。
    Args:
        predictions_nig (np.array): 模型的 NIG 参数输出 (mu, v, alpha, beta)。
        y_test (np.array): 真实的测试目标值 (标准化的)。
        scaler (MinMaxScaler): 用于反标准化的 scaler。
        target_col_index (int): 目标列在 scaler 中的索引。
    Returns:
        tuple: 包含反标准化后的真实值、预测值 (mu)、评估指标字典和置信区间。
    """
    # 提取预测均值 mu (标准化)
    predicted_mu_scaled = predictions_nig[:, 0]

    # --- 反标准化 ---
    num_features = scaler.n_features_in_
    dummy_array_pred = np.zeros((len(predicted_mu_scaled), num_features))
    dummy_array_pred[:, target_col_index] = predicted_mu_scaled
    predicted_mu_rescaled = scaler.inverse_transform(dummy_array_pred)[:, target_col_index]

    dummy_array_true = np.zeros((len(y_test), num_features))
    dummy_array_true[:, target_col_index] = y_test
    y_test_rescaled = scaler.inverse_transform(dummy_array_true)[:, target_col_index]

    # --- 计算评估指标 ---
    # 清理 NaN 值以进行计算
    valid_mask = ~np.isnan(y_test_rescaled) & ~np.isnan(predicted_mu_rescaled)
    y_test_rescaled_clean = y_test_rescaled[valid_mask]
    predicted_mu_rescaled_clean = predicted_mu_rescaled[valid_mask]

    metrics = { # 初始化指标字典
        'MSE': None, 'RMSE': None, 'MAE': None, 'R2 Score': None,
        'Directional Accuracy': None # 添加方向准确率键
    }

    if len(y_test_rescaled_clean) >= 2: # 需要至少两个点来计算指标
        mse = mean_squared_error(y_test_rescaled_clean, predicted_mu_rescaled_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_rescaled_clean, predicted_mu_rescaled_clean)
        r2 = r2_score(y_test_rescaled_clean, predicted_mu_rescaled_clean)
        # *** 计算方向准确率 ***
        dir_accuracy = calculate_directional_accuracy(y_test_rescaled_clean, predicted_mu_rescaled_clean)

        metrics['MSE'] = mse
        metrics['RMSE'] = rmse
        metrics['MAE'] = mae
        metrics['R2 Score'] = r2
        metrics['Directional Accuracy'] = dir_accuracy # 存入字典
        print(f"评估指标: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, DirAcc={dir_accuracy:.4f}")
    else:
        print("警告：有效数据点不足 (<2)，无法计算评估指标。")


    # --- 计算不确定性 (置信区间) ---
    mu_scaled = predictions_nig[:, 0][valid_mask] # 只使用有效部分计算置信区间
    v_scaled = predictions_nig[:, 1][valid_mask]
    alpha_scaled = predictions_nig[:, 2][valid_mask]
    beta_scaled = predictions_nig[:, 3][valid_mask]

    # 初始化置信区间为 NaN 数组，长度与原始预测一致
    lower_bound_rescaled = np.full_like(predicted_mu_rescaled, np.nan)
    upper_bound_rescaled = np.full_like(predicted_mu_rescaled, np.nan)

    if len(mu_scaled) > 0: # 确保有有效数据
        epsilon = 1e-6
        alpha_safe = np.maximum(alpha_scaled, 1.0 + epsilon)
        # 使用 Aleatoric 不确定性: var = beta / (alpha - 1)
        variance_scaled = beta_scaled / (alpha_safe - 1 + epsilon)
        variance_scaled = np.maximum(0, variance_scaled) # 确保非负
        std_dev_scaled = np.sqrt(variance_scaled)

        # 反标准化置信区间边界 (只对有效部分进行)
        dummy_array_upper = np.zeros((len(mu_scaled), num_features))
        dummy_array_lower = np.zeros((len(mu_scaled), num_features))
        dummy_array_upper[:, target_col_index] = mu_scaled + 1.96 * std_dev_scaled # 95% CI
        dummy_array_lower[:, target_col_index] = mu_scaled - 1.96 * std_dev_scaled

        upper_bound_rescaled_clean = scaler.inverse_transform(dummy_array_upper)[:, target_col_index]
        lower_bound_rescaled_clean = scaler.inverse_transform(dummy_array_lower)[:, target_col_index]
        lower_bound_rescaled_clean = np.maximum(lower_bound_rescaled_clean, 0) # 价格不能为负

        # 将计算出的置信区间放回完整数组的对应位置
        lower_bound_rescaled[valid_mask] = lower_bound_rescaled_clean
        upper_bound_rescaled[valid_mask] = upper_bound_rescaled_clean


    confidence_intervals = {
        'lower': lower_bound_rescaled, # 返回与原始预测长度一致的数组 (可能含 NaN)
        'upper': upper_bound_rescaled
    }

    # 返回反标准化后的完整序列 (可能含 NaN) 和指标字典
    return y_test_rescaled, predicted_mu_rescaled, metrics, confidence_intervals

# --- 主函数 (整合指标并返回) ---
def run_stock_prediction_egru(file_path, output_dir, sequence_length=60, hidden_dim=64, num_epochs=50, batch_size=32, learning_rate=0.001):
    """
    运行完整的 EGRU 股票预测流程，并增加模拟交易指标。
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载和预处理数据
    preprocess_result = load_and_preprocess_data(file_path, sequence_length)
    if preprocess_result is None:
        print(f"数据预处理失败: {file_path}")
        return None
    X_train, y_train, X_test, y_test, scaler, train_df, test_df, feature_columns, target_col_index = preprocess_result

    input_dim = X_train.shape[2]
    output_dim = 1

    # 训练模型
    model_filename = os.path.basename(file_path).replace('.csv', f'_egru_{sequence_length}seq_{hidden_dim}hd.pth')
    model_save_path = os.path.join(output_dir, model_filename)
    model, history = train_egru_model(
        X_train, y_train, input_dim, hidden_dim, output_dim,
        sequence_length, num_epochs, batch_size, learning_rate, device, model_save_path
    )
    if model is None:
         print("模型训练失败。")
         return None

    # 进行预测
    predictions_nig = predict_with_egru(model, X_test, device)

    # 评估基础指标 (现在包含方向准确率)
    y_test_rescaled, predicted_mu_rescaled, metrics, confidence_intervals = evaluate_predictions(predictions_nig, y_test, scaler, target_col_index)

    # 构建预测结果DataFrame (使用你提供的 min_len 逻辑)
    # 确定对齐的日期索引
    # test_df 的索引对应原始完整数据
    # y_test_rescaled 和 predicted_mu_rescaled 的长度是 len(X_test)
    # 它们对应 test_df 从第 sequence_length 个点开始的数据
    start_index_in_test_df = sequence_length
    num_predictions = len(y_test_rescaled)

    if len(test_df) >= start_index_in_test_df + num_predictions:
        prediction_dates = test_df.index[start_index_in_test_df : start_index_in_test_df + num_predictions]
        # 确保 confidence_intervals 长度也正确
        lower_bound = confidence_intervals['lower'][:num_predictions]
        upper_bound = confidence_intervals['upper'][:num_predictions]
    else:
        # 如果 test_df 不够长 (可能在预处理中被截断)，则无法安全对齐
        print(f"警告: test_df 长度 ({len(test_df)}) 不足以覆盖所有预测 ({num_predictions})。将使用数字索引。")
        # 创建一个包含所有数据的 DataFrame，但不设置日期索引
        results_df = pd.DataFrame({
            'Actual': y_test_rescaled,
            'Predicted': predicted_mu_rescaled,
            'Lower_Bound': confidence_intervals['lower'],
            'Upper_Bound': confidence_intervals['upper']
        })
        results_df.index.name = 'Index'
        prediction_dates = None # 标记日期未对齐

    if prediction_dates is not None:
        results_df = pd.DataFrame({
            'Actual': y_test_rescaled,
            'Predicted': predicted_mu_rescaled,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        }, index=prediction_dates) # 使用日期作为索引

    # --- 执行你添加的简单模拟交易 ---
    # 使用 results_df 进行交易模拟，需要确保它有 'Actual' 和 'Predicted' 列
    initial_money = 10000
    inventory = []
    balance = initial_money
    buy_times = 0
    sell_times = 0
    transaction_log = [] # 用于记录交易详情

    # 只在 results_df 有效时进行模拟
    if 'results_df' in locals() and not results_df.empty and 'Actual' in results_df and 'Predicted' in results_df:
        # 迭代 DataFrame 的行
        for date, row in results_df.iterrows():
            actual_price = row['Actual']
            pred_price = row['Predicted']

            # 跳过 NaN 值
            if pd.isna(actual_price) or pd.isna(pred_price):
                continue

            # 简单交易逻辑 (与你之前版本一致)
            # 注意：这里的逻辑是基于当天预测和当天实际价格，可能存在未来信息泄露
            # 更实际的模拟应该基于对下一天的预测和当天的价格来决策
            if pred_price > actual_price and balance >= actual_price:
                inventory.append(actual_price)
                balance -= actual_price
                buy_times += 1
                transaction_log.append({'Date': date, 'Action': 'Buy', 'Price': actual_price, 'Shares': 1, 'Balance': balance})
            elif pred_price < actual_price and len(inventory) > 0:
                bought_price = inventory.pop(0) # FIFO
                balance += actual_price
                sell_times += 1
                transaction_log.append({'Date': date, 'Action': 'Sell', 'Price': actual_price, 'Shares': 1, 'Balance': balance})

        # 计算最终持仓价值 (假设以最后一天的实际价格卖出)
        if not results_df.empty:
             last_actual_price = results_df['Actual'].iloc[-1]
             final_balance = balance + len(inventory) * last_actual_price
        else:
             final_balance = balance # 如果没有交易日，最终余额就是初始余额

        total_profit = final_balance - initial_money
        return_rate = (total_profit / initial_money) * 100 if initial_money != 0 else 0
    else:
        print("警告：无法执行模拟交易，因为结果 DataFrame 无效或缺少列。")
        total_profit, return_rate, buy_times, sell_times, final_balance = None, None, 0, 0, initial_money
        transaction_log = [] # 确保为空列表

    # 将交易指标加入 metrics 字典
    metrics['Total Profit ($)'] = total_profit # 使用更清晰的键名
    metrics['Return Rate (%)'] = return_rate
    metrics['Buy Times'] = buy_times
    metrics['Sell Times'] = sell_times
    # metrics['Final Balance ($)'] = final_balance # 可以选择性添加

    # 创建交易日志 DataFrame
    transaction_log_df = pd.DataFrame(transaction_log)
    if not transaction_log_df.empty:
        transaction_log_df = transaction_log_df.set_index('Date')


    # --- 保存文件 ---
    base_filename = os.path.basename(file_path).replace('.csv', '')
    metrics_path = os.path.join(output_dir, f"{base_filename}_prediction_metrics.csv")
    summary_path = os.path.join(output_dir, f"{base_filename}_prediction_summary.txt")
    plot_path = os.path.join(output_dir, f"{base_filename}_prediction_plot.png")
    results_csv_path = os.path.join(output_dir, f"{base_filename}_predictions.csv")
    transactions_csv_path = os.path.join(output_dir, f"{base_filename}_transactions_simple.csv") # 保存交易记录

    # 保存指标 (处理 None 值)
    metrics_to_save = {k: (f"{v:.4f}" if isinstance(v, (float, np.number)) and not pd.isna(v) else (v if v is not None else "N/A")) for k, v in metrics.items()}
    pd.DataFrame([metrics_to_save]).to_csv(metrics_path, index=False)
    # 保存预测结果
    if 'results_df' in locals():
        results_df.to_csv(results_csv_path)
    # 保存交易记录
    transaction_log_df.to_csv(transactions_csv_path)

    # 生成摘要
    summary = f"DESR 模型预测与简单交易模拟摘要 for {base_filename}\n" # 更新标题
    summary += "=" * 30 + "\n"
    summary += f"序列长度: {sequence_length}, 隐藏维度: {hidden_dim}, Epochs: {num_epochs}\n"
    summary += f"输入特征: {feature_columns}\n"
    summary += f"使用的设备: {device}\n"
    summary += f"模型保存在: {model_save_path}\n"
    summary += "\n评估指标:\n"
    for key, value in metrics_to_save.items():
        # 对特定指标格式化
        if key == 'Directional Accuracy':
             summary += f"- {key}: {float(value)*100:.2f}% (预测方向准确率)\n" if value != "N/A" else f"- {key}: N/A\n"
        elif key == 'Return Rate (%)':
             summary += f"- {key}: {float(value):.2f}%\n" if value != "N/A" else f"- {key}: N/A\n"
        elif 'Times' in key:
             summary += f"- {key}: {int(value)}\n" if value != "N/A" else f"- {key}: N/A\n"
        elif 'Profit' in key or 'Balance' in key:
             summary += f"- {key}: ${float(value):,.2f}\n" if value != "N/A" else f"- {key}: N/A\n"
        else:
             summary += f"- {key}: {value}\n"

    summary += f"\n结果保存在:\n- 指标: {metrics_path}\n- 预测: {results_csv_path}\n- 交易: {transactions_csv_path}\n- 图表: {plot_path}\n- 摘要: {summary_path}\n"

    with open(summary_path, 'w') as f:
        f.write(summary)

    # 绘制图表 (使用清理后的数据)
    if 'results_df' in locals() and not results_df.empty:
        valid_mask_plot = ~results_df['Actual'].isna() & ~results_df['Predicted'].isna()
        results_df_clean = results_df[valid_mask_plot]

        if not results_df_clean.empty:
            plt.figure(figsize=(14, 7))
            plt.plot(results_df_clean.index, results_df_clean['Actual'], label='实际价格', color='blue', linewidth=1.5)
            plt.plot(results_df_clean.index, results_df_clean['Predicted'], label='预测价格 (DESR)', color='red', linestyle='--')
            # 绘制置信区间 (如果存在且长度匹配)
            if 'Lower_Bound' in results_df_clean and 'Upper_Bound' in results_df_clean and \
               not results_df_clean['Lower_Bound'].isna().all() and not results_df_clean['Upper_Bound'].isna().all():
                 plt.fill_between(results_df_clean.index, results_df_clean['Lower_Bound'], results_df_clean['Upper_Bound'],
                                  color='red', alpha=0.2, label='95% 置信区间')

            # 在图上标记买卖点 (可选，但需要 transaction_log_df 有日期索引)
            if not transaction_log_df.empty and isinstance(transaction_log_df.index, pd.DatetimeIndex):
                 buy_points = transaction_log_df[transaction_log_df['Action'] == 'Buy']
                 sell_points = transaction_log_df[transaction_log_df['Action'] == 'Sell']
                 if not buy_points.empty:
                     plt.scatter(buy_points.index, buy_points['Price'], label='买入点', marker='^', color='lime', s=60, zorder=5, edgecolors='black')
                 if not sell_points.empty:
                     plt.scatter(sell_points.index, sell_points['Price'], label='卖出点', marker='v', color='magenta', s=60, zorder=5, edgecolors='black')


            # 更新标题以包含新指标
            title_str = f'{base_filename} - DESR 预测与简单交易模拟'
            title_metrics = []
            if metrics.get('RMSE') is not None: title_metrics.append(f'RMSE: {metrics["RMSE"]:.2f}')
            if metrics.get('MAE') is not None: title_metrics.append(f'MAE: {metrics["MAE"]:.2f}')
            if metrics.get('Directional Accuracy') is not None: title_metrics.append(f'DirAcc: {metrics["Directional Accuracy"]:.2%}')
            if metrics.get('Return Rate (%)') is not None: title_metrics.append(f'回报率: {metrics["Return Rate (%)"]:.2f}%')
            title_str += '\n' + ', '.join(title_metrics)
            plt.title(title_str)

            plt.xlabel('日期')
            plt.ylabel('价格')
            plt.legend()
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"预测与交易图已保存到: {plot_path}")
        else:
            print("警告：没有有效的绘图数据。")
            plot_path = None
    else:
         print("警告：无法绘制图表，因为结果 DataFrame 无效。")
         plot_path = None


    print(f"预测与模拟完成。结果保存在 {output_dir}")
    print(f"最终指标: {metrics_to_save}")

    # 返回 Gradio 需要的结果
    # 注意：返回值的顺序和数量需要与 Gradio 接口中的 outputs 列表匹配
    # 假设 Gradio 需要: [gallery_paths, rmse, mae, dir_acc, total_profit, return_rate, buy_times, sell_times, transaction_log_df]
    return [
        [plot_path] if plot_path else [], # 画廊路径列表
        metrics.get('RMSE'),
        metrics.get('MAE'),
        metrics.get('Directional Accuracy'),
        metrics.get('Total Profit ($)'),
        metrics.get('Return Rate (%)'),
        metrics.get('Buy Times'),
        metrics.get('Sell Times'),
        transaction_log_df # 交易日志 DataFrame
    ]

# --- 示例用法 (保持不变) ---
if __name__ == '__main__':
    data_file = 'stock_trading-main/data/AAPL.csv'
    output_directory = 'stock_trading-main/results/output_egru_final' # 使用新目录

    if os.path.exists(data_file):
         run_stock_prediction_egru(
             file_path=data_file,
             output_dir=output_directory,
             sequence_length=60,
             hidden_dim=64,
             num_epochs=50,
             batch_size=32,
             learning_rate=0.001
         )
    else:
         print(f"错误: 示例数据文件未找到: {data_file}")

