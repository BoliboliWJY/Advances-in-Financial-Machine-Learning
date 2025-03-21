#How to run : python -m Part1.CH01.ch01
import polars as pl
import numpy as np
# 绘制时间序列图
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

from variables import TickData_path, TickData_col
from src.data_reader import read_csv

import tqdm

def create_bars(df: pl.DataFrame, bar_type: str, threshold: int) -> pl.DataFrame:
    """根据bar_type和threshold创建bar

    Args:
        df (pl.DataFrame): 读取的tick数据
        bar_type (str): 需要转换的类型
        threshold (int): 阈值

    Returns:
        pl.DataFrame: 转换后的bar数据
    """
    cumsum = 0
    bar_indices = []
    
    if bar_type == "time":
        cumsum = df[:, TickData_col.time_col][0] # 初始时间
        for i, time in enumerate(tqdm.tqdm(df[:, TickData_col.time_col])):
            if time - cumsum >= threshold:
                bar_indices.append(i)
                cumsum = time
    elif bar_type == "volume":
        cumsum = 0
        for i, qty in enumerate(tqdm.tqdm(df[:, TickData_col.qty_col])):
            cumsum += qty
            if cumsum >= threshold:
                bar_indices.append(i)
                cumsum = 0
    elif bar_type == "dollar":
        cumsum = 0
        
        for i, (price, qty) in enumerate(tqdm.tqdm(zip(df[:, TickData_col.price_col], df[:, TickData_col.qty_col]))):
            cumsum += price * qty
            if cumsum >= threshold:
                bar_indices.append(i)
                cumsum = 0
        
    # 确保bar_indices不为空，并检查完成性
    if not bar_indices or bar_indices[-1] != len(df) - 1:
        bar_indices.append(len(df) - 1)
        
    bars = []
    for i in range(len(bar_indices) - 1):
        start_idx = bar_indices[i]
        end_idx = bar_indices[i + 1]
        
        bar_data = df.slice(start_idx, end_idx - start_idx + 1)
        
        bar = {
            "open": bar_data[:, TickData_col.price_col][0],
            "high": bar_data[:, TickData_col.price_col].max(),
            "low": bar_data[:, TickData_col.price_col].min(),
            "close": bar_data[:, TickData_col.price_col][-1],
            "qty": bar_data[:, TickData_col.qty_col].sum(),
            "dollar": (bar_data[:, TickData_col.qty_col] * bar_data[:, TickData_col.price_col]).sum(),
            "timestamp": bar_data[:, TickData_col.time_col][0],
        }
        bars.append(bar)
        
    bars = pl.DataFrame(bars).with_columns(pl.col("timestamp").cast(pl.Datetime(time_unit="ms")).alias("datetime"))

    return pl.DataFrame(bars)

def count_bars(bars: pl.DataFrame, time_interval: str) -> pl.DataFrame:
    """计算每个时间间隔的bar数量

    Args:
        bars (pl.DataFrame): 需要计算的bar数据
        time_interval (int): 时间间隔
    """
    # np_datetime = bars[:, "datetime"].to_numpy()
    # dtype_str = f"timedelta64[{time_interval}]"
    # np_datetime_diff = np.diff(np_datetime).astype(dtype_str).astype(np.int64)
    
    grouped = bars.group_by_dynamic(
        "datetime",
        every=time_interval
    ).agg(
        pl.count().alias("bar_count")
    )
    

    return grouped
    
    

def calculate_jarque_bera(bars: pl.DataFrame) -> float:
    """计算Jarque-Bera正态性检验统计量

    Args:
        bars (pl.DataFrame): 条形图数据

    Returns:
        float: Jarque-Bera检验统计量
    """
    returns = np.diff(np.log(bars["close"]))
    return stats.jarque_bera(returns)[0]



def calculate_autocorrelation(bars, lag):
   # 计算对数收益率
   returns = np.diff(np.log(bars["close"]))
   # 计算指定lag的自相关
   return np.corrcoef(returns[:-lag], returns[lag:])[0,1]
   
def calculate_returns_variance(bars: pl.DataFrame, time_interval: str = "3h") -> dict:
    """计算每3小时子集的收益方差，并计算这些方差的方差

    Args:
        bars (pl.DataFrame): 条形图数据
        time_interval (str): 时间间隔，默认为3小时

    Returns:
        dict: 包含方差统计信息的字典
    """
    # 计算对数收益率
    returns = np.diff(np.log(bars["close"]))
    
    # 按时间间隔分组
    grouped = bars.group_by_dynamic(
        "datetime",
        every=time_interval
    ).agg(
        pl.col("close").pct_change().var().alias("returns_variance")
    )
    
    # 计算方差的统计信息
    variance_stats = {
        "mean_variance": grouped["returns_variance"].mean(),
        "variance_of_variance": grouped["returns_variance"].var(),
        "cv": grouped["returns_variance"].std() / grouped["returns_variance"].mean()
    }
    
    return variance_stats

def create_imbalance_bars(df: pl.DataFrame, bar_type: str, threshold: int) -> pl.DataFrame:
    """根据bar_type和threshold创建imbalance bars

    Args:
        bars (pl.DataFrame): 读取的tick数据
        bar_type (str): 需要转换的类型
        threshold (int): 阈值

    Returns:
        pl.DataFrame: 转换后的imbalance bars数据
    """
    bar_indices = []
    cumsum_imbalance = 0
    if bar_type == "dollar":
        for i, (price, qty) in enumerate(tqdm.tqdm(zip(df[:, TickData_col.price_col], df[:, TickData_col.qty_col]))):
            if i > 0:
                price_diff = price - df[i - 1, TickData_col.price_col]
                imbalance = price_diff * qty * price
                cumsum_imbalance += imbalance
                if cumsum_imbalance >= threshold:
                    bar_indices.append(i)
                    cumsum_imbalance = 0
    
    if not bar_indices or bar_indices[-1] != len(df) - 1:
        bar_indices.append(len(df) - 1)
    
    bars = []
    for i in range(len(bar_indices) - 1):
        start_idx = bar_indices[i]
        end_idx = bar_indices[i + 1]
        
        bar_data = df.slice(start_idx, end_idx - start_idx + 1)
        
        bar = {
            "open": bar_data[:, TickData_col.price_col][0],
            "high": bar_data[:, TickData_col.price_col].max(),
            "low": bar_data[:, TickData_col.price_col].min(),
            "close": bar_data[:, TickData_col.price_col][-1],
            "qty": bar_data[:, TickData_col.qty_col].sum(),
            "dollar": (bar_data[:, TickData_col.qty_col] * bar_data[:, TickData_col.price_col]).sum(),
            "timestamp": bar_data[:, TickData_col.time_col][0],
        }
        
        bars.append(bar)
        
    bars = pl.DataFrame(bars).with_columns(pl.col("timestamp").cast(pl.Datetime(time_unit="ms")).alias("datetime"))

    return pl.DataFrame(bars)

def compute_pca_weights(return_matrix: np.ndarray) -> np.ndarray:
    """计算PCA权重

    Args:
        return_matrix (np.ndarray): 收益率矩阵

    Returns:
        np.ndarray: PCA权重
    """
    from sklearn.decomposition import PCA
    centered_returns = return_matrix - np.mean(return_matrix, axis=0)
    
    pca = PCA(n_components=2)
    pca.fit(centered_returns)
    
    wt = pca.components_[0]
    return wt

def z_score_normalize(data: np.ndarray) -> np.ndarray:
    """
    对输入数据进行 z-score 标准化

    参数:
        data (np.ndarray): 原始数据，形状为 (n_samples, n_features)

    返回:
        np.ndarray: 标准化后的数据，每个元素为 (x - 均值) / 标准差
    """
    # 按列计算均值和标准差
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data

   
def cumsum_filter(df: pl.DataFrame, h: float):
    S_high = 0
    S_low = 0
    S_high_list = []
    S_low_list = []

    
    for i in range(len(df)):
        if i == 0:
            continue  # 跳过第一个数据点
            
        # 计算累积和
        S_high = max(0, S_high + df["high"][i] - df["high"][i-1])
        S_low = min(0, S_low + df["low"][i] - df["low"][i-1])
        
        # 当超过阈值时，记录数据点并重置累积和
        if S_high > h:
            S_high_list.append([df["datetime"][i], df["high"][i]])

            S_high = 0  # 重要：需要重置累积和
            
        if S_low < -h:
            S_low_list.append([df["datetime"][i], df["low"][i]])

            S_low = 0  # 重要：需要重置累积和
    
    # 返回结果
    return S_high_list, S_low_list
    

def main():
    # read the data
    BTCUSDT_df = read_csv(TickData_path.tick_data_path)
    ETHUSDT_df = read_csv(TickData_path.tick_data_path_2)
    # print(df.head())

    question_1(BTCUSDT_df)
    
    # question_2(BTCUSDT_df)
    
    
    # ETHBTC_df = read_csv(TickData_path.tick_data_path_3)
    # question_3(BTCUSDT_df, ETHUSDT_df)
    
    # question_4(BTCUSDT_df)
    
    # question_5(BTCUSDT_df)
    
    
    
# python -m Part1.CH01.ch01
if __name__ == "__main__":
    main()
    
def question_5(df):
    df_dollar_bars = create_bars(df, "dollar", 2e7)
    h = 500
    
    S_low_list, S_high_list = cumsum_filter(df_dollar_bars, h)
    
    S_low_list = pl.DataFrame(S_low_list, schema=["datetime", "price"], orient="row")
    S_high_list = pl.DataFrame(S_high_list, schema=["datetime", "price"], orient="row")
    
    S_abs_list = cumsum_absolute_filter(df_dollar_bars, h)
    
    
    S_abs_list = pl.DataFrame(S_abs_list, schema=["datetime", "price"], orient="row")
    
    window_size = 3
    rolling_std_cumsum_filter = pl.concat([S_high_list, S_low_list]).with_columns(
        pl.col("price").rolling_std(window_size=window_size).alias("rolling_std")
    )
    rolling_std_cumsum_filter_abs = S_abs_list.with_columns(
        pl.col("price").rolling_std(window_size=window_size).alias("rolling_std")
    )
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(rolling_std_cumsum_filter["rolling_std"], label="cumsum_filter", alpha=0.7)
    ax.plot(rolling_std_cumsum_filter_abs["rolling_std"], label="cumsum_absolute_filter", alpha=0.7)
    ax.set_title("Heteroscedasticity")
    ax.legend()
    plt.grid(True)
    plt.show()
    
    cumsum_cv = rolling_std_cumsum_filter["rolling_std"].std() / rolling_std_cumsum_filter["rolling_std"].mean()
    cumsum_cv_abs = rolling_std_cumsum_filter_abs["rolling_std"].std() / rolling_std_cumsum_filter_abs["rolling_std"].mean()
    
    print(f"cumsum_cv: {cumsum_cv}")
    print(f"cumsum_cv_abs: {cumsum_cv_abs}")
    
    
    
    
    
    
    
def cumsum_absolute_filter(df: pl.DataFrame, h: float):
    """
    基于绝对收益的CUSUM过滤器
    
    参数:
    df: 包含价格数据的DataFrame
    h: 阈值参数
    price_col: 用于计算收益的价格列名
    
    返回:
    采样点列表，每个元素包含时间戳和价格
    """
    S_abs = 0  # 绝对收益的累积和
    S_abs_list = []  # 采样点列表
    
    for i in range(len(df)):
        if i == 0:
            continue  # 跳过第一个数据点
            
        # 计算收益（价格变化）的绝对值
        ret_abs = abs(df["close"][i] - df["close"][i-1])
        # 累积绝对收益
        S_abs += ret_abs
        # 当超过阈值时，记录数据点并重置累积和
        if S_abs > h:
            S_abs_list.append([df["datetime"][i], df["close"][i]])
            S_abs = 0  # 重置累积和
    
    return S_abs_list
    
    
    
    
def question_4(df):
    df_dollar_bars = create_bars(df, "dollar", 2e7)
     
    window_size = 20
    # 计算5%的布林带
    df_dollar_bars = df_dollar_bars.with_columns(
        pl.col("close").rolling_mean(window_size=window_size).alias("ma"),
        pl.col("close").rolling_std(window_size=window_size).alias("std"),
        (pl.col("close").rolling_mean(window_size=window_size) + 1.96 * pl.col("close").rolling_std(window_size=window_size)).alias("upper"),
        (pl.col("close").rolling_mean(window_size=window_size) - 1.96 * pl.col("close").rolling_std(window_size=window_size)).alias("lower")
    )
    
    out_data_high = []
    out_data_low = []
    upper_count = 0
    lower_count = 0
    for i in range(len(df_dollar_bars)):
        if i < window_size:
            continue
        if df_dollar_bars["high"][i] > df_dollar_bars["upper"][i]:
            upper_count += 1
            out_data_high.append([df_dollar_bars["datetime"][i], df_dollar_bars["high"][i]])
        if df_dollar_bars["low"][i] < df_dollar_bars["lower"][i]:
            lower_count += 1
            out_data_low.append([df_dollar_bars["datetime"][i], df_dollar_bars["low"][i]])
    out_data_high = pl.DataFrame(out_data_high, schema=["datetime", "price"], orient="row")
    out_data_low = pl.DataFrame(out_data_low, schema=["datetime", "price"], orient="row")
    # print(out_data_high)
    # print(out_data_low)
    
    
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_dollar_bars["datetime"], df_dollar_bars["ma"], label="MA", color="blue")
    ax.plot(df_dollar_bars["datetime"], df_dollar_bars["upper"], label="Upper", color="red")
    ax.plot(df_dollar_bars["datetime"], df_dollar_bars["lower"], label="Lower", color="green")
    
    # 绘制K线图
    for i in range(len(df_dollar_bars)):
        # 获取开盘、收盘、最高和最低价
        # open_price = df_dollar_bars['open'][i]
        # close_price = df_dollar_bars['close'][i]
        high_price = df_dollar_bars['high'][i]
        low_price = df_dollar_bars['low'][i]
        
        # 绘制上下影线
        ax.plot([df_dollar_bars["datetime"][i], df_dollar_bars["datetime"][i]], [low_price, high_price], color='black', linewidth=1)
    
    ax.scatter([out_data_high["datetime"]], [out_data_high["price"]], color='red', linewidth=1, label="bollinger_high")
    ax.scatter([out_data_low["datetime"]], [out_data_low["price"]], color='green', linewidth=1, label="bollinger_low")
    
    
    
    h = 200
    S_high_list, S_low_list = cumsum_filter(df_dollar_bars, h)
    # print(S_high_list, S_low_list)
    
    S_high_list = pl.DataFrame(S_high_list, schema=["datetime", "price"], orient="row")
    S_low_list = pl.DataFrame(S_low_list, schema=["datetime", "price"], orient="row")
    
    ax.scatter(S_high_list["datetime"], S_high_list["price"], color='blue', linewidth=2, label="cumsum_high")
    ax.scatter(S_low_list["datetime"], S_low_list["price"], color='orange', linewidth=2, label="cumsum_low")
    
    
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.set_title("BTCUSDT Dollar Bars Kline")
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.show()
    
    window_size = 3
    rolling_std_bollinger_bands = pl.concat([out_data_high, out_data_low]).with_columns(
        pl.col("price").rolling_std(window_size=window_size).alias("rolling_std")
    )
    rolling_std_cumsum_filter = pl.concat([S_high_list, S_low_list]).with_columns(
        pl.col("price").rolling_std(window_size=window_size).alias("rolling_std")
    )
    
    # print(rolling_std_bollinger_bands["rolling_std"])
    # print(rolling_std_cumsum_filter["rolling_std"])
    
    # calculate the heteroscedasticity
    bollinger_cv = rolling_std_bollinger_bands["rolling_std"].std() / rolling_std_bollinger_bands["rolling_std"].mean()
    cumsum_cv = rolling_std_cumsum_filter["rolling_std"].std() / rolling_std_cumsum_filter["rolling_std"].mean()
    
    print(f"bollinger_cv: {bollinger_cv}")
    print(f"cumsum_cv: {cumsum_cv}")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(rolling_std_bollinger_bands["rolling_std"], label="bollinger_bands", alpha=0.7)
    ax.plot(rolling_std_cumsum_filter["rolling_std"], label="CUSUM", alpha=0.7)
    ax.set_title("Heteroscedasticity")
    ax.legend()
    plt.grid(True)
    plt.show()
    
    
 

    
def question_3(df_1, df_2):
    df_1_dollar_bars = create_bars(df_1, "dollar", 2e7)
    df_2_dollar_bars = create_bars(df_2, "dollar", 1e7)
    # ETHUSDT_dollar_bars = create_bars(ETHUSDT_df, "dollar", 3.5e6)
    
    # we need as close length as possible, test it yourself
    # print(len(BTCUSDT_dollar_bars)) 
    # print(len(ETHUSDT_dollar_bars))
    
    # whatever log or diff, the result is the same
    df_1_returns = np.diff(np.log(df_1_dollar_bars["close"]))
    df_2_returns = np.diff(np.log(df_2_dollar_bars["close"]))
    
    min_length = min(len(df_1_returns), len(df_2_returns))
    df_1_returns = df_1_returns[:min_length]
    df_2_returns = df_2_returns[:min_length]
    
    # BTCUSDT_returns = z_score_normalize(BTCUSDT_returns)
    # ETHUSDT_returns = z_score_normalize(ETHUSDT_returns)
    
    return_matrix = np.column_stack((df_1_returns, df_2_returns))
    
    # fig,ax = plt.subplots(figsize=(14,7))
    # ax.plot(BTCUSDT_returns, label="BTCUSDT")
    # ax.plot(ETHUSDT_returns, label="ETHUSDT")
    # ax.legend()
    # plt.show()
    
    weights = compute_pca_weights(return_matrix)
    
    print("weights:", weights)
    
    spread = df_2_returns - (weights[1] / weights[0]) * df_1_returns
    
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,10))
    ax1.plot(df_1_returns, label="df_1")
    ax1.plot(df_2_returns, label="df_2")
    ax1.set_title("regional returns")
    ax1.legend()
    ax2.plot(spread, label="df_1 - df_2 spread", color="green")
    ax2.set_title("df_1 - df_2 spread")
    ax2.legend()
    plt.tight_layout()
    plt.show()
    
    # 打印一些基本统计信息
    print("\n价差统计信息:")
    print(f"均值: {np.mean(spread):.6f}")
    print(f"标准差: {np.std(spread):.6f}")
    print(f"最大值: {np.max(spread):.6f}")
    print(f"最小值: {np.min(spread):.6f}")
    
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(spread)
    print("\nADF检验结果:")
    print(f"ADF统计量: {adf_result[0]:.6f}")
    print(f"p值: {adf_result[1]:.6f}")
    print("临界值:")
    for key, value in adf_result[4].items():
        print(f"\t{key}: {value:.6f}")
        
    # 判断是否平稳
    if adf_result[1] < 0.05:
        print("\n结论: 序列是平稳的 (在5%的显著性水平下拒绝单位根假设)")
    else:
        print("\n结论: 序列不是平稳的 (无法在5%的显著性水平下拒绝单位根假设)")
    

    

def question_2(df):
    dollar_bars = create_bars(df, "dollar", 1e6)
    
    dollar_imbalance_bars = create_imbalance_bars(df, "dollar", 1e5)
    
    profit_time_interval = "3h"
    dollar_bars_count = count_bars(dollar_bars, profit_time_interval)
    dollar_imbalance_bars_count = count_bars(dollar_imbalance_bars, profit_time_interval)
    
    
    # print(dollar_imbalance_bars)
    # print(dollar_imbalance_bars_count)
    
    dollar_correlation = calculate_autocorrelation(dollar_bars, 1)
    dollar_imbalance_correlation = calculate_autocorrelation(dollar_imbalance_bars, 1)
    
    print(f"dollar_correlation: {dollar_correlation}")
    print(f"dollar_imbalance_correlation: {dollar_imbalance_correlation}")
    
    
    
    
    
    
    
def question_1(df):
    time_bars = create_bars(df, "time", 10000)
    # # print(time_bars.head());print("notice the timestamp's difference")
    
    # print(time_bars_count.head())
    
    volume_bars = create_bars(df, "volume", 100)
    # print(volume_bars.head());print("notice the qty")
    

    dollar_bars = create_bars(df, "dollar", 1e6)
    # print(dollar_bars.head());print("notice the dollar")
    
    
    time_interval = "10m"
    profit_time_interval = "3h"
    time_bars_count = count_bars(time_bars, time_interval)
    volume_bars_count = count_bars(volume_bars, time_interval)
    dollar_bars_count = count_bars(dollar_bars, time_interval)
    time_stability = {
        "mean": time_bars_count["bar_count"].mean(),
        "std": time_bars_count["bar_count"].std(),
        "cv": time_bars_count["bar_count"].std() / time_bars_count["bar_count"].mean()
    }
    volume_stability = {
        "mean": volume_bars_count["bar_count"].mean(),
        "std": volume_bars_count["bar_count"].std(),
        "cv": volume_bars_count["bar_count"].std() / volume_bars_count["bar_count"].mean()
    }
    dollar_stability = {
        "mean": dollar_bars_count["bar_count"].mean(),
        "std": dollar_bars_count["bar_count"].std(),
        "cv": dollar_bars_count["bar_count"].std() / dollar_bars_count["bar_count"].mean()
    }
    
    
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    
    ax.plot(time_bars_count["datetime"], time_bars_count["bar_count"], label="time")
    ax.plot(volume_bars_count["datetime"], volume_bars_count["bar_count"], label="volume")
    ax.plot(dollar_bars_count["datetime"], dollar_bars_count["bar_count"], label="dollar")
    
    ax.set_xlabel("time")
    ax.set_ylabel("bar count")
    ax.set_title("bar count comparison")
    ax.legend()
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    print("各类型柱状图数量的稳定性指标:")
    print("time_stability:", time_stability)
    print("volume_stability:", volume_stability)
    print("dollar_stability:", dollar_stability)
    
    import os
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_script_path, "bar_count_comparison.png")
    plt.savefig(output_file_path)
    # plt.show()
    
    # 计算自相关性
    for lag in range(1, 3):
        time_autocorr = calculate_autocorrelation(time_bars, lag)
        volume_autocorr = calculate_autocorrelation(volume_bars, lag)
        dollar_autocorr = calculate_autocorrelation(dollar_bars, lag)
        print(f"\n自相关性比较(lag={lag}):")
        print(f"Time bars: {time_autocorr}")
        print(f"Volume bars: {volume_autocorr}")
        print(f"Dollar bars: {dollar_autocorr}")
        
    # 计算每3小时子集的收益方差
    print(f"\n每{profit_time_interval}子集的收益方差分析:")
    time_variance_stats = calculate_returns_variance(time_bars, profit_time_interval)
    volume_variance_stats = calculate_returns_variance(volume_bars, profit_time_interval)
    dollar_variance_stats = calculate_returns_variance(dollar_bars, profit_time_interval)
    
    print("\nTime bars方差统计:")
    print(f"平均方差: {time_variance_stats['mean_variance']}")
    print(f"方差的方差: {time_variance_stats['variance_of_variance']}")
    print(f"变异系数: {time_variance_stats['cv']}")
    
    print("\nVolume bars方差统计:")
    print(f"平均方差: {volume_variance_stats['mean_variance']}")
    print(f"方差的方差: {volume_variance_stats['variance_of_variance']}")
    print(f"变异系数: {volume_variance_stats['cv']}")
    
    print("\nDollar bars方差统计:")
    print(f"平均方差: {dollar_variance_stats['mean_variance']}")
    print(f"方差的方差: {dollar_variance_stats['variance_of_variance']}")
    print(f"变异系数: {dollar_variance_stats['cv']}")
    
    # 计算Jarque-Bera检验统计量
    print("\nJarque-Bera正态性检验结果:")
    time_jb = calculate_jarque_bera(time_bars)
    volume_jb = calculate_jarque_bera(volume_bars)
    dollar_jb = calculate_jarque_bera(dollar_bars)
    
    print(f"Time bars JB统计量: {time_jb:.4f}")
    print(f"Volume bars JB统计量: {volume_jb:.4f}")
    print(f"Dollar bars JB统计量: {dollar_jb:.4f}")
    
    # 找出最小的统计量
    min_jb = min(time_jb, volume_jb, dollar_jb)
    if min_jb == time_jb:
        print("\nTime bars具有最小的Jarque-Bera统计量，表明其收益率分布最接近正态分布。")
    elif min_jb == volume_jb:
        print("\nVolume bars具有最小的Jarque-Bera统计量，表明其收益率分布最接近正态分布。")
    else:
        print("\nDollar bars具有最小的Jarque-Bera统计量，表明其收益率分布最接近正态分布。")
    