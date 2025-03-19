#How to run : python -m Part1.CH01.ch01
import polars as pl
import numpy as np

from variables import tick_data_path, TickData_col
from src.data_reader import read_csv

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
        for i, time in enumerate(df[:, TickData_col.time_col]):
            if time - cumsum >= threshold:
                bar_indices.append(i)
                cumsum = time
    elif bar_type == "volume":
        cumsum = 0
        for i, qty in enumerate(df[:, TickData_col.qty_col]):
            cumsum += qty
            if cumsum >= threshold:
                bar_indices.append(i)
                cumsum = 0
    elif bar_type == "dollar":
        cumsum = 0
        for i, (price, qty) in enumerate(zip(df[:, TickData_col.price_col], df[:, TickData_col.qty_col])):
            cumsum += price * qty
            if cumsum >= threshold:
                bar_indices.append(i)
                cumsum = 0
        
    if bar_indices[-1] != len(df - 1): # 检查完成性
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
    np_datetime = bars[:, "datetime"].to_numpy()
    dtype_str = f"timedelta64[{time_interval}]"
    np_datetime_diff = np.diff(np_datetime).astype(dtype_str).astype(np.int64)
    

    return np_datetime_diff
    
    

def question_b(count_bars, time_interval, time_bars, volume_bars, dollar_bars):
    time_bars_count = count_bars(time_bars, time_interval)
    volume_bars_count = count_bars(volume_bars, time_interval)
    dollar_bars_count = count_bars(dollar_bars, time_interval)
    time_stability = {
        "mean": time_bars_count.mean(),
        "std": time_bars_count.std(),
        "cv": time_bars_count.std() / time_bars_count.mean()
    }
    volume_stability = {
        "mean": volume_bars_count.mean(),
        "std": volume_bars_count.std(),
        "cv": volume_bars_count.std() / volume_bars_count.mean()
    }
    dollar_stability = {
        "mean": dollar_bars_count.mean(),
        "std": dollar_bars_count.std(),
        "cv": dollar_bars_count.std() / dollar_bars_count.mean()
    }
    
    # 绘制时间序列图
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    
    ax.plot(time_bars_count, label="time")
    ax.plot(volume_bars_count, label="volume")
    ax.plot(dollar_bars_count, label="dollar")
    
    ax.set_xlabel("time")
    ax.set_ylabel("bar count")
    ax.set_title("bar count comparison")
    ax.legend()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    
    print("各类型柱状图数量的稳定性指标:")
    print("time_stability:", time_stability)
    print("volume_stability:", volume_stability)
    print("dollar_stability:", dollar_stability)
    
    import os
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_script_path, "bar_count_comparison.png")
    plt.savefig(output_file_path)
    plt.show()

def main():
    # read the data
    df = read_csv(tick_data_path)
    # print(df.head())
    
    time_interval = "5m"

    time_bars = create_bars(df, "time", 1000000)
    # # print(time_bars.head());print("notice the timestamp's difference")
    
    # print(time_bars_count.head())
    
    volume_bars = create_bars(df, "volume", 2000)
    # print(volume_bars.head());print("notice the qty")
    

    dollar_bars = create_bars(df, "dollar", 1e8)
    # print(dollar_bars.head());print("notice the dollar")
    

    question_b(count_bars, time_interval, time_bars, volume_bars, dollar_bars)
    
if __name__ == "__main__":
    main()
    