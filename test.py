import polars as pl
import numpy as np

from src.data_reader import read_csv
from variables import tick_data_path, TickData_col

df = read_csv(tick_data_path)
print(df.head())

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
        bar_indices = list(range(0, len(df), threshold))
        
    if bar_indices[-1] != len(df - 1): # 检查完成性
        bar_indices.append(len(df) - 1)
        
    bars = []
    for i in range(len(bar_indices) - 1):
        start_idx = bar_indices[i]
        end_idx = bar_indices[i + 1]
        
        bar_data = df.slice(start_idx, end_idx - start_idx + 1)
        
        bar = {
            "open": bar_data[TickData_col.price_col][0],
            "high": bar_data[TickData_col.price_col].max(),
            "low": bar_data[TickData_col.price_col].min(),
            "close": bar_data[TickData_col.price_col][-1],
            "volume": (bar_data[TickData_col.qty_col] * bar_data[TickData_col.price_col]).sum(),
            "timestamp": bar_data[TickData_col.time_col][0]
        }
        bars.append(bar)
        

time_bars = create_bars(df, "time", 100)
print(time_bars.head())
