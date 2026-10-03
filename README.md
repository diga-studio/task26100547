# vibration-forecast

进程内七日振动趋势预测库：按普通最小二乘（OLS）直线拟合最近 7 天的传感器读数，
预测设备何时达到维护阈值 H。

## 用法

```python
from vibration_forecast import VibrationTrendForecaster

f = VibrationTrendForecaster(threshold=100.0)   # 每个实例对应一条传感器流
f.append(1_700_000_000_000_000, 12.5)           # (整数微秒时间戳, 有限 binary64 读数)
f.append(1_700_000_060_000_000, 13.0)
result = f.forecast(1_700_000_120_000_000)      # -> Forecast(seconds, urgent)
```

- 有效窗口为最近 7 天（左开右闭）；`append` 与 `forecast` 推进同一逻辑时钟，时间不后退，采样时间严格递增。
- 预测从拟合直线在 `now` 处的取值出发，而非最后一个原始读数；每个样本等权。
- 少于 2 个点时 `seconds=None, urgent=False`；拟合值已达阈值时 `seconds=0.0`；
  仅正斜率返回未来到达秒数，零/负斜率返回 `None`。
- `urgent` 仅在预测存在且严格小于 259200 秒（72 小时）时为真。

## 运行测试

```bash
python3 -m pytest tests/ -q
```
