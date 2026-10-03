# task26100547

进程内七日振动趋势预测库：按普通最小二乘（OLS）直线拟合最近 7 天
（左开右闭）的传感器读数，预测设备到达维护阈值 `H` 的时间。

## 用法

```python
from vibration_trend import TrendForecaster

f = TrendForecaster(threshold=10.0)   # 每个实例对应一条传感器流
f.append(1_700_000_000_000_000, 1.0)  # (整数微秒时间戳, 有限 binary64 读数)
f.append(1_700_000_001_000_000, 1.5)
forecast = f.forecast(1_700_000_002_000_000)
forecast.seconds  # 距离维护的预测秒数；无法预测时为 None
forecast.urgent   # 预测存在且严格小于 259200 秒（72h）时为 True
```

- `append(t, value)` 与 `forecast(now)` 推进同一逻辑时钟，时间不后退；
  采样时间戳严格递增。
- 预测从拟合直线在 `now` 处的取值出发，而非最后一个原始读数。
- 少于 2 个有效样本、或斜率非正时，`seconds` 为 `None` 且 `urgent` 为假；
  拟合值已达到或超过阈值时预测为 `0`。

## 运行测试

```bash
python3 -m pytest tests/ -q
```
