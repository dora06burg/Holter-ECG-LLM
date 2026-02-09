# stream_result

这个目录存放一次或多次“LSL 流式模拟 + 写回 WFDB + 可视化”的输出结果。

推荐入口（latest 报告）：

- `ECG_Model/stream_result/report/README.md`
- `ECG_Model/stream_result/report/assets/250Hz/summary.png`

本次最新 run 目录：

- `ECG_Model/stream_result/run_20260209_163829/`

其中：
- `wfdb_250Hz/`：写回的 WFDB 窗口（`.dat/.hea/.atr` + `RECORDS`）
- `logs/`：sender/receiver/plot 的日志
- `report/`：更详细的说明（同样会在 `ECG_Model/stream_result/report/` 镜像一份）

如果你想重新生成一份结果（自动运行 sender+receiver+plot+report）：

```bash
python ECG_Model/code_for_stream/generate_stream_result.py --windows 3 --speed 20
```

