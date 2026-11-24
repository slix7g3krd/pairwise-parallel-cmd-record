# pairwise-parallel-cmd-record

## 领域

游戏、图形与多媒体

## 主题

并行命令录制与资源状态

## 说明

全局状态表、围栏回收与固定录制序。

初始状态下 `graphics/parallel_cmd_record.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
