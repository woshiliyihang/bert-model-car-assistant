# bert-model-car-assistant

## 简介

本仓库提供一个用于车载场景的在线 NLU（意图识别 + 槽位抽取）测试脚本。脚本会从 Hugging Face Hub 拉取 JointBERT 模型，并在本地以交互式控制台形式进行中文指令的预测测试。

## 功能

- 在线加载 Hugging Face 上的模型仓库（支持公开与私有仓库的 `use_auth_token`）。
- 对中文命令同时输出意图（intent）与 BIO 槽位（slot）预测，并返回意图置信度。
- 提供简单的 BIO 解码函数将槽位标签还原为键值对。
- 支持在 CPU 或 GPU（CUDA）上运行。

## 特点

- 基于 Transformers（BERT）的轻量 JointBertModel 实现。
- 内置常见车载指令的意图与槽位标签集合，便于快速测试与演示。
- 控制台交互友好：支持历史查看与退出命令。

## 快速开始

在含有本脚本的目录下运行：

```bash
python test_nlu_dialogue_online.py --repo-id mhhyoucom/bert-model-car-assistant
```

可选参数：
- `--device cuda` 或 `--device cpu` 指定运行设备（若不指定，脚本会优先使用可用的 CUDA）。
- `--auth-token <TOKEN>` 在访问私有模型仓库或需要身份验证时传入 Hugging Face 令牌。

## 交互说明

- 启动脚本后，在控制台输入中文命令并回车查看识别结果。
- 输入 `history` 查看会话历史（包含意图与槽位）。
- 输入 `exit` 或 `quit` 退出测试。

## 参考脚本

具体实现请参见：

[test_nlu_dialogue_online.py](test_nlu_dialogue_online.py)

## 许可证

本项目遵循仓库根目录的 LICENSE 文件中的许可条款。
