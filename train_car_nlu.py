#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车机端侧 NLU 联合意图识别与槽位提取 - 企业级训练脚本 (支持腾讯钉钉版 NLPCC2018 Task4 数据)
依赖安装:
  pip install "transformers>=4.40" torch onnxruntime "optimum[onnxruntime]" accelerate onnxscript pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
运行:
  # 使用内置 Demo 数据
  python train_car_nlu.py --demo

  # 使用 NLPCC2018 钉钉版数据
  python train_car_nlu.py --nlpcc2018 corpus.train.txt intent-definition.xlsx
"""

import os
import re
import json
import random
import argparse
import zipfile
import xml.etree.ElementTree as ET
import numpy as np

# ============================================================
# 第 0 部分: 环境配置 (必须在导入 transformers 之前)
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertModel,
    BertPreTrainedModel,
    TrainingArguments,
    Trainer,
)

# 版本兼容检测
_tf_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
_USE_EVAL_STRATEGY = _tf_version >= (4, 46)  # 4.46+ 用 eval_strategy

# ============================================================
# 第 1 部分: 标签体系定义
# ============================================================
INTENT_LABELS = [
    "navigation",
    "control_window",
    "control_ac",
    "play_music",
    "weather_query",
    "control_wiper",
    "phone_call",
    "other",
]
INTENT2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2INTENT = {i: label for label, i in INTENT2ID.items()}

SLOT_LABELS = [
    "O",
    "B-destination",
    "I-destination",
    "B-location",
    "I-location",
    "B-song",
    "I-song",
    "B-artist",
    "I-artist",
    "B-contact",
    "I-contact",
    "B-temperature",
    "I-temperature",
    "B-window_position",
    "I-window_position",
    "B-ac_mode",
    "I-ac_mode",
]
SLOT2ID = {label: i for i, label in enumerate(SLOT_LABELS)}
ID2SLOT = {i: label for label, i in SLOT2ID.items()}

NUM_INTENTS = len(INTENT_LABELS)
NUM_SLOTS = len(SLOT_LABELS)
MAX_LEN = 64

print("=" * 65)
print(f"  车机 NLU 联合模型训练 (transformers {transformers.__version__})")
print(f"  意图类别数: {NUM_INTENTS} | 槽位类别数: {NUM_SLOTS}")
print("=" * 65)

# ============================================================
# 第 2 部分: 数据加载 (新增 NLPCC2018 钉钉版支持)
# ============================================================
def generate_demo_training_data():
    data = []

    # 导航场景
    nav_dests = [
        "北京", "上海虹桥机场", "广州南站", "深圳福田区", "成都太古里",
        "杭州西湖", "南京路步行街", "武汉光谷", "西安大雁塔", "重庆解放碑",
    ]
    nav_templates = [
        "导航到{dest}", "我要去{dest}", "帮我规划路线去{dest}",
        "怎么去{dest}", "导航去{dest}", "开车去{dest}",
    ]
    for tmpl in nav_templates:
        for dest in nav_dests:
            data.append({
                "text": tmpl.format(dest=dest),
                "intent": "navigation",
                "slots": {"destination": dest},
            })

    # 车窗控制
    window_positions = ["主驾", "副驾", "后排左侧", "后排右侧", "全部"]
    window_verbs = ["打开", "关闭", "摇下", "升起"]
    for pos in window_positions:
        for verb in window_verbs:
            data.append({
                "text": f"{verb}{pos}车窗",
                "intent": "control_window",
                "slots": {"window_position": pos},
            })

    # 空调温度
    temps = ["22度", "24度", "26度", "18度", "20度", "28度", "30度"]
    for temp in temps:
        data.append({
            "text": f"把空调温度调到{temp}",
            "intent": "control_ac",
            "slots": {"temperature": temp},
        })
        data.append({
            "text": f"空调设为{temp}",
            "intent": "control_ac",
            "slots": {"temperature": temp},
        })

    # 空调模式
    ac_modes = ["制冷", "制热", "除雾", "内循环", "外循环", "自动模式"]
    for mode in ac_modes:
        data.append({
            "text": f"空调开{mode}",
            "intent": "control_ac",
            "slots": {"ac_mode": mode},
        })
        data.append({
            "text": f"切换到{mode}模式",
            "intent": "control_ac",
            "slots": {"ac_mode": mode},
        })

    # 音乐播放
    songs = ["晴天", "稻香", "七里香", "夜曲", "青花瓷", "简单爱", "双截棍"]
    artists = ["周杰伦", "林俊杰", "邓紫棋", "薛之谦", "毛不易"]
    for song in songs:
        data.append({
            "text": f"播放{song}",
            "intent": "play_music",
            "slots": {"song": song},
        })
        data.append({
            "text": f"我想听{song}",
            "intent": "play_music",
            "slots": {"song": song},
        })
    for artist in artists:
        data.append({
            "text": f"播放{artist}的歌",
            "intent": "play_music",
            "slots": {"artist": artist},
        })
        data.append({
            "text": f"来一首{artist}的歌",
            "intent": "play_music",
            "slots": {"artist": artist},
        })
    for song, artist in zip(songs[:5], artists[:5]):
        data.append({
            "text": f"播放{artist}的{song}",
            "intent": "play_music",
            "slots": {"artist": artist, "song": song},
        })

    # 天气查询
    locations = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"]
    for loc in locations:
        data.append({
            "text": f"{loc}今天天气怎么样",
            "intent": "weather_query",
            "slots": {"location": loc},
        })
        data.append({
            "text": f"查一下{loc}的天气",
            "intent": "weather_query",
            "slots": {"location": loc},
        })
        data.append({
            "text": f"{loc}明天会下雨吗",
            "intent": "weather_query",
            "slots": {"location": loc},
        })

    # 电话拨打
    contacts = ["张三", "李四", "王五", "妈妈", "老板", "老婆", "爸爸"]
    for contact in contacts:
        data.append({
            "text": f"打电话给{contact}",
            "intent": "phone_call",
            "slots": {"contact": contact},
        })
        data.append({
            "text": f"拨打{contact}的电话",
            "intent": "phone_call",
            "slots": {"contact": contact},
        })

    # 雨刷控制 (无槽位)
    wiper_cmds = ["打开雨刷", "关闭雨刷", "雨刷调到最快", "雨刷调慢一点", "开雨刷", "关雨刷"]
    for cmd in wiper_cmds:
        data.append({"text": cmd, "intent": "control_wiper", "slots": {}})

    # 兜底意图
    others = ["你好", "谢谢", "今天星期几", "讲个笑话", "你叫什么名字", "现在几点了"]
    for text in others:
        data.append({"text": text, "intent": "other", "slots": {}})

    random.shuffle(data)
    return data


def parse_nlpcc2018_intent(intent_str):
    """
    将 NLPCC2018 钉钉版的原始 intent 映射为内部意图标签。
    示例：music.play → play_music, navigation.start_navigation → navigation
    """
    intent_str = intent_str.strip().lower()
    if intent_str.startswith("music"):
        return "play_music"
    elif intent_str.startswith("navigation"):
        return "navigation"
    elif intent_str.startswith("phone"):
        return "phone_call"
    else:
        return "other"


def _load_intent_definition(intent_def_path):
    """读取 intent-definition.xlsx 的第一张表，并返回一个简单的意图列表。"""
    if not os.path.exists(intent_def_path):
        raise FileNotFoundError(f"intent-definition 文件不存在: {intent_def_path}")

    with zipfile.ZipFile(intent_def_path, 'r') as z:
        if 'xl/sharedStrings.xml' in z.namelist():
            shared_strings = ET.fromstring(z.read('xl/sharedStrings.xml'))
            ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
            strings = [
                ''.join(t.text or '' for t in si.findall('.//' + ns + 't'))
                for si in shared_strings.findall(ns + 'si')
            ]
        else:
            strings = []

        sheet_files = [name for name in z.namelist() if name.startswith('xl/worksheets/sheet')]
        if not sheet_files:
            raise ValueError('无法读取 intent-definition.xlsx 中的 worksheet')

        sheet = ET.fromstring(z.read(sheet_files[0]))
        ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
        rows = sheet.findall('.//' + ns + 'row')
        header = []
        results = []
        for row_idx, row in enumerate(rows):
            values = []
            for cell in row.findall(ns + 'c'):
                value = cell.find(ns + 'v')
                if value is None:
                    values.append('')
                    continue
                text = value.text or ''
                if cell.get('t') == 's':
                    text = strings[int(text)] if text.isdigit() and int(text) < len(strings) else text
                values.append(text)
            if row_idx == 0:
                header = [h.strip() for h in values]
                continue
            if not any(values):
                continue
            row_dict = {header[i]: values[i].strip() if i < len(values) else '' for i in range(len(header))}
            results.append(row_dict)
    return results


def load_nlpcc2018_dingdang_data(corpus_path, intent_def_path=None):
    """
    加载 NLPCC2018 钉钉版数据集：
    - corpus_path: corpus.train.txt 路径
    - intent_def_path: intent-definition.xlsx 路径（可选）
    """
    data = []
    if intent_def_path:
        if os.path.exists(intent_def_path):
            try:
                _ = _load_intent_definition(intent_def_path)
                print(f"  [NLPCC2018 钉钉版] 已加载 intent-definition: {intent_def_path}")
            except Exception as e:
                print(f"  [WARN] 无法解析 intent-definition.xlsx，将忽略该文件: {e}")
        else:
            print(f"  [WARN] intent-definition.xlsx 文件不存在，已忽略: {intent_def_path}")

    # 读取训练语料，兼容 4 列格式：id\t文本\t意图\t槽位标注
    slot_pattern = re.compile(r'<([^/>]+)>(.*?)</\1>')
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                continue

            utterance = parts[1].strip() if len(parts) > 1 else ''
            intent_str = parts[2].strip().lower() if len(parts) > 2 else ''
            slot_annotation = parts[3].strip() if len(parts) > 3 else ''

            if not utterance or not intent_str:
                continue

            intent = parse_nlpcc2018_intent(intent_str)
            slots = {}
            for match in slot_pattern.finditer(slot_annotation):
                slot_type = match.group(1).strip()
                slot_value = match.group(2).strip()
                if slot_type and slot_value:
                    slots[slot_type] = slot_value

            data.append({
                'text': utterance,
                'intent': intent,
                'slots': slots,
            })

    print(f"  [NLPCC2018 钉钉版] 加载 {len(data)} 条样本来自: {corpus_path}")
    return data


# ============================================================
# 第 3 部分: BIO 标签对齐
# ============================================================
def text_to_bio(text, slots):
    tags = ["O"] * len(text)
    for slot_name, slot_value in slots.items():
        if not slot_value or slot_value not in text:
            continue
        start = text.index(slot_value)
        end = start + len(slot_value)
        tags[start] = f"B-{slot_name}"
        for i in range(start + 1, end):
            tags[i] = f"I-{slot_name}"
    return tags

# ============================================================
# 第 4 部分: 数据集类
# ============================================================
class CarNLUDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=MAX_LEN):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        intent = item["intent"]
        slots = item.get("slots", {})
        bio_tags = text_to_bio(text, slots)
        chars = list(text)
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors=None,
        )
        slot_ids = [SLOT2ID.get(tag, SLOT2ID["O"]) for tag in bio_tags]
        slot_ids = [-100] + slot_ids
        slot_ids = slot_ids[: self.max_len - 1] + [-100]
        slot_ids = slot_ids + [-100] * (self.max_len - len(slot_ids))
        intent_id = INTENT2ID.get(intent, INTENT2ID["other"])
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "intent_labels": torch.tensor(intent_id, dtype=torch.long),
            "slot_labels": torch.tensor(slot_ids, dtype=torch.long),
        }

# ============================================================
# 第 5 部分: 联合模型定义
# ============================================================
class JointBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_intents = config.num_intents
        self.num_slots = config.num_slots
        self.intent_loss_weight = getattr(config, "intent_loss_weight", 1.0)
        self.slot_loss_weight = getattr(config, "slot_loss_weight", 1.0)
        self.bert = BertModel(config)
        self.intent_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, self.num_intents)
        self.slot_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.slot_classifier = nn.Linear(config.hidden_size, self.num_slots)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                intent_labels=None, slot_labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]
        intent_logits = self.intent_classifier(self.intent_dropout(pooled_output))
        slot_logits = self.slot_classifier(self.slot_dropout(sequence_output))
        total_loss = None
        if intent_labels is not None and slot_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            intent_loss = loss_fct(
                intent_logits.view(-1, self.num_intents),
                intent_labels.view(-1),
            )
            slot_loss = loss_fct(
                slot_logits.view(-1, self.num_slots),
                slot_labels.view(-1),
            )
            total_loss = (
                self.intent_loss_weight * intent_loss
                + self.slot_loss_weight * slot_loss
            )
        return {
            "loss": total_loss,
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
        }

# ============================================================
# 第 6 部分: 评估指标
# ============================================================
def _extract_entities(label_seq):
    entities = set()
    current_type = None
    current_start = None
    for i, label in enumerate(label_seq):
        if label == -100:
            if current_type is not None:
                entities.add((current_type, current_start, i - 1))
                current_type = None
            continue
        slot_name = ID2SLOT.get(label, "O")
        if slot_name.startswith("B-"):
            if current_type is not None:
                entities.add((current_type, current_start, i - 1))
            current_type = slot_name[2:]
            current_start = i
        elif slot_name.startswith("I-"):
            if current_type is None:
                current_type = slot_name[2:]
                current_start = i
        else:
            if current_type is not None:
                entities.add((current_type, current_start, i - 1))
                current_type = None
    if current_type is not None:
        entities.add((current_type, current_start, len(label_seq) - 1))
    return entities

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    intent_preds = np.argmax(predictions[0], axis=-1)
    slot_preds = np.argmax(predictions[1], axis=-1)
    intent_labels = labels[0]
    slot_labels = labels[1]
    intent_acc = (intent_preds == intent_labels).mean()
    tp, fp, fn = 0, 0, 0
    for i in range(len(slot_preds)):
        pred_entities = _extract_entities(slot_preds[i])
        true_entities = _extract_entities(slot_labels[i])
        for ent in pred_entities:
            if ent in true_entities:
                tp += 1
            else:
                fp += 1
        for ent in true_entities:
            if ent not in pred_entities:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "intent_acc": float(intent_acc),
        "slot_precision": float(precision),
        "slot_recall": float(recall),
        "slot_f1": float(f1),
    }

# ============================================================
# 第 7 部分: 兼容版 Trainer 子类
# ============================================================
class JointTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v for k, v in inputs.items() if k != "num_items_in_batch"}
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = {k: v for k, v in inputs.items() if k != "num_items_in_batch"}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"]
            intent_logits = outputs["intent_logits"]
            slot_logits = outputs["slot_logits"]
        if prediction_loss_only:
            return loss, None, None
        return (
            loss,
            (intent_logits.cpu(), slot_logits.cpu()),
            (inputs["intent_labels"].cpu(), inputs["slot_labels"].cpu()),
        )

# ============================================================
# 第 8 部分: 推理与后处理
# ============================================================
def _bio_decode(chars, slot_ids):
    slots = {}
    current_slot = None
    current_value = ""
    for i in range(1, len(chars) + 1):
        if i >= len(slot_ids):
            break
        tag = ID2SLOT.get(slot_ids[i], "O")
        char = chars[i - 1]
        if tag.startswith("B-"):
            if current_slot:
                slots[current_slot] = current_value
            current_slot = tag[2:]
            current_value = char
        elif tag.startswith("I-") and current_slot:
            current_value += char
        else:
            if current_slot:
                slots[current_slot] = current_value
                current_slot = None
    if current_slot:
        slots[current_slot] = current_value
    return slots

def predict(text, model, tokenizer, device="cpu"):
    model.eval()
    chars = list(text)
    encoding = tokenizer(
        chars,
        is_split_into_words=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    intent_logits = outputs["intent_logits"][0]
    slot_logits = outputs["slot_logits"][0]
    intent_probs = torch.softmax(intent_logits, dim=-1)
    intent_id = torch.argmax(intent_probs).item()
    slot_ids = torch.argmax(slot_logits, dim=-1).tolist()
    slots = _bio_decode(chars, slot_ids)
    return {
        "text": text,
        "intent": ID2INTENT[intent_id],
        "slots": slots,
        "confidence": round(intent_probs[intent_id].item(), 4),
    }

# ============================================================
# 第 9 部分: ONNX 导出与验证 (修复版)
# ============================================================
def export_onnx(model, tokenizer, save_dir="./onnx_nlu_model"):
    import onnxruntime as ort
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()  # 确保在 eval 模式
    
    dummy_text = "导航到北京"
    encoding = tokenizer(
        list(dummy_text),
        is_split_into_words=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    class OnnxWrapper(nn.Module):
        def __init__(self, joint_model):
            super().__init__()
            self.model = joint_model

        def forward(self, input_ids, attention_mask):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return out["intent_logits"], out["slot_logits"]

    wrapped = OnnxWrapper(model)
    onnx_path = os.path.join(save_dir, "model.onnx")
    
    # 关键修复: 使用 opset_version=18 兼容新版 PyTorch 的 LayerNorm 实现
    torch.onnx.export(
        wrapped,
        (encoding["input_ids"], encoding["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["intent_logits", "slot_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "intent_logits": {0: "batch"},
            "slot_logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=18,  # 兼容 PyTorch 2.0+ 和 ONNX Runtime 1.16+
    )
    
    print(f"  [ONNX] 模型已导出至: {onnx_path} (opset=18)")
    
    label_map = {
        "intent2id": INTENT2ID,
        "slot2id": SLOT2ID,
        "max_len": MAX_LEN,
    }
    with open(os.path.join(save_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"  [ONNX] 标签映射已保存至: {save_dir}/label_map.json")

    # --- 新增: ONNX 模型验证 ---
    print(f"  [VERIFY] 正在验证 ONNX 模型完整性...")
    try:
        session = ort.InferenceSession(onnx_path)
        
        # 检查输入输出签名
        input_names = {inp.name for inp in session.get_inputs()}
        output_names = {out.name for out in session.get_outputs()}
        assert "input_ids" in input_names, "缺失 input_ids 输入"
        assert "attention_mask" in input_names, "缺失 attention_mask 输入"
        assert "intent_logits" in output_names, "缺失 intent_logits 输出"
        assert "slot_logits" in output_names, "缺失 slot_logits 输出"
        
        # 运行一次推理测试
        dummy_input = {
            "input_ids": encoding["input_ids"].numpy(),
            "attention_mask": encoding["attention_mask"].numpy(),
        }
        outputs = session.run(None, dummy_input)
        
        # 检查输出 shape
        assert outputs[0].shape[0] == 1, "Intent batch 维度错误"
        assert outputs[1].shape[0] == 1, "Slot batch 维度错误"
        
        print(f"  [VERIFY] ✅ ONNX 模型验证通过! (Input: {input_names} -> Output: {output_names})")
        
    except Exception as e:
        print(f"  [VERIFY] ❌ ONNX 模型验证失败: {e}")

# ============================================================
# 第 10 部分: 主函数 (新增命令行参数)
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="车机 NLU 联合模型训练脚本")
    parser.add_argument("--demo", action="store_true",
                        help="使用内置 Demo 数据训练")
    parser.add_argument("--nlpcc2018", nargs=2, metavar=("CORPUS_PATH", "INTENT_DEF_PATH"),
                        help="使用 NLPCC2018 钉钉版数据，指定 corpus.train.txt 和 intent-definition.xlsx 路径")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n[1/6] 生成/加载训练数据...")
    if args.demo:
        print("  使用内置 Demo 数据训练（你之前的示例数据）")
        all_data = generate_demo_training_data()
    elif args.nlpcc2018:
        corpus_path, intent_def_path = args.nlpcc2018
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"文件不存在: {corpus_path}")
        all_data = load_nlpcc2018_dingdang_data(corpus_path, intent_def_path)
    else:
        # 默认行为：使用 Demo 数据（保持向后兼容）
        print("  未指定 --demo 或 --nlpcc2018，默认使用内置 Demo 数据")
        all_data = generate_demo_training_data()

    if len(all_data) == 0:
        raise RuntimeError(
            "训练数据加载失败：未读取到任何样本。请检查 corpus.train.txt 的格式是否为 id\t文本\t意图\t槽位 标注。"
        )
    split = int(len(all_data) * 0.8)
    train_data = all_data[:split]
    eval_data = all_data[split:]
    if len(eval_data) == 0:
        raise RuntimeError("训练集/验证集划分失败：数据量不足。")
    print(f"  训练集: {len(train_data)} 条 | 验证集: {len(eval_data)} 条")
    print(f"  示例: {train_data[0]}")

    print("\n[2/6] 加载 Tokenizer 和模型配置...")
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_intents = NUM_INTENTS
    config.num_slots = NUM_SLOTS
    config.intent_loss_weight = 1.0
    config.slot_loss_weight = 1.0

    print("\n[3/6] 构建数据集...")
    train_dataset = CarNLUDataset(train_data, tokenizer)
    eval_dataset = CarNLUDataset(eval_data, tokenizer)
    sample = train_dataset[0]
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  intent: {sample['intent_labels'].item()} ({ID2INTENT[sample['intent_labels'].item()]})")

    print("\n[4/6] 初始化模型...")
    model = JointBertModel.from_pretrained(model_name, config=config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型总参数: {total_params / 1e6:.1f}M")

    print("\n[5/6] 配置训练参数并启动训练...")
    eval_kwargs = {"eval_strategy": "epoch"} if _USE_EVAL_STRATEGY else {"evaluation_strategy": "epoch"}
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=8,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="slot_f1",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        report_to="none",
        dataloader_pin_memory=False,
        **eval_kwargs,
    )
    trainer = JointTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    final_dir = "./final_nlu_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n  最终模型已保存至: {final_dir}")

    print("\n[6/6] 推理验证...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    test_cases = [
        "导航到上海虹桥机场",
        "把空调温度调到24度",
        "打开主驾车窗",
        "播放周杰伦的晴天",
        "查一下深圳的天气",
        "打电话给张三",
        "关闭雨刷",
        "空调开除雾模式",
    ]
    for text in test_cases:
        result = predict(text, model, tokenizer, device)
        print(f"  输入: {result['text']}")
        print(f"  意图: {result['intent']} (置信度: {result['confidence']})")
        print(f"  槽位: {result['slots']}")
        print(f"  {'-' * 60}")

    print("\n  导出并验证 ONNX 模型...")
    export_onnx(model, tokenizer)

    print("\n" + "=" * 65)
    print("  训练完成! 产出物:")
    print(f"    1. PyTorch: {final_dir}/")
    print(f"    2. ONNX:    ./onnx_nlu_model/model.onnx")
    print(f"    3. 标签:    ./onnx_nlu_model/label_map.json")
    print("=" * 65)

if __name__ == "__main__":
    main()
