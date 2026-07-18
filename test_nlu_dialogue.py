#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多轮对话方式测试训练好的 car NLU 模型。

运行示例:
  .venv/Scripts/python.exe test_nlu_dialogue.py --model-dir final_nlu_model
"""

import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, BertModel, BertPreTrainedModel

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
ID2INTENT = {i: label for i, label in enumerate(INTENT_LABELS)}
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
ID2SLOT = {i: label for i, label in enumerate(SLOT_LABELS)}
MAX_LEN = 64


class JointBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_intents = config.num_intents
        self.num_slots = config.num_slots
        self.intent_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, self.num_intents)
        self.slot_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.slot_classifier = nn.Linear(config.hidden_size, self.num_slots)
        self.bert = BertModel(config)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]
        intent_logits = self.intent_classifier(self.intent_dropout(pooled_output))
        slot_logits = self.slot_classifier(self.slot_dropout(sequence_output))
        return {
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
        }


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


def load_nlu_model(model_dir, device="cpu"):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    config.num_intents = len(INTENT_LABELS)
    config.num_slots = len(SLOT_LABELS)
    model = JointBertModel.from_pretrained(model_dir, config=config)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict(text, model, tokenizer, device="cpu"):
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
        "intent": ID2INTENT.get(intent_id, "other"),
        "slots": slots,
        "confidence": float(intent_probs[intent_id].item()),
    }


def run_console(model, tokenizer, device="cpu"):
    print("\n=== NLU 模型多轮对话测试 ===")
    print("输入文本后按回车预测，输入 'exit' 或 'quit' 结束，输入 'history' 查看对话历史。\n")
    history = []
    while True:
        try:
            text = input("用户: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出测试。")
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit", "q"}:
            print("退出测试。")
            break
        if text.lower() == "history":
            if not history:
                print("无对话历史。")
                continue
            for idx, item in enumerate(history, 1):
                print(f"{idx}. 用户: {item['text']} -> 意图: {item['intent']} 置信度: {item['confidence']}")
                print(f"   槽位: {item['slots']}")
            continue
        result = predict(text, model, tokenizer, device)
        history.append(result)
        print(f"意图: {result['intent']} 置信度: {result['confidence']:.4f}")
        print(f"槽位: {result['slots']}\n")


def main():
    parser = argparse.ArgumentParser(description="测试训练好的车载 NLU 模型")
    parser.add_argument("--model-dir", default="./final_nlu_model",
                        help="训练后模型目录，默认 ./final_nlu_model")
    parser.add_argument("--device", default=None,
                        help="运行设备，默认自动选择 gpu 或 cpu")
    parser.add_argument("--test", action="store_true",
                        help="仅测试模型加载和一次预测，然后退出")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载模型: {args.model_dir} 设备: {device}")
    model, tokenizer = load_nlu_model(args.model_dir, device=device)

    if args.test:
        sample = "我想听周杰伦的歌"
        print("测试样例：", sample)
        result = predict(sample, model, tokenizer, device)
        print(result)
        return

    run_console(model, tokenizer, device)


if __name__ == "__main__":
    main()
