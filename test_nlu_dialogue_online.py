#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Online NLU model dialogue tester.

This script downloads a JointBERT NLU model from the Hugging Face Hub
and runs a simple interactive conversation loop for intent + slot prediction.

该脚本从 Hugging Face Hub 拉取在线模型，并提供基础对话形式的测试接口。
"""

import argparse
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
    for idx, char in enumerate(chars):
        if idx >= len(slot_ids):
            break
        tag = ID2SLOT.get(slot_ids[idx], "O")
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
                current_value = ""
    if current_slot:
        slots[current_slot] = current_value
    return slots


def load_online_model(repo_id, device="cpu", auth_token=None):
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=auth_token)
    config = AutoConfig.from_pretrained(repo_id, use_auth_token=auth_token)
    config.num_intents = len(INTENT_LABELS)
    config.num_slots = len(SLOT_LABELS)
    model = JointBertModel.from_pretrained(repo_id, config=config, use_auth_token=auth_token)
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
    intent_id = int(torch.argmax(intent_probs).item())
    slot_ids = torch.argmax(slot_logits, dim=-1).tolist()
    slots = _bio_decode(chars, slot_ids)
    return {
        "text": text,
        "intent": ID2INTENT.get(intent_id, "other"),
        "slots": slots,
        "confidence": float(intent_probs[intent_id].item()),
    }


def run_console(model, tokenizer, device="cpu"):
    print("\n=== Online NLU Dialogue Test ===")
    print("Type a Chinese command and press Enter. Type 'exit' or 'quit' to stop. Type 'history' to show conversation history.\n")
    history = []
    while True:
        try:
            text = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting dialogue test.")
            break
        if not text:
            continue
        command = text.lower()
        if command in {"exit", "quit", "q"}:
            print("Exiting dialogue test.")
            break
        if command == "history":
            if not history:
                print("No history yet.\n")
                continue
            for idx, item in enumerate(history, 1):
                print(f"{idx}. User: {item['text']}")
                print(f"   Intent: {item['intent']} confidence={item['confidence']:.4f}")
                print(f"   Slots: {item['slots']}\n")
            continue
        result = predict(text, model, tokenizer, device)
        history.append(result)
        print(f"Intent: {result['intent']} confidence={result['confidence']:.4f}")
        print(f"Slots: {result['slots']}\n")


def main():
    parser = argparse.ArgumentParser(description="Load an online NLU model from Hugging Face and run a dialogue tester.")
    parser.add_argument(
        "--repo-id",
        default="mhhyoucom/bert-model-car-assistant",
        help="Hugging Face model repo ID, e.g. username/model-name",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on, e.g. cpu or cuda. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Optional Hugging Face token for private repos or authenticated access.",
    )
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading online model: {args.repo_id} on device: {device}")
    model, tokenizer = load_online_model(args.repo_id, device=device, auth_token=args.auth_token)
    run_console(model, tokenizer, device)


if __name__ == "__main__":
    main()
