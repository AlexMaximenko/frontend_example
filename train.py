-----BEGIN TRAINING SCRIPT: train.py-----
#!/usr/bin/env python3
"""
Minimal requirements:
    transformers>=4.44.0
    datasets>=2.20.0
    accelerate>=0.33.0
    peft>=0.11.1
    torchaudio>=2.3.0
    soundfile>=0.12.1
    jiwer>=3.0.3
    einops
    bitsandbytes (optional for --use-4bit)

Example usages:
    python train.py --dataset spokenwoz --train-max-samples 2000 --val-max-samples 200 \
        --epochs 1 --batch-size 4 --grad-accum 4 --output-dir ./runs/audiollm_small

    torchrun --nproc_per_node=2 train.py --dataset spokenwoz --epochs 1 --ddp \
        --train-max-samples 4000 --val-max-samples 400 --output-dir ./runs/ddp_test

    python train.py --eval-mmlu --mmlu-max 1500 --resume ./runs/audiollm_small --dataset spokenwoz
"""

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import torchaudio
import soundfile as sf

from datasets import load_dataset
from jiwer import wer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    get_linear_schedule_with_warmup,
)
from transformers import WhisperFeatureExtractor, WhisperModel

SPECIAL_AUDIO_TOKEN = "<|AUDIO|>"


@dataclass
class TrainingConfig:
    dataset: str = "spokenwoz"
    train_max_samples: int = 2000
    val_max_samples: int = 200
    mmlu_max: int = 1500
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    grad_accum: int = 4
    use_4bit: bool = False
    amp: bool = True
    seed: int = 42
    output_dir: str = "./runs/audiollama"
    cache_dir: Optional[str] = None
    eval_mmlu: bool = False
    eval_every: int = 200
    save_every: int = 200
    resume: Optional[str] = None
    projector_dim: int = 1024
    n_audio_tokens: int = 12
    sample_rate: int = 16000
    max_duration_s: float = 12.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    clip_grad_norm: Optional[float] = 1.0
    ddp: bool = False
    num_workers: int = 2
    pin_memory: bool = True
    eval_split: str = "validation"
    amp_bf16: bool = True
    system_prompt: str = "You are an assistant for spoken multiple-choice exams."
    user_template: str = "<AUDIO> Select the correct option."
    assistant_prefix: str = "Answer:"
    infer_audio: Optional[str] = None
    save_state: bool = True
    ddp_find_unused: bool = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="AudioChatLLaMA-like training script")
    parser.add_argument("--dataset", choices=["spokenwoz", "slurp", "dailytalk", "commonvoice"], default="spokenwoz")
    parser.add_argument("--train-max-samples", type=int, default=2000)
    parser.add_argument("--val-max-samples", type=int, default=200)
    parser.add_argument("--mmlu-max", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./runs/audiollama")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--eval-mmlu", action="store_true")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--resume", type=str, default=None, help="Path to directory with adapters/audio_projector.pt")
    parser.add_argument("--projector-dim", type=int, default=1024)
    parser.add_argument("--n-audio-tokens", type=int, default=12)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-duration-s", type=float, default=12.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--ddp-find-unused", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--infer-audio", type=str, default=None)
    parser.add_argument("--save-state", action="store_true", default=True)
    parser.add_argument("--no-save-state", action="store_false", dest="save_state")
    args = parser.parse_args()

    cfg = TrainingConfig(
        dataset=args.dataset,
        train_max_samples=args.train_max_samples,
        val_max_samples=args.val_max_samples,
        mmlu_max=args.mmlu_max,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        use_4bit=args.use_4bit,
        amp=args.amp,
        seed=args.seed,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        eval_mmlu=args.eval_mmlu,
        eval_every=args.eval_every,
        save_every=args.save_every,
        resume=args.resume,
        projector_dim=args.projector_dim,
        n_audio_tokens=args.n_audio_tokens,
        sample_rate=args.sample_rate,
        max_duration_s=args.max_duration_s,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        clip_grad_norm=args.clip_grad_norm,
        ddp=args.ddp,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        infer_audio=args.infer_audio,
        save_state=args.save_state,
        ddp_find_unused=args.ddp_find_unused,
    )
    return cfg


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------


def load_llm_and_tokenizer(cfg: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        cache_dir=cfg.cache_dir,
        use_fast=False,
    )
    if SPECIAL_AUDIO_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_AUDIO_TOKEN]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16

    quant_config = None
    if cfg.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        cache_dir=cfg.cache_dir,
        torch_dtype=compute_dtype,
        quantization_config=quant_config,
        device_map="auto" if cfg.use_4bit else None,
    )
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, tokenizer


def load_frozen_whisper_encoder(cfg: TrainingConfig):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-medium",
        cache_dir=cfg.cache_dir,
    )
    whisper = WhisperModel.from_pretrained(
        "openai/whisper-medium",
        cache_dir=cfg.cache_dir,
    )
    whisper.encoder.requires_grad_(False)
    whisper.encoder.eval()
    return whisper.encoder, feature_extractor


class AudioProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_tokens: int, hidden_dim: int):
        super().__init__()
        self.n_tokens = n_tokens
        hidden_dim = hidden_dim or max(input_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_tokens * output_dim),
        )
        self.output_dim = output_dim

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = encoder_hidden_states.mean(dim=1)
        projected = self.net(pooled)
        projected = projected.view(-1, self.n_tokens, self.output_dim)
        return projected


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _resample_audio(audio: Dict[str, Any], target_sr: int) -> np.ndarray:
    array = audio["array"] if isinstance(audio, dict) else audio
    sr = audio.get("sampling_rate", target_sr) if isinstance(audio, dict) else target_sr
    if sr != target_sr:
        tensor = torch.tensor(array).float()
        array = torchaudio.functional.resample(tensor, sr, target_sr).numpy()
    return np.asarray(array, dtype=np.float32)


def _compute_features(batch: Dict[str, Any], feature_extractor, cfg: TrainingConfig) -> Dict[str, Any]:
    waveform = _resample_audio(batch["audio"], cfg.sample_rate)
    duration = waveform.shape[-1] / cfg.sample_rate
    if duration > cfg.max_duration_s:
        waveform = waveform[: int(cfg.max_duration_s * cfg.sample_rate)]
    batch["input_features"] = feature_extractor(
        waveform,
        sampling_rate=cfg.sample_rate,
        return_tensors="pt",
    )["input_features"][0]
    return batch


def _fallback_dataset(cfg: TrainingConfig, feature_extractor, num_samples: int = 200):
    rng = np.random.default_rng(cfg.seed)
    examples = []
    for _ in range(num_samples):
        waveform = rng.normal(0, 0.01, size=int(cfg.sample_rate * 2.5)).astype(np.float32)
        features = feature_extractor(
            waveform,
            sampling_rate=cfg.sample_rate,
            return_tensors="pt",
        )["input_features"][0]
        examples.append({
            "input_features": features,
            "text": "Synthetic answer placeholder.",
            "audio": {"array": waveform, "sampling_rate": cfg.sample_rate},
        })
    from datasets import Dataset

    return Dataset.from_list(examples)


def load_dataset_by_name(cfg: TrainingConfig, feature_extractor) -> Dataset:
    target = cfg.dataset.lower()
    name_map = {
        "spokenwoz": "facebook/SpokenWOZ",
        "slurp": "slurp",
        "dailytalk": "daily_dialog",
        "commonvoice": "mozilla-foundation/common_voice_11_0",
    }
    dataset_name = name_map.get(target, "facebook/SpokenWOZ")
    split = "train"
    if target == "commonvoice":
        split = "train[:2000]"
    try:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cfg.cache_dir)
    except Exception as exc:
        if is_main_process():
            print(f"Dataset {dataset_name} unavailable due to {exc}; using synthetic fallback")
        return _fallback_dataset(cfg, feature_extractor)

    def mapper(example):
        audio = None
        for key in ["audio", "speech", "target_speech", "spoken_audio", "sound"]:
            if key in example and example[key] is not None:
                audio = example[key]
                break
        if audio is None:
            return None
        text = example.get("text") or example.get("response") or example.get("sentence") or example.get("answer")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text if t)
        if text is None:
            text = ""
        return {"audio": audio, "text": text}

    dataset = dataset.map(mapper, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: x is not None and x["text"])
    dataset = dataset.map(lambda x: _compute_features(x, feature_extractor, cfg))
    return dataset


class SFTCollator:
    def __init__(self, tokenizer, cfg: TrainingConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_features = [torch.tensor(f["input_features"], dtype=torch.float32) for f in features]
        texts = [f["text"] for f in features]
        prompt_pairs = []
        for text in texts:
            system = self.cfg.system_prompt
            user = self.cfg.user_template.replace("<AUDIO>", SPECIAL_AUDIO_TOKEN)
            assistant = f"{self.cfg.assistant_prefix} {text}".strip()
            prompt_pairs.append((f"SYSTEM: {system}\nUSER: {user}\nASSISTANT:", assistant))

        input_ids = []
        loss_masks = []
        for prompt, answer in prompt_pairs:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            answer_ids = self.tokenizer(answer + self.tokenizer.eos_token, add_special_tokens=False).input_ids
            ids = prompt_ids + answer_ids
            mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            loss_masks.append(torch.tensor(mask, dtype=torch.long))

        max_len = max(t.size(0) for t in input_ids)
        padded_ids, padded_masks = [], []
        for ids, mask in zip(input_ids, loss_masks):
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            padded_ids.append(ids)
            padded_masks.append(mask)

        batch = {
            "input_features": torch.stack(audio_features),
            "input_ids": torch.stack(padded_ids),
            "loss_mask": torch.stack(padded_masks),
        }
        return batch


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def make_inputs_embeds_and_labels(
    cfg: TrainingConfig,
    model: AutoModelForCausalLM,
    tokenizer,
    audio_projector: AudioProjector,
    whisper_encoder,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
):
    input_features = batch["input_features"].to(device)
    input_ids = batch["input_ids"].to(device)
    loss_mask = batch["loss_mask"].to(device)

    with torch.no_grad():
        whisper_outputs = whisper_encoder(input_features)
        audio_hidden = whisper_outputs.last_hidden_state
    audio_tokens = audio_projector(audio_hidden)

    token_embeds = model.get_input_embeddings()(input_ids)
    audio_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_AUDIO_TOKEN)

    expanded_embeds = []
    expanded_labels = []
    expanded_masks = []
    for b in range(input_ids.size(0)):
        ids = input_ids[b]
        embeds = token_embeds[b]
        labels = ids.clone()
        labels[loss_mask[b] == 0] = -100
        seq_embeds: List[torch.Tensor] = []
        seq_labels: List[torch.Tensor] = []
        seq_mask: List[torch.Tensor] = []
        inserted = 0
        for idx, (token_id, token_embed, label_val) in enumerate(zip(ids, embeds, labels)):
            if token_id == audio_token_id:
                for j in range(audio_projector.n_audio_tokens):
                    seq_embeds.append(audio_tokens[b, j])
                    seq_labels.append(torch.tensor(-100, device=device))
                    seq_mask.append(torch.tensor(1, device=device))
                inserted += 1
            else:
                seq_embeds.append(token_embed)
                seq_labels.append(label_val)
                seq_mask.append(torch.tensor(1, device=device))
        seq_embeds = torch.stack(seq_embeds)
        seq_labels = torch.stack(seq_labels)
        seq_mask = torch.stack(seq_mask)
        expanded_embeds.append(seq_embeds)
        expanded_labels.append(seq_labels)
        expanded_masks.append(seq_mask)

    max_len = max(t.size(0) for t in expanded_embeds)
    padded_embeds, padded_labels, padded_masks = [], [], []
    for embeds, labels, mask in zip(expanded_embeds, expanded_labels, expanded_masks):
        pad_len = max_len - embeds.size(0)
        if pad_len > 0:
            embeds = torch.cat([embeds, torch.zeros((pad_len, embeds.size(1)), device=device)], dim=0)
            labels = torch.cat([labels, torch.full((pad_len,), -100, device=device, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len, device=device)], dim=0)
        padded_embeds.append(embeds)
        padded_labels.append(labels)
        padded_masks.append(mask)

    inputs_embeds = torch.stack(padded_embeds)
    attention_mask = torch.stack(padded_masks)
    labels = torch.stack(padded_labels)
    return inputs_embeds, attention_mask, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_epoch(
    cfg: TrainingConfig,
    model: AutoModelForCausalLM,
    tokenizer,
    audio_projector: AudioProjector,
    whisper_encoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    model.train()
    audio_projector.train()
    total_loss = 0.0
    step = 0

    for batch_idx, batch in enumerate(dataloader):
        inputs_embeds, attention_mask, labels = make_inputs_embeds_and_labels(
            cfg,
            model,
            tokenizer,
            audio_projector,
            whisper_encoder,
            batch,
            device,
        )

        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / cfg.grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % cfg.grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if cfg.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(audio_projector.parameters(), cfg.clip_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg.grad_accum
        step += 1
        if step % cfg.logging_steps == 0 and is_main_process():
            print(f"Epoch {epoch} Step {step} Loss {total_loss / step:.4f}")

    return total_loss / max(step, 1)


def run_training(cfg: TrainingConfig, model, tokenizer, audio_projector, whisper_encoder, train_dataset):
    sampler = None
    if cfg.ddp and dist.is_initialized():
        sampler = DistributedSampler(train_dataset, shuffle=True)
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=SFTCollator(tokenizer, cfg),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    optimizer = AdamW(
        list(model.parameters()) + list(audio_projector.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    num_training_steps = math.ceil(len(dataloader) / cfg.grad_accum) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and torch.cuda.is_available())
    device = next(model.parameters()).device

    global_step = 0
    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        avg_loss = train_epoch(
            cfg,
            model,
            tokenizer,
            audio_projector,
            whisper_encoder,
            dataloader,
            optimizer,
            scheduler,
            device,
            epoch,
            scaler,
        )
        if is_main_process():
            print(f"Epoch {epoch} avg loss {avg_loss:.4f}")
        best_loss = min(best_loss, avg_loss)
        global_step += len(dataloader)
        if is_main_process():
            save_checkpoint(cfg, model, audio_projector, global_step, avg_loss, best_loss)
    return best_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def generate_from_audio_batch(
    cfg: TrainingConfig,
    model: AutoModelForCausalLM,
    tokenizer,
    audio_projector: AudioProjector,
    whisper_encoder,
    audio_features: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
):
    device = next(model.parameters()).device
    with torch.no_grad():
        whisper_outputs = whisper_encoder(audio_features.to(device))
        audio_hidden = whisper_outputs.last_hidden_state
    audio_tokens = audio_projector(audio_hidden)

    prompts = []
    for _ in range(audio_features.size(0)):
        system = cfg.system_prompt
        user = cfg.user_template.replace("<AUDIO>", SPECIAL_AUDIO_TOKEN)
        prompts.append(f"SYSTEM: {system}\nUSER: {user}\nASSISTANT:")

    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    token_embeds = model.get_input_embeddings()(encoded.input_ids)
    audio_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_AUDIO_TOKEN)

    embeds_list = []
    mask_list = []
    for b in range(encoded.input_ids.size(0)):
        ids = encoded.input_ids[b]
        embeds = token_embeds[b]
        seq = []
        for token_id, token_embed in zip(ids, embeds):
            if token_id == audio_token_id:
                seq.append(audio_tokens[b])
            else:
                seq.append(token_embed.unsqueeze(0))
        seq = torch.cat(seq, dim=0)
        embeds_list.append(seq)
        mask_list.append(torch.ones(seq.size(0), device=device))

    max_len = max(t.size(0) for t in embeds_list)
    padded_embeds = []
    padded_masks = []
    for embeds, mask in zip(embeds_list, mask_list):
        pad_len = max_len - embeds.size(0)
        if pad_len > 0:
            embeds = torch.cat([embeds, torch.zeros((pad_len, embeds.size(1)), device=device)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_len, device=device)], dim=0)
        padded_embeds.append(embeds)
        padded_masks.append(mask)

    inputs_embeds = torch.stack(padded_embeds)
    attention_mask = torch.stack(padded_masks)

    outputs = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts


def evaluate_mmlu_speech(
    cfg: TrainingConfig,
    model: AutoModelForCausalLM,
    tokenizer,
    audio_projector: AudioProjector,
    whisper_encoder,
    feature_extractor,
    max_samples: Optional[int] = None,
):
    dataset = load_dataset("mistralai/mmlu_speech", split="validation", cache_dir=cfg.cache_dir)
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    device = next(model.parameters()).device

    total = 0
    correct = 0
    per_subject: Dict[str, Dict[str, int]] = {}
    qualitative: List[Dict[str, Any]] = []

    for batch in dataloader:
        audio_list = []
        for audio in batch["audio"]:
            waveform = _resample_audio(audio, cfg.sample_rate)
            feats = feature_extractor(
                waveform,
                sampling_rate=cfg.sample_rate,
                return_tensors="pt",
            )["input_features"][0]
            audio_list.append(feats)
        audio_tensor = torch.stack(audio_list).to(device)
        with torch.no_grad():
            whisper_outputs = whisper_encoder(audio_tensor)
            audio_hidden = whisper_outputs.last_hidden_state
        audio_tokens = audio_projector(audio_hidden)

        prompts = []
        prefix = "Listen to the audio question and select the correct answer (A/B/C/D). Return just the letter."
        for _ in range(audio_tensor.size(0)):
            prompts.append(
                f"SYSTEM: {cfg.system_prompt}\nUSER: {SPECIAL_AUDIO_TOKEN} {prefix} Select the correct option: A, B, C, or D. Reply with one letter.\nASSISTANT:"
            )

        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        token_embeds = model.get_input_embeddings()(encoded.input_ids)
        audio_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_AUDIO_TOKEN)

        embeds_list = []
        mask_list = []
        for b in range(encoded.input_ids.size(0)):
            ids = encoded.input_ids[b]
            embeds = token_embeds[b]
            seq = []
            for token_id, token_embed in zip(ids, embeds):
                if token_id == audio_token_id:
                    seq.append(audio_tokens[b])
                else:
                    seq.append(token_embed.unsqueeze(0))
            seq = torch.cat(seq, dim=0)
            embeds_list.append(seq)
            mask_list.append(torch.ones(seq.size(0), device=device))

        max_len = max(t.size(0) for t in embeds_list)
        padded_embeds = []
        padded_masks = []
        for embeds, mask in zip(embeds_list, mask_list):
            pad_len = max_len - embeds.size(0)
            if pad_len > 0:
                embeds = torch.cat([embeds, torch.zeros((pad_len, embeds.size(1)), device=device)], dim=0)
                mask = torch.cat([mask, torch.zeros(pad_len, device=device)], dim=0)
            padded_embeds.append(embeds)
            padded_masks.append(mask)

        inputs_embeds = torch.stack(padded_embeds)
        attention_mask = torch.stack(padded_masks)

        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=8,
            temperature=0.0,
        )
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers = batch["answer"]
        subjects = batch["subject"]
        for text, ans, subject in zip(texts, answers, subjects):
            match = re.search(r"\b([ABCD])\b", text.strip().upper())
            pred = match.group(1) if match else ""
            total += 1
            stats = per_subject.setdefault(subject, {"correct": 0, "total": 0})
            stats["total"] += 1
            if pred == ans:
                correct += 1
                stats["correct"] += 1
            if len(qualitative) < 10 and is_main_process():
                qualitative.append({"subject": subject, "prediction": pred, "gold": ans, "raw": text})

    overall = correct / max(total, 1)
    subject_scores = {
        subj: stats["correct"] / max(stats["total"], 1)
        for subj, stats in sorted(per_subject.items(), key=lambda item: item[0])
    }
    if is_main_process():
        print(f"MMLU-Speech overall accuracy: {overall:.4f}")
        for subject, score in list(subject_scores.items())[:10]:
            print(f"  {subject}: {score:.3f}")
        print("Qualitative samples:")
        for example in qualitative:
            print(example)
    return overall, subject_scores, qualitative


# ---------------------------------------------------------------------------
# Checkpointing & Resume
# ---------------------------------------------------------------------------


def save_checkpoint(cfg: TrainingConfig, model, audio_projector, global_step: int, loss: float, best_loss: float):
    if not is_main_process():
        return
    os.makedirs(cfg.output_dir, exist_ok=True)
    adapter_dir = Path(cfg.output_dir) / "adapters"
    projector_path = Path(cfg.output_dir) / "audio_projector.pt"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, DDP) else model
    model_to_save.save_pretrained(str(adapter_dir))
    torch.save(audio_projector.state_dict(), projector_path)
    if cfg.save_state:
        state_path = Path(cfg.output_dir) / "training_state.json"
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"global_step": global_step, "loss": loss, "best_loss": best_loss}, f, indent=2)
    if is_main_process():
        print(f"Saved checkpoint to {cfg.output_dir}")


def load_resume(cfg: TrainingConfig, model, audio_projector):
    if cfg.resume is None:
        return
    adapter_dir = Path(cfg.resume)
    projector_path = adapter_dir / "audio_projector.pt"
    adapter_subdir = adapter_dir / "adapters"
    if adapter_subdir.exists():
        model.load_adapter(str(adapter_subdir), adapter_name="default", is_trainable=True)
        model.set_adapter("default")
    if projector_path.exists():
        state = torch.load(projector_path, map_location="cpu")
        audio_projector.load_state_dict(state)
    if is_main_process():
        print(f"Resumed from {cfg.resume}")


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


def infer_from_audio(cfg: TrainingConfig, model, tokenizer, audio_projector, whisper_encoder, feature_extractor, path: str):
    device = next(model.parameters()).device
    waveform, sr = sf.read(path)
    if sr != cfg.sample_rate:
        waveform = torchaudio.functional.resample(torch.tensor(waveform).float(), sr, cfg.sample_rate).numpy()
    features = feature_extractor(
        waveform,
        sampling_rate=cfg.sample_rate,
        return_tensors="pt",
    )["input_features"].to(device)
    texts = generate_from_audio_batch(cfg, model, tokenizer, audio_projector, whisper_encoder, features)
    if is_main_process():
        print("Inference output:", texts[0])
    return texts[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = parse_args()
    if cfg.ddp:
        dist.init_process_group(backend="nccl")
    setup_seed(cfg.seed + (dist.get_rank() if dist.is_initialized() else 0))
    if is_main_process():
        print("Config:")
        print(json.dumps(asdict(cfg), indent=2))

    model, tokenizer = load_llm_and_tokenizer(cfg)
    whisper_encoder, feature_extractor = load_frozen_whisper_encoder(cfg)
    audio_projector = AudioProjector(
        input_dim=whisper_encoder.config.d_model,
        output_dim=model.config.hidden_size,
        n_tokens=cfg.n_audio_tokens,
        hidden_dim=cfg.projector_dim,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    audio_projector = audio_projector.to(device)
    whisper_encoder = whisper_encoder.to(device)

    load_resume(cfg, model, audio_projector)

    if cfg.infer_audio:
        infer_from_audio(cfg, model, tokenizer, audio_projector, whisper_encoder, feature_extractor, cfg.infer_audio)
        if cfg.ddp and dist.is_initialized():
            dist.barrier()
        return

    train_dataset = load_dataset_by_name(cfg, feature_extractor)
    if cfg.train_max_samples:
        train_dataset = train_dataset.select(range(min(len(train_dataset), cfg.train_max_samples)))

    if cfg.epochs > 0:
        run_training(cfg, model, tokenizer, audio_projector, whisper_encoder, train_dataset)

    if cfg.eval_mmlu:
        overall, subject_scores, qualitative = evaluate_mmlu_speech(
            cfg,
            model,
            tokenizer,
            audio_projector,
            whisper_encoder,
            feature_extractor,
            max_samples=cfg.mmlu_max,
        )
        if is_main_process():
            metrics_path = Path(cfg.output_dir) / "metrics_mmlu.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({
                    "overall_accuracy": overall,
                    "per_subject": subject_scores,
                    "examples": qualitative,
                }, f, indent=2)
            print(f"Saved MMLU metrics to {metrics_path}")

    if cfg.ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
-----END TRAINING SCRIPT: train.py-----
