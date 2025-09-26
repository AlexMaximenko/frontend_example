-----BEGIN COLAB TEACHER NOTEBOOK-----
#=== Cell 0: Setup & Installs
"""Teacher Colab notebook for lightweight AudioChatLLaMA-style pipeline."""
import sys
import subprocess
import pkgutil

REQUIRED_PACKAGES = {
    "transformers": "4.44.2",
    "datasets": "2.20.0",
    "accelerate": "0.33.0",
    "peft": "0.11.1",
    "torchaudio": "2.3.1",
    "soundfile": "0.12.1",
    "jiwer": "3.0.3",
    "einops": "0.7.0",
}

for pkg, ver in REQUIRED_PACKAGES.items():
    if pkgutil.find_loader(pkg) is None:
        print(f"Installing {pkg}=={ver}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{pkg}=={ver}"])

try:
    import bitsandbytes  # noqa: F401
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes==0.43.1"])
    except Exception as exc:
        print("bitsandbytes install failed, continuing without it:", exc)

import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchaudio
import soundfile as sf

from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    WhisperFeatureExtractor,
    WhisperModel,
    get_linear_schedule_with_warmup,
)

#=== Cell 1: Config (CFG) with toggles/paths
@dataclass
class CFG:
    cache_dir: Optional[str] = None
    output_dir: str = "./audiollama_outputs"
    system_prompt: str = "You are an assistant for spoken multiple-choice exams."
    user_prompt_template: str = "<AUDIO> Select the best answer to the spoken question."
    assistant_prefix: str = "Answer:"
    sample_rate: int = 16000
    max_duration_s: float = 12.0
    audio_subsample: int = 2
    projector_hidden: int = 1024
    train_max_sft_samples: int = 2000
    val_max_sft_samples: int = 200
    mmlu_speech_max: int = 1500
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 1
    batch_size: int = 4
    grad_accum_steps: int = 4
    amp: bool = True
    use_4bit: bool = False
    seed: int = 42
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_every: int = 200
    eval_every: int = 200
    train_use_slurp: bool = False
    train_use_dailytalk: bool = False
    train_use_commonvoice: bool = False
    num_workers: int = 2
    gradient_checkpointing: bool = True

cfg = CFG()
os.makedirs(cfg.output_dir, exist_ok=True)
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

SPECIAL_AUDIO_TOKEN = "<|AUDIO|>"

#=== Cell 2: Load tokenizer + LLM (Qwen3-0.6B-Base) and add LoRA

def load_llm_and_tokenizer(cfg: CFG):
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
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model, tokenizer, compute_dtype


model, tokenizer, compute_dtype = load_llm_and_tokenizer(cfg)
print("Loaded Qwen3 0.6B with LoRA adapters.")

#=== Cell 3: Load Whisper-medium encoder & feature_extractor (frozen)

def load_frozen_whisper_encoder(cfg: CFG):
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


whisper_encoder, whisper_feature_extractor = load_frozen_whisper_encoder(cfg)
print("Whisper encoder hidden size:", whisper_encoder.config.d_model)

#=== Cell 4: Define AudioProjector (in_dim = whisper d_model, out_dim = LLM hidden, N tokens)


class AudioSubsamplingProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        subsample_factor: int = 2,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if subsample_factor < 1:
            raise ValueError("subsample_factor must be >= 1")
        self.subsample_factor = subsample_factor
        hidden_dim = hidden_dim or max(input_dim, output_dim)
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # encoder_hidden_states: (batch, time, dim)
        x = encoder_hidden_states
        if self.subsample_factor > 1:
            x = x.transpose(1, 2)
            x = torch.nn.functional.avg_pool1d(
                x,
                kernel_size=self.subsample_factor,
                stride=self.subsample_factor,
                ceil_mode=True,
            )
            x = x.transpose(1, 2)
        x = self.proj_out(self.act(self.proj_in(x)))
        return x


audio_projector = AudioSubsamplingProjector(
    input_dim=whisper_encoder.config.d_model,
    output_dim=model.config.hidden_size,
    subsample_factor=cfg.audio_subsample,
    hidden_dim=cfg.projector_hidden,
)
print(audio_projector)

audio_llm = AudioLLMModel(
    llm=model,
    whisper_encoder=whisper_encoder,
    audio_projector=audio_projector,
    tokenizer=tokenizer,
    audio_token=SPECIAL_AUDIO_TOKEN,
)


class AudioLLMModel(nn.Module):
    def __init__(
        self,
        llm: AutoModelForCausalLM,
        whisper_encoder: WhisperModel,
        audio_projector: nn.Module,
        tokenizer,
        audio_token: str,
    ):
        super().__init__()
        self.llm = llm
        self.whisper_encoder = whisper_encoder
        self.audio_projector = audio_projector
        self.tokenizer = tokenizer
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)

    def unwrap(self, module: nn.Module) -> nn.Module:
        return getattr(module, "module", module)

    def _input_embedding_layer(self):
        return self.unwrap(self.llm).get_input_embeddings()

    def encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        device = next(self.llm.parameters()).device
        input_features = input_features.to(device)
        with torch.no_grad():
            outputs = self.whisper_encoder(input_features)
        return outputs.last_hidden_state

    def project_audio(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.audio_projector(encoder_hidden_states)
        embed_dtype = self._input_embedding_layer().weight.dtype
        return projected.to(embed_dtype)

    def trainable_parameters(self):
        for module in (self.llm, self.audio_projector):
            for param in module.parameters():
                if param.requires_grad:
                    yield param

    def _merge_sequences(
        self,
        input_ids: torch.Tensor,
        token_embeds: torch.Tensor,
        audio_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_embeds: List[torch.Tensor] = []
        batch_masks: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = [] if labels is not None else None

        for idx in range(input_ids.size(0)):
            seq_embeds: List[torch.Tensor] = []
            seq_mask: List[torch.Tensor] = []
            seq_labels: List[torch.Tensor] = [] if labels is not None else None
            audio_seq = audio_embeddings[idx]
            label_seq = labels[idx] if labels is not None else None
            audio_inserted = False

            for token_position, (token_id, embed) in enumerate(zip(input_ids[idx], token_embeds[idx])):
                if token_id.item() == self.audio_token_id:
                    seq_embeds.append(audio_seq)
                    seq_mask.append(torch.ones(audio_seq.size(0), device=embed.device, dtype=torch.long))
                    if labels is not None:
                        seq_labels.append(
                            torch.full((audio_seq.size(0),), -100, device=embed.device, dtype=torch.long)
                        )
                    audio_inserted = True
                else:
                    seq_embeds.append(embed.unsqueeze(0))
                    seq_mask.append(torch.ones(1, device=embed.device, dtype=torch.long))
                    if labels is not None:
                        seq_labels.append(label_seq[token_position].view(1))

            if not audio_inserted:
                raise ValueError("Input sequence is missing the audio special token.")

            seq_embeds_tensor = torch.cat(seq_embeds, dim=0)
            seq_mask_tensor = torch.cat(seq_mask, dim=0)
            batch_embeds.append(seq_embeds_tensor)
            batch_masks.append(seq_mask_tensor)
            if labels is not None and seq_labels is not None:
                seq_labels_tensor = torch.cat(seq_labels, dim=0)
                batch_labels.append(seq_labels_tensor)

        padded_embeds = torch.nn.utils.rnn.pad_sequence(batch_embeds, batch_first=True)
        padded_masks = torch.nn.utils.rnn.pad_sequence(batch_masks, batch_first=True)
        padded_labels = None
        if batch_labels is not None:
            padded_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        return padded_embeds, padded_masks, padded_labels

    def prepare_inputs_and_labels(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_features = batch["input_features"].to(device)
        input_ids = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        audio_hidden = self.encode_audio(input_features)
        audio_embeddings = self.project_audio(audio_hidden)
        embedding_layer = self._input_embedding_layer()
        token_embeds = embedding_layer(input_ids)
        labels = input_ids.clone()
        labels[loss_mask == 0] = -100
        inputs_embeds, attention_mask, labels = self._merge_sequences(
            input_ids,
            token_embeds,
            audio_embeddings,
            labels,
        )
        return inputs_embeds, attention_mask, labels

    def prepare_generation_inputs(
        self,
        prompts: List[str],
        audio_features: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)

        audio_hidden = self.encode_audio(audio_features.to(device))
        audio_embeddings = self.project_audio(audio_hidden)
        token_embeds = self._input_embedding_layer()(encoded.input_ids)
        inputs_embeds, attention_mask, _ = self._merge_sequences(
            encoded.input_ids,
            token_embeds,
            audio_embeddings,
        )
        return inputs_embeds, attention_mask

    def generate(
        self,
        prompts: List[str],
        audio_features: torch.Tensor,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        generation_kwargs = generation_kwargs or {}
        model = self.unwrap(self.llm)
        device = next(self.llm.parameters()).device
        inputs_embeds, attention_mask = self.prepare_generation_inputs(
            prompts,
            audio_features,
            device,
        )
        default_kwargs = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        default_kwargs.update(generation_kwargs)
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **default_kwargs,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

#=== Cell 5: Dataset loaders with SpokenWOZ + optional fallbacks

def _resample_audio(audio: Dict[str, Any], target_sr: int) -> np.ndarray:
    array = audio["array"] if isinstance(audio, dict) else audio
    sr = audio.get("sampling_rate", target_sr) if isinstance(audio, dict) else target_sr
    if sr != target_sr:
        tensor = torch.tensor(array).float()
        array = torchaudio.functional.resample(tensor, sr, target_sr).numpy()
    return np.asarray(array, dtype=np.float32)


def _compute_features(batch: Dict[str, Any], feature_extractor, cfg: CFG) -> Dict[str, Any]:
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


def _fallback_dataset(cfg: CFG, feature_extractor, num_samples: int = 200) -> Dataset:
    rng = np.random.default_rng(cfg.seed)
    examples = []
    dummy_texts = [
        "The capital of France is Paris.",
        "Two plus two equals four.",
        "Water boils at one hundred degrees Celsius.",
        "Photosynthesis occurs in plant chloroplasts.",
    ]
    for _ in range(num_samples):
        waveform = rng.normal(0, 0.01, size=int(cfg.sample_rate * 2.5)).astype(np.float32)
        features = feature_extractor(
            waveform,
            sampling_rate=cfg.sample_rate,
            return_tensors="pt",
        )["input_features"][0]
        examples.append({
            "input_features": features,
            "text": random.choice(dummy_texts),
            "audio": {"array": waveform, "sampling_rate": cfg.sample_rate},
        })
    return Dataset.from_list(examples)


def load_spokenwoz(cfg: CFG, feature_extractor) -> Dataset:
    tried = ["facebook/SpokenWOZ", "LIUM/SpokenWOZ"]
    dataset = None
    last_error = None
    for name in tried:
        try:
            dataset = load_dataset(name, split="train", cache_dir=cfg.cache_dir)
            break
        except Exception as exc:
            last_error = exc
    if dataset is None:
        print("Falling back to synthetic dataset:", last_error)
        return _fallback_dataset(cfg, feature_extractor)

    def mapper(example):
        audio = None
        for key in ["audio", "speech", "target_speech", "spoken_audio"]:
            if key in example and example[key] is not None:
                audio = example[key]
                break
        if audio is None:
            return None
        text = example.get("text") or example.get("response") or example.get("answer") or ""
        if isinstance(text, list):
            text = " ".join([t for t in text if t])
        return {"audio": audio, "text": text}

    dataset = dataset.map(mapper, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: x is not None and x["text"])
    dataset = dataset.map(lambda x: _compute_features(x, feature_extractor, cfg))
    return dataset


def load_datasets(cfg: CFG, feature_extractor) -> Tuple[Dataset, Dataset]:
    train_dataset = load_spokenwoz(cfg, feature_extractor)
    if cfg.train_use_slurp:
        try:
            slurp = load_dataset("slurp", split="train[:200]", cache_dir=cfg.cache_dir)
            slurp = slurp.map(lambda x: {"audio": x["audio"], "text": x.get("sentence", "")})
            slurp = slurp.map(lambda x: _compute_features(x, feature_extractor, cfg))
            train_dataset = Dataset.from_list(train_dataset.to_list() + slurp.to_list())
        except Exception as exc:
            print("Failed to load SLURP:", exc)
    if cfg.train_max_sft_samples:
        train_dataset = train_dataset.select(range(min(len(train_dataset), cfg.train_max_sft_samples)))
    val_dataset = train_dataset.take(min(len(train_dataset), cfg.val_max_sft_samples))
    return train_dataset, val_dataset


train_dataset, val_dataset = load_datasets(cfg, whisper_feature_extractor)
print(train_dataset)

#=== Cell 6: Collator & preprocessing


class SFTCollator:
    def __init__(self, tokenizer, cfg: CFG):
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_features = [torch.tensor(f["input_features"], dtype=torch.float32) for f in features]
        texts = [f["text"] for f in features]
        prompt_pairs = []
        for text in texts:
            system = cfg.system_prompt
            user = cfg.user_prompt_template.replace("<AUDIO>", SPECIAL_AUDIO_TOKEN)
            assistant = f"{cfg.assistant_prefix} {text}".strip()
            prompt_pairs.append((f"SYSTEM: {system}\nUSER: {user}\nASSISTANT:", assistant))

        input_ids = []
        loss_masks = []
        for prompt, answer in prompt_pairs:
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False).input_ids
            ids = prompt_ids + answer_ids
            mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            loss_masks.append(torch.tensor(mask, dtype=torch.long))

        max_len = max(t.size(0) for t in input_ids)
        padded_ids, padded_masks = [], []
        for ids, mask in zip(input_ids, loss_masks):
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            padded_ids.append(ids)
            padded_masks.append(mask)

        return {
            "input_features": torch.stack(audio_features),
            "input_ids": torch.stack(padded_ids),
            "loss_mask": torch.stack(padded_masks),
        }


collator = SFTCollator(tokenizer, cfg)

#=== Cell 7: Prompt embeddings builder (make_inputs_embeds_and_labels)

def make_inputs_embeds_and_labels(
    cfg: CFG,
    audio_model: AudioLLMModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
):
    return audio_model.prepare_inputs_and_labels(batch, device)


#=== Cell 8: Training loop (loss prints every N steps)
accelerator = Accelerator()
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    collate_fn=collator,
    num_workers=cfg.num_workers,
)
optimizer = AdamW(
    list(audio_llm.trainable_parameters()),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)
num_training_steps = math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(cfg.warmup_ratio * num_training_steps),
    num_training_steps=num_training_steps,
)

prepared_llm, prepared_projector, optimizer, train_loader, scheduler = accelerator.prepare(
    audio_llm.llm,
    audio_llm.audio_projector,
    optimizer,
    train_loader,
    scheduler,
)
audio_llm.llm = prepared_llm
audio_llm.audio_projector = prepared_projector
audio_llm.whisper_encoder = audio_llm.whisper_encoder.to(accelerator.device)
audio_llm.whisper_encoder.eval()

def train_epoch(epoch: int):
    audio_llm.llm.train()
    audio_llm.audio_projector.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and accelerator.device.type == "cuda")

    for step, batch in enumerate(train_loader, start=1):
        with accelerator.accumulate(audio_llm.llm):
            inputs_embeds, attention_mask, labels = make_inputs_embeds_and_labels(
                cfg,
                audio_llm,
                batch,
                accelerator.device,
            )
            with torch.cuda.amp.autocast(enabled=cfg.amp and accelerator.device.type == "cuda"):
                outputs = audio_llm.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        if step % cfg.logging_steps == 0 and accelerator.is_main_process:
            print(f"Epoch {epoch} Step {step} Loss {total_loss / step:.4f}")
    return total_loss / max(step, 1)


for epoch in range(cfg.epochs):
    avg_loss = train_epoch(epoch)
if accelerator.is_main_process:
    print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

#=== Cell 9: Save adapters (PEFT dir) + audio_projector.pt
if accelerator.is_main_process:
    adapter_dir = os.path.join(cfg.output_dir, "adapters")
    projector_path = os.path.join(cfg.output_dir, "audio_projector.pt")
    os.makedirs(adapter_dir, exist_ok=True)
    base_model = accelerator.unwrap_model(audio_llm.llm)
    base_model.save_pretrained(adapter_dir)
    projector_to_save = accelerator.unwrap_model(audio_llm.audio_projector)
    torch.save(projector_to_save.state_dict(), projector_path)
    print("Saved adapters to", adapter_dir)
    print("Saved audio projector to", projector_path)

#=== Cell 10: MMLU-Speech evaluation utilities & run (overall accuracy, per-subject, 10 examples)

def generate_from_audio_batch(
    cfg: CFG,
    audio_model: AudioLLMModel,
    audio_features: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
):
    prompts = []
    for _ in range(audio_features.size(0)):
        system = cfg.system_prompt
        user = cfg.user_prompt_template.replace("<AUDIO>", SPECIAL_AUDIO_TOKEN)
        prompts.append(f"SYSTEM: {system}\nUSER: {user}\nASSISTANT:")

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
    }
    return audio_model.generate(prompts, audio_features, generation_kwargs)


def evaluate_mmlu_speech(
    cfg: CFG,
    audio_model: AudioLLMModel,
    feature_extractor,
    max_samples: Optional[int] = None,
):
    dataset = load_dataset("mistralai/mmlu_speech", split="validation", cache_dir=cfg.cache_dir)
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    total = 0
    correct = 0
    per_subject: Dict[str, Dict[str, int]] = {}
    qualitative = []

    for batch in dataloader:
        audio_list = []
        for audio in batch["audio"]:
            waveform = _resample_audio(audio, cfg.sample_rate)
            features = feature_extractor(
                waveform,
                sampling_rate=cfg.sample_rate,
                return_tensors="pt",
            )["input_features"][0]
            audio_list.append(features.to(torch.float32))
        audio_tensor = torch.stack(audio_list)

        prompts = []
        prefix = "Listen to the audio question and select the correct answer (A/B/C/D). Return just the letter."
        for _ in range(audio_tensor.size(0)):
            prompts.append(
                f"SYSTEM: {cfg.system_prompt}\nUSER: {SPECIAL_AUDIO_TOKEN} {prefix} Select the correct option: A, B, C, or D. Reply with one letter.\nASSISTANT:"
            )

        texts = audio_model.generate(
            prompts,
            audio_tensor,
            {"max_new_tokens": 8, "temperature": 0.0, "do_sample": False},
        )
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
            if len(qualitative) < 10:
                qualitative.append({"subject": subject, "prediction": pred, "gold": ans, "raw": text})

    overall_acc = correct / max(total, 1)
    per_subject_acc = {
        subject: stats["correct"] / max(stats["total"], 1)
        for subject, stats in sorted(per_subject.items(), key=lambda item: item[0])
    }
    print("MMLU-Speech overall accuracy:", overall_acc)
    print("Per-subject accuracy (first 5):")
    for subject, acc in list(per_subject_acc.items())[:5]:
        print(subject, f"{acc:.3f}")
    print("Qualitative examples:")
    for example in qualitative:
        print(example)
    return overall_acc, per_subject_acc, qualitative


evaluate_mmlu_speech(
    cfg,
    audio_llm,
    whisper_feature_extractor,
    cfg.mmlu_speech_max,
)

#=== Cell 11: Inference helper for a local wav

def transcribe_and_answer(wav_path: str):
    waveform, sr = sf.read(wav_path)
    if sr != cfg.sample_rate:
        waveform = torchaudio.functional.resample(torch.tensor(waveform).float(), sr, cfg.sample_rate).numpy()
    features = whisper_feature_extractor(
        waveform,
        sampling_rate=cfg.sample_rate,
        return_tensors="pt",
    )["input_features"]
    outputs = generate_from_audio_batch(
        cfg,
        audio_llm,
        features,
    )
    return outputs[0]


print("Inference helper ready. Call transcribe_and_answer('path_to_audio.wav').")
-----END COLAB TEACHER NOTEBOOK-----
