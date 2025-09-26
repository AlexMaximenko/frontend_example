-----BEGIN COLAB STUDENT NOTEBOOK-----
#=== Cell 0: Setup & Installs
"""Student Colab notebook for the AudioChatLLaMA-style pipeline."""
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
    except Exception:
        print("bitsandbytes unavailable; continue without 4-bit mode.")

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
    WhisperFeatureExtractor,
    WhisperModel,
    get_linear_schedule_with_warmup,
)

#=== Cell 1: Config (CFG) with toggles/paths
@dataclass
class CFG:
    cache_dir: Optional[str] = None
    output_dir: str = "./audiollama_outputs_student"
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
    return model, tokenizer


model, tokenizer = load_llm_and_tokenizer(cfg)
print("LLM with LoRA ready.")

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
print("Whisper encoder loaded.")

#=== Cell 4: Define AudioSubsamplingProjector (TODO forward)


class AudioSubsamplingProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        subsample_factor: int = 2,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim, output_dim)
        self.subsample_factor = subsample_factor
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """TODO: subsample encoder_hidden_states along time before projecting."""
        raise NotImplementedError("TODO: implement AudioSubsamplingProjector.forward")


audio_projector = AudioSubsamplingProjector(
    input_dim=whisper_encoder.config.d_model,
    output_dim=model.config.hidden_size,
    subsample_factor=cfg.audio_subsample,
    hidden_dim=cfg.projector_hidden,
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
        """TODO: replace SPECIAL_AUDIO tokens with projected audio embeddings."""
        raise NotImplementedError("TODO: implement AudioLLMModel._merge_sequences")

    def prepare_inputs_and_labels(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TODO: encode audio, embed text, and build training inputs/labels."""
        raise NotImplementedError("TODO: implement AudioLLMModel.prepare_inputs_and_labels")

    def prepare_generation_inputs(
        self,
        prompts: List[str],
        audio_features: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: build inputs_embeds + attention_mask for generation."""
        raise NotImplementedError("TODO: implement AudioLLMModel.prepare_generation_inputs")

    def generate(
        self,
        prompts: List[str],
        audio_features: torch.Tensor,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """TODO: call self.unwrap(self.llm).generate with merged audio/text inputs."""
        raise NotImplementedError("TODO: implement AudioLLMModel.generate")


audio_llm = AudioLLMModel(
    llm=model,
    whisper_encoder=whisper_encoder,
    audio_projector=audio_projector,
    tokenizer=tokenizer,
    audio_token=SPECIAL_AUDIO_TOKEN,
)

#=== Cell 5: Dataset loaders with SpokenWOZ + fallbacks

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
    dummy_texts = [
        "Synthetic knowledge sample.",
        "Placeholder response for audio.",
    ]
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
            "text": random.choice(dummy_texts),
            "audio": {"array": waveform, "sampling_rate": cfg.sample_rate},
        })
    return Dataset.from_list(examples)


def load_spokenwoz(cfg: CFG, feature_extractor) -> Dataset:
    try_names = ["facebook/SpokenWOZ", "LIUM/SpokenWOZ"]
    dataset = None
    for name in try_names:
        try:
            dataset = load_dataset(name, split="train", cache_dir=cfg.cache_dir)
            break
        except Exception:
            continue
    if dataset is None:
        print("Falling back to synthetic SpokenWOZ subset.")
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

#=== Cell 7: Prompt embeddings builder (TODO make_inputs_embeds_and_labels)

def make_inputs_embeds_and_labels(
    cfg: CFG,
    audio_model: AudioLLMModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
):
    """Combine audio features with text tokens to build model inputs."""
    # TODO: call audio_model.prepare_inputs_and_labels and return its outputs
    raise NotImplementedError("TODO: implement make_inputs_embeds_and_labels")


#=== Cell 8: Training loop (TODO)
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
    """Single epoch training with gradient accumulation and AMP."""
    # TODO: iterate over train_loader, use make_inputs_embeds_and_labels(cfg, audio_llm, ...),
    #       compute loss with audio_llm.llm, backprop, optimizer/scheduler steps, and log every cfg.logging_steps
    raise NotImplementedError("TODO: implement train_epoch")


for epoch in range(cfg.epochs):
    # TODO: call train_epoch(epoch) and report averaged loss
    pass

#=== Cell 9: Save adapters (TODO)
# TODO: unwrap accelerator model, save adapters to cfg.output_dir/adapters, save audio_projector state_dict

#=== Cell 10: MMLU-Speech evaluation (TODO)

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
    """Evaluate zero-shot multiple-choice accuracy on mistralai/mmlu_speech."""
    # TODO: iterate over evaluation DataLoader, build prompts with SPECIAL_AUDIO_TOKEN,
    #       call audio_model.generate, parse answers, compute metrics, log qualitative samples.
    raise NotImplementedError("TODO: implement evaluate_mmlu_speech")


#=== Cell 11: Inference helper (TODO)
# TODO: implement transcribe_and_answer that loads a WAV, extracts features, calls generate_from_audio_batch
-----END COLAB STUDENT NOTEBOOK-----
