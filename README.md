# Qwen3.5-9B · Fast Local Inference on an RTX 5070

![llama.cpp](https://img.shields.io/badge/llama.cpp-b9505-3b82f6)
![CUDA](https://img.shields.io/badge/CUDA-13.3-10b981)
![GPU](https://img.shields.io/badge/GPU-RTX%205070%20(12GB)-8b5cf6)
![Format](https://img.shields.io/badge/format-GGUF-ef4444)
![Quant](https://img.shields.io/badge/quant-UD--Q4_K_XL-f59e0b)

A practical, fully-tuned setup for running **Qwen3.5-9B** locally on an **NVIDIA GeForce RTX 5070 (12 GB, Blackwell)** using **llama.cpp (CUDA 13.3)** and an **Unsloth UD-Q4_K_XL** quantized GGUF.

**Benchmarked result: ~88 tokens/s generation and ~3,500 tokens/s prompt processing** on this GPU.

This repo is meant to be read as a guide: not just *what* changed, but *why*. Every flag, quant choice, and tradeoff is explained below.

---

## Table of Contents

1. [Why llama.cpp (not Ollama)](#why-llamacpp-not-ollama)
2. [Hardware requirements](#hardware-requirements)
3. [Model & quantization choice](#model--quantization-choice)
4. [Setup](#setup)
5. [Performance tuning](#performance-tuning)
6. [Performance results](#performance-results)
7. [Findings & takeaways](#findings--takeaways)
8. [Data locality](#data-locality)
9. [Credits](#credits)

---

## Why llama.cpp (not Ollama)

Ollama is a convenient wrapper around llama.cpp, but for this specific model it has a hard blocker:

> **Currently no Qwen3.5 GGUF works in Ollama** due to separate `mmproj` vision files. Use llama.cpp-compatible backends.
> (Unsloth documentation)

Running llama.cpp directly also gives you:

- **The latest CUDA kernels**, tuned for Blackwell (the RTX 50 series auto-detects as compute capability 12.0)
- **Full control over every tuning flag** (flash attention, quantized KV cache, layer offload, chat template)
- An **OpenAI-compatible `llama-server`** endpoint, so any client (chat UIs, coding agents, harnesses) can connect

---

## Hardware requirements

| Component | Minimum | This setup |
|---|---|---|
| GPU | Any NVIDIA with ≥ 8 GB VRAM | **RTX 5070, 12 GB GDDR7** |
| Compute capability | Blackwell (12.0) for the tuned path | 12.0, driver 591.86 |
| RAM | Enough to host the model file | Not the bottleneck when fully offloaded |
| Disk | ~7 GB free for the model + runtime | All data confined to this project folder |

A dense 9B model at 4-bit (~5.5 GiB) fits comfortably in 12 GB with room for KV cache and context. On smaller cards (8 GB), drop the context length or use a lighter quant.

---

## Model & quantization choice

| Quant | File size | Quality | Fits 12 GB? |
|---|---|---|---|
| Q8_0 | ~9.5 GB | Best (near-lossless) | Only if just the weights fit, leaves no VRAM headroom |
| Q6_K | ~7.5 GB | Very good | Yes |
| **UD-Q4_K_XL** | **~5.5 GB** | ≈ Q8, within ~1-3% | **Yes, with lots of headroom** |
| Q4_K_M | ~5.3 GB | Good (slightly lower) | Yes |

**Why UD-Q4_K_XL:** Unsloth's "UD" (dynamic) quants **upcast the important layers to 8-bit or 16-bit** while keeping the unimportant ones at 4-bit. The result is close-to-Q8 quality at roughly half the size. On a 12 GB card, this is the sweet spot: it leaves room for a large context window and a quantized KV cache.

**Why not Q8_0 on this GPU:** A 9B at Q8 is ~9.5 GB of weights alone. After Windows/desktop usage, a 12 GB card has almost no headroom left, so llama.cpp would spill layers back to the CPU, tanking throughput. The quality you lose jumping from Q8 to dynamic-Q4 is generally imperceptible for chat and coding.

---

## Setup

### 1. Get llama.cpp

Either grab a release binary (recommended, no build needed), or build from source:

```bash
# Release binary (Windows CUDA 13.3)
# Download the "win-cuda-13.3" zip from https://github.com/ggml-org/llama.cpp/releases
#   e.g. llama-b9505-bin-win-cuda-13.3-x64.zip  (+ the companion cudart-*.zip)

# Or build from source
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j --target llama-server llama-cli llama-bench
```

### 2. Download the model

```bash
# Qwen3.5-9B, Unsloth dynamic 4-bit
curl -L -o models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf"
```

### 3. Run the server

```bash
# Linux / macOS
./llama.cpp/llama-server \
  -m models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -ngl 99 --flash-attn on -ctk q8_0 -ctv q8_0 -c 8192 -t 8 \
  --jinja --host 127.0.0.1 --port 8080 --alias "Qwen3.5-9B"
```

```powershell
# Windows (this repo ships a tuned launcher)
.\run-server.ps1
```

Then open <http://127.0.0.1:8080>, or hit the OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1`.

---

## Performance tuning

These are the flags that matter most on a Blackwell GPU, and why:

| Flag | Purpose | Why on Blackwell |
|---|---|---|
| `-ngl 99` | Offload every layer to the GPU | Avoids the CPU bridge, the #1 throughput killer |
| `--flash-attn on` | Fast attention | Blackwell's Tensor Cores get a big win here |
| `-ctk q8_0 -ctv q8_0` | Quantized KV cache | Same VRAM holds more context; negligible quality cost |
| `-c 8192` | Context length | Tune to your use; more context = more VRAM |
| `-t 8` | Thread count | Too many threads actually *slows* GPU inference |
| `--jinja` | Correct chat template | Required for proper Qwen3.5 formatting and tool calling |
| `--alias` | Server model id | What clients see in `/v1/models` |

**Disabling thinking mode** (Qwen3.5 supports reasoning) gives instant answers and simpler, more predictable output. Set `"enable_thinking":false` via the chat-template kwargs to turn it off; `true` re-enables it.

---

## Performance results

Measured with `llama-bench` using the exact same model and flags as the launcher:

| Metric | Result |
|---|---|
| Prompt processing (pp512) | **3,545.85 ± 537 t/s** |
| Generation (tg256) | **88.21 ± 0.19 t/s** |

**Context:** the RTX 5070 has ~672 GB/s of memory bandwidth. A 9B Q4 model is ~5.5 GiB, so the practical decode ceiling is roughly `672 / 5.5 ≈ 112 t/s`. At 88 t/s you're at **~80% of that ceiling**, which is about as good as a same-sized dense model gets on a 12 GB card.

---

## Findings & takeaways

These were verified directly on the RTX 5070:

- **Bandwidth, not compute, is the limit for decode.** Once a model is fully on the GPU, tokens/s scales with memory bandwidth and *inverse* to model size. There is no flag that changes that.
- **Q4 and Q8 are closer than their names suggest.** Community benchmarks show dynamic 4-bit is within error bars of Q8 on many tasks, at half the size.
- **Speculative decoding is not a free win.** A small draft model was set up (Qwen3.5-2B) and tested. It did *not* improve throughput here: at 80% of the bandwidth ceiling, the overhead and imperfect acceptance ate the theoretical gain, and it costs extra VRAM. It's worth trying on bigger, compute-bound models or when acceptance is high, but it was disabled here.
- **Thinking mode changes quality, not speed.** Disabling it shortens time-to-answer dramatically (no `Reasoning...` preamble) without changing the raw `t/s` number.
- **Ollama is not an option for Qwen3.5 right now.** The vision `mmproj` split breaks it; llama.cpp works fine.

---

## Data locality

This instance is configured to keep **all** artifacts (binaries, model, caches, temp, config) inside this project folder and to never write to the system drive. The launcher (on Windows) redirects `TEMP`/`TMP` to a local `tmp\` directory to enforce this. If you clone this for your own use, keep the same discipline: put the GGUF on the same fast disk as your working folder.

---

## Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggml-org
- [Qwen3.5 GGUF quantizations](https://huggingface.co/unsloth) by Unsloth
- Qwen3.5 by Alibaba
