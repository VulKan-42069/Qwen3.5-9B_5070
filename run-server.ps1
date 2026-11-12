# Qwen3.5-9B tuned llama-server launcher.
# Portable: $PSScriptRoot is this script's own folder, so everything stays inside it wherever you
# place it, and nothing is written to the system drive.
$root = $PSScriptRoot
$env:TEMP = "$root\tmp"; $env:TMP = $env:TEMP   # keep any temp use local to the project

# Thinking: ON, at a "medium" budget. See $thinkBudget below.
#
# Qwen3.5 has NO reasoning-effort tiers. Verified against the model's own chat template: the only
# knob it exposes is the boolean `enable_thinking`, and llama.cpp's `--reasoning` accepts only
# on|off|auto. So "medium" can only mean one thing on this stack: a thinking TOKEN BUDGET.
#
# b9505 DEPRECATED `--chat-template-kwargs '{"enable_thinking":false}'`: it is still accepted and
# only warns, but it is silently ignored, which leaves thinking ON. Measured cost of missing that:
# a 3-word reply burned 193 reasoning tokens instead of 6. `--reasoning off` is the supported
# replacement, and `--reasoning-budget` is the graduated one.

$llama = "$root\llama.cpp\llama-server.exe"
$model = "$root\models\Qwen3.5-9B-UD-Q4_K_XL.gguf"

# Thinking budget: the closest thing this model has to a low/medium/high dial.
#   0     no thinking (equivalent to --reasoning off)
#   512   light
#   2048  MEDIUM  <- current
#   8192  heavy
#   -1    unrestricted
# Measured: a plain math question reasons ~1k tokens unrestricted and finishes naturally, so 2048
# is a CEILING for runaway loops rather than a throttle on normal work. Verified to actually fire:
# at budget 100 the reasoning stops mid-sentence, $thinkNudge is injected, and the final answer is
# still produced correctly.
$thinkBudget = 2048
$thinkNudge  = 'Reasoning budget reached. Give your final answer now.'

# Speculative decoding (a 2B draft model) was tried but did NOT improve t/s here: a 9B Q4 on 12 GB is
# already near the memory-bandwidth ceiling, so it isn't worth the extra VRAM. To experiment again,
# uncomment the next line and add "-md $draft --spec-draft-n-max 16" to the run command below:
# $draft = "$root\models\draft\Qwen3.5-2B-Q4_0.gguf"

# Tuned for max context on the RTX 5070 (12 GB Blackwell):
#   -ngl 99            offload every layer to the GPU
#   --flash-attn on    fast attention (big win on Blackwell)
#   -ctk q4_0 -ctv q4_0  quantized KV cache; q4_0 (not q8_0) is what makes 256K fit
#   -c 262144          full native context (qwen35.context_length = 262144)
#   -t 8               thread count (too many threads actually slows GPU inference)
#   --jinja            correct Qwen3.5 chat template (also enables tool calling)
#   --reasoning on     thinking enabled, capped by $thinkBudget (see note above)
#   --alias            model id exposed by /v1/models
#
# Why 256K is affordable here: Qwen3.5 is a HYBRID model. `full_attention_interval = 4` means only
# 8 of the 32 layers keep a context-length KV cache; the other 24 use linear attention with a fixed
# recurrent state. A conventional dense 32-layer model would need ~8x this KV footprint.
#
# Measured on this card (12226 MiB, ~1.1 GiB already taken by the desktop):
#   -ctk q8_0 at -c 262144 -> 11786 MiB used,  158 MiB free  (too tight: one browser tab kills it)
#   -ctk q4_0 at -c 262144 ->  9882 MiB used, 2062 MiB free  (the default below)
# Decode measured ~84 t/s at 256K context (vs ~88 t/s at 8K), so the extra context is nearly free.

& $llama -m $model -ngl 99 --flash-attn on -ctk q4_0 -ctv q4_0 -c 262144 -t 8 `
  --jinja --reasoning on --reasoning-budget $thinkBudget --reasoning-budget-message $thinkNudge `
  --host 127.0.0.1 --port 8080 --alias "Qwen3.5-9B"
