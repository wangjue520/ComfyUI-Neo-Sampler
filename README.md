# ComfyUI-Neo-Sampler

[English](#english) | [中文](#中文)

---

<a id="english"></a>
## English

Brings the sampling pipeline of **SD WebUI Forge Neo (neo branch, commit 41359cd, 2026-09-18)** into ComfyUI, so the same parameters give you the same results as Neo.

### Installation

Put the whole `ComfyUI-Neo-Sampler` folder into `ComfyUI/custom_nodes/`, then restart ComfyUI.

Dependencies (`lark`, `torchsde`, `scipy`) are installed automatically into the Python that runs ComfyUI on first load (works with the embedded Python of portable packages too). If automatic installation fails, the console prints the command you need to run manually.

### The three nodes

| Node | Where to place it | What it does |
|---|---|---|
| **Neo CLIP Converter** | After CLIP loader / LoRA, before any text encode node | Makes downstream text encode nodes follow Neo's encoding rules |
| **Neo Empty Latent** | Replaces "Empty Latent" | Generates initial noise with Neo's ImageRNG and saves the RNG state |
| **Neo KSampler** | Replaces "KSampler" | Neo's samplers, schedulers and full sampling pipeline |

Wiring:

```
UNET/Checkpoint ─→ (LoRA) ─→ Neo KSampler.model
CLIP ─→ (LoRA) ─→ Neo CLIP Converter ─→ CLIPTextEncode(positive) ─→ Neo KSampler.positive
                                      └→ CLIPTextEncode(negative) ─→ Neo KSampler.negative
Neo Empty Latent ───────────────────────────────────────────────→ Neo KSampler.latent_image
```

Hires.fix wiring: first Neo KSampler → Latent upscale → second Neo KSampler (set denoise to the redraw strength and select "Hires.fix" mode). The second sampler automatically reuses the seed from the first pass and regenerates noise at the new size, exactly like Neo.

img2img wiring: VAE Encode → the `latent` input of Neo Empty Latent → Neo KSampler.

### Prompt LoRA (`<lora:name:weight>`)

The Neo KSampler's `prompt_lora` option is on by default and loads LoRAs from the positive prompt following Neo's rules:

- Supports `<lora:name:weight>`, `<lora:name:TE weight:UNet weight>`, and named arguments like `te=` / `unet=`. Only the positive prompt is read; LoRAs are stacked in writing order and applied to both the model and the text encoder — the negative prompt is encoded with the LoRA-patched text encoder too.
- Lookup rules: first matched by filename (without extension, subfolders included), then by the `ss_output_name` alias inside the file. Missing LoRAs are skipped with a console error, same as Neo.
- Internally it calls the exact same functions as the "LoRA Loader" node, so both approaches are bit-identical (tested).

Note: if you use a LoRA Loader node, turn `prompt_lora` off, otherwise the same LoRA is applied twice.

### Mapping to Neo settings

- **Neo Empty Latent**: `rng_source` corresponds to "RNG"; `eta_noise_seed_delta` corresponds to ENSD; `subseed` and `subseed_strength` correspond to the variation seed. ⚠ Neo's factory default is **CPU** — keep it consistent with your Neo settings.
- **Neo KSampler**: `shift` corresponds to the Shift slider; use 0 for the model's built-in value (3.0 for Anima). The optional parameters correspond to the same-named items in Neo's "Settings → Sampler parameters", with identical defaults.
- **Neo CLIP Converter**: `emphasis` corresponds to "Emphasis mode", `clip_skip` to Clip Skip, and `anima_te_precision` to the Anima text encoder compute precision (Neo defaults to fp16).

### What is replicated

- 21 samplers and 17 schedulers, with lists and ordering identical to Neo's dropdowns. Sampler code is taken directly from Neo's source.
- Neo's ImageRNG, including GPU/CPU/NV noise sources, ENSD, variation seeds, batch seed rules (seed+i), and how ancestral and SDE samplers draw noise at every step (TorchHijack, GPU Brownian tree).
- The complete CFGDenoiser logic: per-step prompt scheduling, `AND` combination weights (including edit_strength), ignoring the negative prompt at CFG=1, Skip Early CFG, and NGMS.
- Prompt syntax: `[a:b:steps]`, `[a|b]`, `AND`, `BREAK`, `(x:1.2)`, `[x]`, and stripping `<lora:…>` from the positive prompt. Second-order samplers compute schedules with doubled step counts; the hires pass uses Neo's offset rules.
- Text encoding:
  - Anima: chunked tokenization, weights multiplied directly onto T5 positions without normalization, zero-padding to 512, attention mask rules — all identical to Neo.
  - SD1.5 / SDXL: 75-token chunking, comma backtracking, BREAK, Original / No norm emphasis, clip skip, SDXL L/G concatenation and pooled output — all identical to Neo.
- Both img2img step-count rules (Hires.fix exact steps, or steps × denoise), plus extra noise.

### Verification results

Neo reference code and this plugin were compared item by item on CPU:

- Sampling pipeline: **bit-identical across 594 combinations**. Combinations cover 21 samplers × multiple schedulers × txt2img/hires/img2img × flow and eps models × ENSD × batch 2, with schedule syntax and AND prompts included.
- 17 schedulers: bit-identical sigmas. The FlowMatch scheduler is also bit-identical with the diffusers source.
- RNG: CPU and NV modes are bit-identical. GPU mode runs the same code path.
- Anima, SD1.5, SDXL tokenization and weight logic: fully identical.
- CLIP encoding: compared with the same weights, maximum error 2.5e-5, caused by different attention operator implementations — floating-point noise.
- Text-to-image, hires.fix and various samplers were run end-to-end in ComfyUI with a tiny Anima model, with reproducible results.

### Known remaining differences

- Different GPU operators, PyTorch versions and attention implementations introduce tiny floating-point differences that cannot be eliminated.
- Anima's LLM adapter runs at the diffusion model's precision in ComfyUI (usually bf16), while Neo uses fp16.
- Not supported yet: Textual Inversion embeddings, inpaint noise masks (noise_mask), Refiner. The text encoder currently supports only Anima, SD1.5 and SDXL — other models' CLIPs fall back to ComfyUI's native encoding with a console notice.
- Sampling still runs through ComfyUI's model execution, so model patches like LoRA, ControlNet and RescaleCFG all take effect. However, nodes such as Impact face detailer call ComfyUI's own sampler internally and do not go through Neo logic.
- If a text encode node does not go through `clip.tokenize` / `clip.encode_from_tokens`, the raw text cannot be carried along; the Neo KSampler then treats it as ordinary conditioning and schedule syntax and AND stop working.

### License

This plugin contains code ported from Forge Neo and is therefore licensed under AGPL-3.0 (see LICENSE).

---

<a id="中文"></a>
## 中文

把 **SD WebUI Forge Neo（neo 分支，2026-09-18 提交 41359cd）** 的采样流程搬进 ComfyUI，让同样的参数得到和 Neo 一致的结果。

### 安装

把整个 `ComfyUI-Neo-Sampler` 文件夹放进 `ComfyUI/custom_nodes/`，然后重启 ComfyUI。

依赖（`lark`、`torchsde`、`scipy`）会在首次加载时自动安装到运行 ComfyUI 的那个 Python 里（整合包的 `python_embeded` 也可以）。如果自动安装失败，控制台会打印出需要手动执行的命令。

### 三个节点

| 节点 | 放在哪里 | 作用 |
|---|---|---|
| **Neo CLIP 转换器** | CLIP 加载器 / LoRA 之后，文本编码节点之前 | 让下游任何文本编码节点按 Neo 的规则编码 |
| **Neo 空Latent** | 代替「空Latent」 | 按 Neo 的 ImageRNG 生成初始噪声，并保存随机数发生器状态 |
| **Neo K采样器** | 代替「K采样器」 | Neo 的采样器、调度器和完整采样流程 |

连法：

```
UNET/Checkpoint ─→ (LoRA) ─→ Neo K采样器.model
CLIP ─→ (LoRA) ─→ Neo CLIP 转换器 ─→ CLIPTextEncode(正) ─→ Neo K采样器.positive
                                    └→ CLIPTextEncode(负) ─→ Neo K采样器.negative
Neo 空Latent ─────────────────────────────────────────→ Neo K采样器.latent_image
```

高清修复的连法：第一个 Neo K采样器 → Latent 放大 → 第二个 Neo K采样器（denoise 设为重绘幅度，模式选「Hires.fix」）。第二个采样器会自动沿用第一阶段的种子，并在新尺寸上重新生成噪声，这和 Neo 的做法一致。

图生图的连法：VAE 编码 → Neo 空Latent 的 `latent` 输入 → Neo K采样器。

### 提示词 LoRA（`<lora:名称:权重>`）

Neo K采样器的 `prompt_lora` 默认开启，会按 Neo 的规则加载正面提示词里的 LoRA：

- 语法支持 `<lora:名称:权重>`、`<lora:名称:TE权重:UNet权重>` 以及 `te=` / `unet=` 这种具名写法。只读取正面提示词，按书写顺序叠加，同时作用于模型和文本编码器，负面提示词也会用加了 LoRA 的文本编码器来编码。
- 查找规则：先按文件名（不含扩展名，子文件夹里的也算）匹配，找不到再按文件内的 `ss_output_name` 别名匹配。找不到的 LoRA 会被跳过，并在控制台报错，这些都和 Neo 一样。
- 内部调用的函数和「LoRA 加载器」节点完全相同，所以两种方式的结果逐比特一致（已测试）。

注意：用了 LoRA 加载器就要把 `prompt_lora` 关掉，否则同一个 LoRA 会被叠加两次。

### 和 Neo 设置的对应关系

- **Neo 空Latent**：`rng_source` 对应「随机数生成器」；`eta_noise_seed_delta` 对应 ENSD；`subseed` 和 `subseed_strength` 对应变异种子。⚠ Neo 的出厂默认是 **CPU**，请和你 Neo 里的设置保持一致。
- **Neo K采样器**：`shift` 对应 Shift 滑条，填 0 表示用模型自带值（Anima 为 3.0）。可选参数对应 Neo「设置 → 采样器参数」里的同名项，默认值与 Neo 相同。
- **Neo CLIP 转换器**：`emphasis` 对应「强调模式」，`clip_skip` 对应 Clip Skip，`anima_te_precision` 对应 Anima 文本编码器的计算精度（Neo 默认 fp16）。

### 复刻了什么

- 21 个采样器、17 个调度器，列表和顺序与 Neo 下拉框完全一致。采样器代码直接取自 Neo 的源码。
- Neo 的 ImageRNG，包括 GPU/CPU/NV 三种噪声源、ENSD、变异种子、批量种子规则（seed+i），以及祖先采样和 SDE 每一步噪声的取法（TorchHijack、GPU 布朗树）。
- CFGDenoiser 的完整逻辑：按步切换提示词调度、`AND` 组合权重（包括 edit_strength）、CFG=1 时忽略负面、Skip Early CFG、NGMS。
- 提示词语法：`[a:b:步数]`、`[a|b]`、`AND`、`BREAK`、`(x:1.2)`、`[x]`，以及正面提示词中的 `<lora:…>` 会被剥除。二阶采样器按双倍步数计算调度，高清修复阶段使用 Neo 的偏移规则。
- 文本编码：
  - Anima：分段分词、权重直接乘在 T5 位置上且不做归一化、补零到 512、注意力掩码规则，都与 Neo 相同。
  - SD1.5 / SDXL：75 token 分块、逗号回溯、BREAK、Original / No norm 强调、clip skip、SDXL 的 L/G 拼接与 pooled 输出，都与 Neo 相同。
- 图生图步数的两种规则（Hires.fix 精确步数，或 步数×重绘幅度），以及额外噪声。

### 验证结果

测试在 CPU 上把 Neo 原版代码和本插件放在一起逐项对比：

- 采样流程：594 个组合**逐比特一致**。组合覆盖 21 个采样器 × 多种调度器 × 文生图/高清修复/图生图 × flow 与 eps 模型 × ENSD × 批量 2，并带有调度语法和 AND 提示词。
- 17 个调度器：sigma 逐比特一致。FlowMatch 调度器也与 diffusers 源码结果逐比特一致。
- RNG：CPU 与 NV 模式逐比特一致。GPU 模式走的是同一段代码。
- Anima、SD1.5、SDXL 的分词与权重逻辑：完全一致。
- CLIP 编码：用同一套权重对比，最大误差 2.5e-5，来自注意力算子实现的不同，属于浮点误差。
- 在 ComfyUI 中用微型 Anima 模型跑通了文生图、高清修复和多种采样器，结果可以复现。

### 已知的剩余差异

- GPU 算子、PyTorch 版本和注意力实现不同，会带来极小的浮点差，这一点无法消除。
- Anima 的 LLM adapter 在 ComfyUI 里跟随扩散模型精度运行（通常是 bf16），而 Neo 是 fp16。
- 暂不支持：Textual Inversion embedding、局部重绘遮罩（noise_mask）、Refiner。文本编码器目前只支持 Anima、SD1.5 和 SDXL，其他模型的 CLIP 接上转换器后会按 ComfyUI 原逻辑编码，并在控制台提示。
- 采样器照常用 ComfyUI 跑模型，所以 LoRA、ControlNet、RescaleCFG 这类模型补丁都会生效。但 Impact 面部修复等节点内部调用的是 ComfyUI 自己的采样器，不会走 Neo 逻辑。
- 如果某个文本编码节点不经过 `clip.tokenize` / `clip.encode_from_tokens`，就无法携带原文，Neo K采样器只能按普通条件处理，调度语法和 AND 会失效。

### 许可

插件包含从 Forge Neo 移植的代码，因此遵循 AGPL-3.0（见 LICENSE）。
