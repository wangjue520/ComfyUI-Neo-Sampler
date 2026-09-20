# Ports of Forge Neo's text processing engines, running on ComfyUI's already-loaded text encoder weights.
#   * backend/text_processing/anima_engine.py   -> AnimaEngine
#   * backend/text_processing/classic_engine.py -> ClassicEngine (SD1.5 / SDXL / SDXL refiner)
#   * backend/text_processing/emphasis.py       -> apply_emphasis
#   * backend/diffusion_engine/sdxl.py (get_learned_conditioning) -> SDXLConditioner
# Token logic, chunking, emphasis and attention-mask rules are copied from Neo. Only the "call the
# transformer" lines are adapted to ComfyUI's module API.
# Not ported: textual inversion embeddings (Neo matches bare embedding file names inside the prompt).
import logging

import torch

from .parsing import parse_prompt_attention

logger = logging.getLogger("NeoSampler")

EMPHASIS_MODES = ["Original", "No norm", "Ignore", "None"]


# region helpers


def _prepare_clip(clip):
    """Load the text encoder the same way comfy.sd.CLIP.encode_from_tokens does."""
    import comfy.model_management as mm

    clip.cond_stage_model.reset_clip_options()
    clip.load_model()
    device = clip.patcher.load_device
    try:
        clip.cond_stage_model.set_clip_options({"execution_device": device})
    except Exception:
        pass
    return device, mm


def _embed(embedding_module, tokens, dtype):
    try:
        return embedding_module(tokens, out_dtype=dtype)
    except TypeError:
        return embedding_module(tokens).to(dtype)


def apply_emphasis(name, z, multipliers):
    """backend/text_processing/emphasis.py"""
    if name == "Original":
        original_mean = z.mean()
        z = z * multipliers.reshape(multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z = z * (original_mean / new_mean)
    elif name == "No norm":
        z = z * multipliers.reshape(multipliers.shape + (1,)).expand(z.shape)
    # "Ignore" and "None": after_transformers() does nothing
    return z


# endregion

# region Anima


class AnimaEngine:
    """backend/text_processing/anima_engine.py"""

    kind = "anima"

    def __init__(self, clip, emphasis="Original", te_dtype=torch.float16):
        self.clip = clip
        self.emphasis = emphasis
        self.te_dtype = te_dtype
        self.qwen_tokenizer = clip.tokenizer.qwen3_06b.tokenizer
        self.t5_tokenizer = clip.tokenizer.t5xxl.tokenizer
        self.id_pad = 151643
        self.id_end = 1

    def tokenize(self, texts):
        return (
            self.qwen_tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"],
            self.t5_tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"],
        )

    def tokenize_line(self, line):
        parsed = parse_prompt_attention(line, self.emphasis)
        qwen_tokenized, t5_tokenized = self.tokenize([text for text, _ in parsed])

        qwen_tokens, t5_tokens, t5_multipliers = [], [], []

        for tokens in qwen_tokenized:
            qwen_tokens.extend(tokens)

        for tokens, (text, weight) in zip(t5_tokenized, parsed):
            for token in tokens:
                t5_tokens.append(token)
                t5_multipliers.append(weight)

        # next_chunk()
        if not qwen_tokens:
            qwen_tokens.append(self.id_pad)
        t5_tokens.append(self.id_end)
        t5_multipliers.append(1.0)

        return qwen_tokens, t5_tokens, t5_multipliers

    @torch.no_grad()
    def encode_texts(self, texts):
        """Returns a list of (cross_attn_before_adapter, extra_dict) - the adapter/weights/pad-to-512 step
        is performed by ComfyUI's Anima model (identical math to Neo's anima_preprocess)."""
        device, mm = _prepare_clip(self.clip)
        transformer = self.clip.cond_stage_model.qwen3_06b.transformer
        results = []
        cache = {}
        with mm.cuda_device_context(device) if hasattr(mm, "cuda_device_context") else _nullctx():
            for line in texts:
                if line in cache:
                    results.append(cache[line])
                    continue
                qwen_tokens, t5_tokens, t5_multipliers = self.tokenize_line(line)

                # process_embeds
                attention_mask, eos = [], False
                for token in qwen_tokens:
                    attention_mask.append(0 if eos else 1)
                    if not eos and token == self.id_pad:
                        eos = True
                tok = torch.tensor([qwen_tokens], device=device, dtype=torch.long)
                embeds = _embed(transformer.get_input_embeddings(), tok, self.te_dtype)
                mask = torch.tensor([attention_mask], device=device, dtype=torch.long)
                kwargs = dict(attention_mask=mask, embeds=embeds, num_tokens=[sum(attention_mask)], dtype=self.te_dtype)
                try:
                    out = transformer(None, embeds_info=[], **kwargs)
                except TypeError:
                    out = transformer(None, **kwargs)
                z = out[0].float().to(mm.intermediate_device())

                extra = {
                    "t5xxl_ids": torch.tensor(t5_tokens, dtype=torch.int),
                    "t5xxl_weights": torch.tensor(t5_multipliers),
                }
                cache[line] = (z, extra)
                results.append(cache[line])
        return results


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# endregion

# region Classic (CLIP)


class _Chunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class ClassicEngine:
    """backend/text_processing/classic_engine.py (textual inversion not supported)"""

    def __init__(self, sdclip, hf_tokenizer, emphasis, chunk_length=75, text_projection=False, minimal_clip_skip=1, clip_skip=1, return_pooled=False, final_layer_norm=True):
        self.sdclip = sdclip  # comfy.sd1_clip.SDClipModel
        self.tokenizer = hf_tokenizer
        self.emphasis = emphasis
        self.chunk_length = chunk_length
        self.text_projection = text_projection
        self.minimal_clip_skip = minimal_clip_skip
        self.clip_skip = clip_skip
        self.return_pooled = return_pooled
        self.final_layer_norm = final_layer_norm

        special = getattr(sdclip, "special_tokens", {})
        self.id_start = special.get("start", 49406)
        self.id_end = special.get("end", 49407)
        self.id_pad = special.get("pad", 49407)  # Neo: tokenizer.pad_token_id (clip_l: 49407, SDXL clip_g: 0 "!")

        vocab = self.tokenizer.get_vocab()
        self.comma_token = vocab[",</w>"]

    def empty_chunk(self):
        chunk = _Chunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def tokenize(self, texts):
        return self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

    def tokenize_line(self, line):
        parsed = parse_prompt_attention(line, self.emphasis)
        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = _Chunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = _Chunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                comma_padding_backtrack = 20

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                elif comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count

    def encode_with_transformers(self, tokens, device):
        transformer = self.sdclip.transformer  # comfy.clip_model.CLIPTextModel
        tokens = tokens.to(device)
        layer_id = -max(self.clip_skip, self.minimal_clip_skip)
        # Neo: hidden_states[layer_id] (+ final_layer_norm), computed with float32 embeddings
        out = transformer(tokens, None, intermediate_output=layer_id, final_layer_norm_intermediate=self.final_layer_norm, dtype=torch.float32)
        z = out[1]
        pooled = None
        if self.return_pooled:
            pooled = out[2] if self.text_projection else out[3]
        return z, pooled

    def process_tokens(self, remade_batch_tokens, batch_multipliers, device):
        tokens = torch.asarray(remade_batch_tokens)

        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1 : tokens.shape[1]] = self.id_pad

        z, pooled = self.encode_with_transformers(tokens, device)
        multipliers = torch.asarray(batch_multipliers).to(z)
        z = apply_emphasis(self.emphasis, z, multipliers)
        return z, pooled

    def __call__(self, line, device):
        chunks, _ = self.tokenize_line(line)
        zs = []
        first_pooled = None
        for i, chunk in enumerate(chunks):
            z, pooled = self.process_tokens([chunk.tokens], [chunk.multipliers], device)
            if i == 0:
                first_pooled = pooled
            zs.append(z)
        z = torch.hstack(zs)
        return (z, first_pooled) if self.return_pooled else z


class SD15Engine:
    kind = "sd15"

    def __init__(self, clip, emphasis="Original", clip_skip=2):
        self.clip = clip
        csm = clip.cond_stage_model
        self.engine = ClassicEngine(csm.clip_l, clip.tokenizer.clip_l.tokenizer, emphasis, text_projection=False, minimal_clip_skip=1, clip_skip=clip_skip, return_pooled=False, final_layer_norm=True)

    @torch.no_grad()
    def encode_texts(self, texts):
        device, mm = _prepare_clip(self.clip)
        res = []
        for line in texts:
            z = self.engine(line, device)
            res.append((z.float().to(mm.intermediate_device()), {"pooled_output": None}))
        return res


class SDXLEngine:
    """backend/diffusion_engine/sdxl.py StableDiffusionXL.get_learned_conditioning
    (the width/height/crop vector is built by ComfyUI's SDXL model from the same numbers)."""

    kind = "sdxl"

    def __init__(self, clip, emphasis="Original", clip_skip=2, refiner=False):
        self.clip = clip
        self.refiner = refiner
        csm = clip.cond_stage_model
        tok = clip.tokenizer
        self.engine_l = None
        if not refiner:
            self.engine_l = ClassicEngine(csm.clip_l, tok.clip_l.tokenizer, emphasis, text_projection=False, minimal_clip_skip=2, clip_skip=clip_skip, return_pooled=False, final_layer_norm=False)
        self.engine_g = ClassicEngine(csm.clip_g, tok.clip_g.tokenizer, emphasis, text_projection=True, minimal_clip_skip=2, clip_skip=clip_skip, return_pooled=True, final_layer_norm=False)

    @torch.no_grad()
    def encode_texts(self, texts):
        device, mm = _prepare_clip(self.clip)
        res = []
        for line in texts:
            cond_g, clip_pooled = self.engine_g(line, device)
            if self.engine_l is not None:
                cond_l = self.engine_l(line, device)
                max_len = max(cond_l.shape[1], cond_g.shape[1])
                cond_l = torch.cat([cond_l, cond_l.new_zeros(cond_l.size(0), max_len - cond_l.shape[1], cond_l.size(2))], dim=1)
                cond_g = torch.cat([cond_g, cond_g.new_zeros(cond_g.size(0), max_len - cond_g.shape[1], cond_g.size(2))], dim=1)
                crossattn = torch.cat([cond_l, cond_g], dim=2)
            else:
                crossattn = cond_g
            inter = mm.intermediate_device()
            res.append((crossattn.float().to(inter), {"pooled_output": clip_pooled.float().to(inter)}))
        return res


# endregion


def detect_engine_kind(clip):
    """Returns 'anima' / 'sdxl' / 'sdxl_refiner' / 'sd15' / None"""
    try:
        import comfy.sd1_clip as sd1_clip
        import comfy.sdxl_clip as sdxl_clip
    except Exception:
        return None
    csm = clip.cond_stage_model
    tok = clip.tokenizer
    mod = type(csm).__module__
    if hasattr(csm, "qwen3_06b") and hasattr(tok, "qwen3_06b") and hasattr(tok, "t5xxl") and "anima" in mod:
        return "anima"
    if isinstance(csm, sdxl_clip.SDXLClipModel) and type(tok) is sdxl_clip.SDXLTokenizer:
        return "sdxl"
    if isinstance(csm, sdxl_clip.SDXLRefinerClipModel):
        return "sdxl_refiner"
    if type(csm) is sd1_clip.SD1ClipModel and getattr(csm, "clip", None) == "clip_l" and type(tok) is sd1_clip.SD1Tokenizer:
        return "sd15"
    return None


def build_engine(clip, kind, settings):
    if kind == "anima":
        return AnimaEngine(clip, settings["emphasis"], settings["anima_te_dtype"])
    if kind == "sdxl":
        return SDXLEngine(clip, settings["emphasis"], settings["clip_skip"], refiner=False)
    if kind == "sdxl_refiner":
        return SDXLEngine(clip, settings["emphasis"], settings["clip_skip"], refiner=True)
    if kind == "sd15":
        return SD15Engine(clip, settings["emphasis"], settings["clip_skip"])
    return None
