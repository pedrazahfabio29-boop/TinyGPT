import os

# For small matrix-heavy CPU inference, too many BLAS threads can make a
# low-CPU cloud instance slower because of thread-management overhead.
# Override with TINY_GPT_THREADS if needed (for example, 2 or 4 on a stronger machine).
_TINY_GPT_THREADS = os.environ.get("TINY_GPT_THREADS", "1")
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, _TINY_GPT_THREADS)

import json
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from flask import Flask, jsonify, request


# ============================================================
# Configuration
# ============================================================

MODEL_FILE = os.environ.get("MODEL_FILE", "tiny_gpt_pc.npz")

MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "8000"))
DEFAULT_MAX_NEW_TOKENS = int(
    os.environ.get("DEFAULT_MAX_NEW_TOKENS", "180")
)

# Keep this at 1 on the Render free plan. A single CPU is available on the
# plan, and the model already serializes generation requests.
MODEL_LOCK = threading.Lock()


# ============================================================
# Tokenizer
# ============================================================

class HybridTokenizer:
    def __init__(self, tokens: Sequence[str]):
        self.tokens = list(tokens)
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = {i: token for i, token in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)

        self.special_tokens = {
            "<START>",
            "<USER>",
            "<ASSISTANT>",
            "<END>",
            "<UNK>",
        }

        self.pattern = re.compile(
            r"<START>|<USER>|<ASSISTANT>|<END>|<UNK>|"
            r"[A-Za-z]+(?:'[A-Za-z]+)?|"
            r"[0-9]+(?:\.[0-9]+)?|"
            r"\s+|"
            r"[^A-Za-z0-9\s]"
        )

        # Avoid repeated regex fullmatch calls for normal words.
        self.word_pattern = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")

    def encode(self, text: str) -> List[int]:
        pieces = self.pattern.findall(text)
        result: List[int] = []
        append = result.append
        token_to_id = self.token_to_id
        unk_id = token_to_id.get("<UNK>", 0)
        word_pattern = self.word_pattern

        for piece in pieces:
            # Exact token match, including special tokens, whitespace, and punctuation.
            token_id = token_to_id.get(piece)
            if token_id is not None:
                append(token_id)
                continue

            # Case-normalized word match.
            if word_pattern.fullmatch(piece):
                lower_id = token_to_id.get(piece.lower())
                if lower_id is not None:
                    append(lower_id)
                    continue

            # Character fallback.
            for char in piece:
                append(token_to_id.get(char, unk_id))

        return result

    def decode(self, ids: Sequence[int]) -> str:
        id_to_token = self.id_to_token
        specials = self.special_tokens
        parts = []

        for token_id in ids:
            token = id_to_token.get(int(token_id), "")
            if token not in specials:
                parts.append(token)

        return "".join(parts)


# ============================================================
# TinyGPT optimized inference model
# ============================================================

class TinyGPT:
    """
    Optimized inference implementation compatible with the existing TinyGPT
    .npz parameter format.

    Main optimizations:
      1. Fuses Wq/Wk/Wv into one QKV matrix per Transformer layer.
      2. Uses a KV cache during autoregressive generation.
      3. Reuses preallocated KV arrays instead of concatenating every token.
      4. Uses in-place softmax math where practical to reduce allocations.
      5. Uses a contiguous output projection matrix.
      6. Only computes top-p over the top-k candidates when top_k is enabled.
      7. Avoids rebuilding the full context until the fixed-size positional
         context window is actually full.
    """

    def __init__(self, data: np.lib.npyio.NpzFile):
        meta_raw: Any = data["meta_json"]

        if isinstance(meta_raw, np.ndarray):
            meta_raw = meta_raw.item()

        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")

        if isinstance(meta_raw, str):
            self.meta = json.loads(meta_raw)
        else:
            self.meta = meta_raw

        self.block_size = int(self.meta["block_size"])
        self.embed_size = int(self.meta["embed_size"])
        self.num_heads = int(self.meta["num_heads"])
        self.num_layers = int(self.meta["num_layers"])
        self.vocab_size = int(self.meta["vocab_size"])

        if self.embed_size % self.num_heads != 0:
            raise ValueError(
                "embed_size must be divisible by num_heads."
            )

        self.head_dim = self.embed_size // self.num_heads
        self.attn_scale = 1.0 / np.sqrt(self.head_dim)

        tokens = self.meta["tokens"]
        self.tokenizer = HybridTokenizer(tokens)

        # ----------------------------------------------------
        # Load parameters
        # ----------------------------------------------------

        raw_params: Dict[str, np.ndarray] = {}
        for key in data.files:
            if not key.startswith("param_"):
                continue
            name = key[len("param_"):]
            raw_params[name] = data[key].astype(
                np.float32,
                copy=False,
            )

        self.params = raw_params

        # Embeddings are used on every token, so keep references to them.
        self.token_embedding = np.ascontiguousarray(
            self.params["token_embedding"],
            dtype=np.float32,
        )
        self.position_embedding = np.ascontiguousarray(
            self.params["position_embedding"],
            dtype=np.float32,
        )

        # Weight tying: final output projection is token_embedding.T.
        # Make the transposed matrix contiguous once rather than paying for a
        # strided transpose during every generated token.
        self.output_weight = np.ascontiguousarray(
            self.token_embedding.T,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Special token IDs
        # ----------------------------------------------------

        self.start_id = self.tokenizer.token_to_id.get("<START>")
        self.user_id = self.tokenizer.token_to_id.get("<USER>")
        self.assistant_id = self.tokenizer.token_to_id.get("<ASSISTANT>")
        self.end_id = self.tokenizer.token_to_id.get("<END>")

        # ----------------------------------------------------
        # Causal mask for prompt prefill only.
        # Decode steps do not need a mask because a new token only attends
        # to cached past tokens plus itself.
        # ----------------------------------------------------

        self.causal_mask = np.triu(
            np.ones(
                (self.block_size, self.block_size),
                dtype=bool,
            ),
            k=1,
        )

        # ----------------------------------------------------
        # Pre-build optimized layer records.
        # ----------------------------------------------------

        self.layers: List[Dict[str, np.ndarray]] = []

        for layer_index in range(self.num_layers):
            prefix = f"layer{layer_index}_"

            # Fuse Q/K/V weights once. This replaces three matrix multiplies
            # with one larger matrix multiply in both prefill and decode.
            Wq = np.ascontiguousarray(
                raw_params[prefix + "Wq"],
                dtype=np.float32,
            )
            Wk = np.ascontiguousarray(
                raw_params[prefix + "Wk"],
                dtype=np.float32,
            )
            Wv = np.ascontiguousarray(
                raw_params[prefix + "Wv"],
                dtype=np.float32,
            )

            Wqkv = np.ascontiguousarray(
                np.concatenate((Wq, Wk, Wv), axis=1),
                dtype=np.float32,
            )

            layer = {
                "norm1": np.ascontiguousarray(
                    raw_params[prefix + "norm1"], dtype=np.float32
                ),
                "Wqkv": Wqkv,
                "Wo": np.ascontiguousarray(
                    raw_params[prefix + "Wo"], dtype=np.float32
                ),
                "norm2": np.ascontiguousarray(
                    raw_params[prefix + "norm2"], dtype=np.float32
                ),
                "W1": np.ascontiguousarray(
                    raw_params[prefix + "W1"], dtype=np.float32
                ),
                "b1": np.ascontiguousarray(
                    raw_params[prefix + "b1"], dtype=np.float32
                ),
                "W2": np.ascontiguousarray(
                    raw_params[prefix + "W2"], dtype=np.float32
                ),
                "b2": np.ascontiguousarray(
                    raw_params[prefix + "b2"], dtype=np.float32
                ),
            }

            self.layers.append(layer)

        self.final_norm = np.ascontiguousarray(
            raw_params["final_norm"],
            dtype=np.float32,
        )

        self.position_ids = np.arange(
            self.block_size,
            dtype=np.int64,
        )

        print("TinyGPT optimized model loaded.")
        print("Vocabulary:", self.vocab_size)
        print("Embedding:", self.embed_size)
        print("Layers:", self.num_layers)
        print("Heads:", self.num_heads)
        print("Context:", self.block_size)
        print("Parameters:", len(self.params))
        print("Head dimension:", self.head_dim)
        print("KV cache: enabled")
        print("QKV fusion: enabled")

    # --------------------------------------------------------
    # RMSNorm
    # --------------------------------------------------------

    @staticmethod
    def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean_square = np.mean(
            x * x,
            axis=-1,
            keepdims=True,
        )
        return (x / np.sqrt(mean_square + eps)) * weight

    # --------------------------------------------------------
    # GELU
    # --------------------------------------------------------

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (
            1.0
            + np.tanh(
                np.sqrt(2.0 / np.pi)
                * (x + 0.044715 * x * x * x)
            )
        )

    # --------------------------------------------------------
    # Softmax helper for attention.
    # Reuses the score array to avoid another large allocation.
    # --------------------------------------------------------

    @staticmethod
    def softmax_inplace(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x -= np.max(x, axis=axis, keepdims=True)
        np.exp(x, out=x)
        x /= np.sum(x, axis=axis, keepdims=True) + 1e-9
        return x

    # --------------------------------------------------------
    # Allocate KV cache
    # --------------------------------------------------------

    def create_cache(self) -> Tuple[List[Dict[str, np.ndarray]], int]:
        cache: List[Dict[str, np.ndarray]] = []

        shape = (
            self.num_heads,
            self.block_size,
            self.head_dim,
        )

        for _ in range(self.num_layers):
            cache.append({
                "k": np.empty(shape, dtype=np.float32),
                "v": np.empty(shape, dtype=np.float32),
            })

        return cache, 0

    # --------------------------------------------------------
    # Full Transformer prefill
    # --------------------------------------------------------

    def prefill(
        self,
        token_ids: Sequence[int],
    ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]], int]:
        """Process a prompt once and populate the KV cache."""

        if not token_ids:
            raise ValueError("prefill() received an empty token sequence")

        # Match the original model behavior: only the last block_size tokens
        # participate in inference, and positions restart from zero.
        ids = np.asarray(token_ids[-self.block_size:], dtype=np.int64)
        T = int(ids.shape[0])

        cache, cache_length = self.create_cache()

        x = (
            self.token_embedding[ids]
            + self.position_embedding[self.position_ids[:T]]
        ).astype(np.float32, copy=False)

        for layer_index, layer in enumerate(self.layers):
            h = self.rms_norm(x, layer["norm1"])

            # [T, 3C] -> Q/K/V, each [H, T, D]
            qkv = h @ layer["Wqkv"]
            Q_flat, K_flat, V_flat = np.split(qkv, 3, axis=-1)

            Q = Q_flat.reshape(
                T, self.num_heads, self.head_dim
            ).transpose(1, 0, 2)
            K = K_flat.reshape(
                T, self.num_heads, self.head_dim
            ).transpose(1, 0, 2)
            V = V_flat.reshape(
                T, self.num_heads, self.head_dim
            ).transpose(1, 0, 2)

            # Fill cache for autoregressive decode.
            cache[layer_index]["k"][:, :T, :] = K
            cache[layer_index]["v"][:, :T, :] = V

            scores = (Q @ K.transpose(0, 2, 1)) * self.attn_scale

            # Causal masking.
            mask = self.causal_mask[:T, :T]
            scores = np.where(
                mask[None, :, :],
                -1e9,
                scores,
            )

            weights = self.softmax_inplace(scores, axis=-1)
            attention = weights @ V

            attention = attention.transpose(1, 0, 2).reshape(T, self.embed_size)
            x = x + (attention @ layer["Wo"])

            h = self.rms_norm(x, layer["norm2"])
            h = h @ layer["W1"] + layer["b1"]
            h = self.gelu(h)
            h = h @ layer["W2"] + layer["b2"]
            x = x + h

        x = self.rms_norm(x, self.final_norm)
        logits = x @ self.output_weight

        cache_length = T
        return logits[-1], cache, cache_length

    # --------------------------------------------------------
    # One-token cached decode step
    # --------------------------------------------------------

    def decode_one(
        self,
        token_id: int,
        cache: List[Dict[str, np.ndarray]],
        cache_length: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Process exactly one newly generated token.

        No causal mask is needed: this token can attend to all cached tokens
        and itself, and cannot see anything in the future.
        """

        if cache_length >= self.block_size:
            raise ValueError("KV cache is full; rebuild with prefill().")

        x = (
            self.token_embedding[int(token_id)]
            + self.position_embedding[cache_length]
        ).reshape(1, self.embed_size).astype(np.float32, copy=False)

        for layer_index, layer in enumerate(self.layers):
            h = self.rms_norm(x, layer["norm1"])

            qkv = h @ layer["Wqkv"]
            Q_flat, K_flat, V_flat = np.split(qkv, 3, axis=-1)

            Q = Q_flat.reshape(
                1, self.num_heads, self.head_dim
            ).transpose(1, 0, 2)
            K_new = K_flat.reshape(
                1, self.num_heads, self.head_dim
            ).transpose(1, 0, 2)
            V_new = V_flat.reshape(
                1, self.num_heads, self.head_dim
            ).transpose(1, 0, 2)

            layer_cache = cache[layer_index]
            layer_cache["k"][:, cache_length:cache_length + 1, :] = K_new
            layer_cache["v"][:, cache_length:cache_length + 1, :] = V_new

            K_all = layer_cache["k"][:, :cache_length + 1, :]
            V_all = layer_cache["v"][:, :cache_length + 1, :]

            scores = (Q @ K_all.transpose(0, 2, 1)) * self.attn_scale
            weights = self.softmax_inplace(scores, axis=-1)
            attention = weights @ V_all

            attention = attention.transpose(1, 0, 2).reshape(1, self.embed_size)
            x = x + (attention @ layer["Wo"])

            h = self.rms_norm(x, layer["norm2"])
            h = h @ layer["W1"] + layer["b1"]
            h = self.gelu(h)
            h = h @ layer["W2"] + layer["b2"]
            x = x + h

        x = self.rms_norm(x, self.final_norm)
        logits = x @ self.output_weight

        return logits[0], cache_length + 1

    # --------------------------------------------------------
    # Single-token convenience forward for compatibility/debugging
    # --------------------------------------------------------

    def forward(self, token_ids: Sequence[int]) -> np.ndarray:
        """
        Full forward pass retained for compatibility/debugging.
        Generation uses prefill()+decode_one() instead.
        """
        ids = list(token_ids)
        if not ids:
            raise ValueError("forward() received an empty token sequence")

        logits, _, _ = self.prefill(ids)
        return logits


# ============================================================
# Model loading
# ============================================================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}\n"
        "Make sure tiny_gpt_pc.npz is in the repository."
    )

print(f"Loading TinyGPT model from {MODEL_FILE}...")

model_data = np.load(
    MODEL_FILE,
    allow_pickle=True,
)

model = TinyGPT(model_data)


# ============================================================
# Sampling
# ============================================================

def sample_next_token(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 0.70,
    top_k: int = 30,
    top_p: float = 0.90,
    repetition_penalty: float = 1.05,
    recent_token_ids: Optional[Sequence[int]] = None,
) -> int:
    """
    Fast sampling equivalent in behavior to the original backend:
      repetition penalty -> temperature -> top-k -> softmax -> top-p -> sample

    When top_k is active, top-p is computed only over those top-k candidates,
    which avoids sorting the entire vocabulary.
    """

    logits = np.asarray(logits, dtype=np.float32).copy()

    # Repetition penalty.
    if repetition_penalty > 1.0 and recent_token_ids:
        recent = set(int(x) for x in recent_token_ids)
        vocab_size = logits.shape[0]

        for token_id in recent:
            if 0 <= token_id < vocab_size:
                if logits[token_id] > 0.0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

    temperature = max(float(temperature), 1e-5)
    logits /= temperature

    vocab_size = logits.shape[0]

    # --------------------------------------------------------
    # Top-k first.
    # --------------------------------------------------------

    if 0 < top_k < vocab_size:
        candidate_ids = np.argpartition(
            logits,
            -top_k,
        )[-top_k:]

        candidate_logits = logits[candidate_ids]
    else:
        candidate_ids = np.arange(vocab_size, dtype=np.int64)
        candidate_logits = logits

    # --------------------------------------------------------
    # Stable softmax over candidates only.
    # --------------------------------------------------------

    candidate_logits = candidate_logits - np.max(candidate_logits)
    probabilities = np.exp(candidate_logits)
    total = float(np.sum(probabilities))

    if total <= 0.0 or not np.isfinite(total):
        probabilities = np.full(
            probabilities.shape,
            1.0 / len(probabilities),
            dtype=np.float32,
        )
    else:
        probabilities /= total

    # --------------------------------------------------------
    # Top-p.
    # --------------------------------------------------------

    if top_p < 1.0 and len(probabilities) > 1:
        order = np.argsort(probabilities)[::-1]
        sorted_probs = probabilities[order]
        cumulative = np.cumsum(sorted_probs)

        remove = cumulative > top_p
        remove[0] = False

        probabilities[order[remove]] = 0.0

        total = float(np.sum(probabilities))
        if total > 0.0:
            probabilities /= total

    chosen_index = int(
        rng.choice(
            len(candidate_ids),
            p=probabilities,
        )
    )

    return int(candidate_ids[chosen_index])


# ============================================================
# Generation
# ============================================================

def generate(
    prompt: str,
    temperature: float = 0.70,
    top_k: int = 30,
    top_p: float = 0.90,
    repetition_penalty: float = 1.05,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: Optional[int] = None,
) -> Tuple[str, int, float, bool]:
    """
    Return:
      answer, generated_token_count, generation_time_seconds, kv_cache_used
    """

    tokenizer = model.tokenizer

    full_prompt = (
        "<START>\n"
        "<USER> "
        + prompt
        + "\n"
        "<ASSISTANT> "
    )

    prompt_ids = tokenizer.encode(full_prompt)
    if not prompt_ids:
        return "", 0, 0.0, False

    generated = list(prompt_ids)
    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()

    # One full prompt pass; subsequent tokens use cached K/V.
    logits, cache, cache_length = model.prefill(generated)
    kv_cache_used = True
    generated_count = 0

    for _ in range(max_new_tokens):
        next_token = sample_next_token(
            logits,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            recent_token_ids=generated[-64:],
        )

        generated.append(next_token)
        generated_count += 1

        if model.end_id is not None and next_token == model.end_id:
            break

        # If there is room in the fixed positional context, decode the new
        # token incrementally using the existing cache.
        if cache_length < model.block_size:
            logits, cache_length = model.decode_one(
                next_token,
                cache,
                cache_length,
            )
        else:
            # The original model discards the oldest token and resets the
            # position IDs when context exceeds block_size. To preserve that
            # exact behavior, rebuild the cache from the new sliding window.
            logits, cache, cache_length = model.prefill(
                generated[-model.block_size:]
            )

    elapsed = max(time.perf_counter() - start_time, 1e-9)

    # Decode only the generated answer.
    answer = tokenizer.decode(generated)

    marker = "<ASSISTANT>"
    if marker in answer:
        answer = answer.split(marker, 1)[1]

    if "<USER>" in answer:
        answer = answer.split("<USER>", 1)[0]

    if "<END>" in answer:
        answer = answer.split("<END>", 1)[0]

    return answer.strip(), generated_count, elapsed, kv_cache_used


# ============================================================
# Flask API
# ============================================================

app = Flask(__name__)


def check_api_key() -> bool:
    expected = os.environ.get("API_KEY")

    # If no API key is configured, the API is public.
    if not expected:
        return True

    authorization = request.headers.get("Authorization", "")

    if authorization.startswith("Bearer "):
        supplied = authorization[len("Bearer "):]
        if supplied == expected:
            return True

    supplied = request.headers.get("X-API-Key", "")
    return supplied == expected


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "TinyGPT API",
        "status": "online",
        "model_loaded": True,
        "optimized_inference": True,
        "kv_cache": True,
        "qkv_fusion": True,
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /info": "Model information",
            "POST /generate": "Generate a TinyGPT response",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
    })


@app.route("/info", methods=["GET"])
def info():
    metadata = {}

    for key, value in model.meta.items():
        if isinstance(value, np.generic):
            value = value.item()
        metadata[key] = value

    return jsonify({
        "model": metadata,
        "server": {
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "default_max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "optimized_inference": True,
            "kv_cache": True,
            "qkv_fusion": True,
            "blas_threads": _TINY_GPT_THREADS,
        },
    })


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    if not check_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be JSON."}), 400

    prompt = data.get("prompt")

    if not isinstance(prompt, str):
        return jsonify({"error": "'prompt' must be a string."}), 400

    prompt = prompt.strip()

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    if len(prompt) > MAX_PROMPT_CHARS:
        return jsonify({
            "error": (
                f"Prompt is too long. Maximum is {MAX_PROMPT_CHARS} characters."
            )
        }), 400

    try:
        temperature = float(data.get("temperature", 0.70))
        top_k = int(data.get("top_k", 30))
        top_p = float(data.get("top_p", 0.90))
        repetition_penalty = float(
            data.get("repetition_penalty", 1.05)
        )
        max_new_tokens = int(
            data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        )

        seed = data.get("seed", None)
        if seed is not None:
            seed = int(seed)

        # Safety / validity limits.
        temperature = min(max(temperature, 0.05), 3.0)
        top_k = min(max(top_k, 0), model.vocab_size)
        top_p = min(max(top_p, 0.05), 1.0)
        repetition_penalty = min(max(repetition_penalty, 1.0), 2.0)
        max_new_tokens = min(max(max_new_tokens, 1), 180)

    except (ValueError, TypeError):
        return jsonify({
            "error": "Invalid generation parameters."
        }), 400

    try:
        # Serialize CPU-heavy generation so concurrent requests do not
        # compete for the small free CPU allocation.
        with MODEL_LOCK:
            answer, token_count, elapsed, kv_cache_used = generate(
                prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )

        tokens_per_second = token_count / elapsed if elapsed > 0 else 0.0

        return jsonify({
            "prompt": prompt,
            "response": answer,
            "generated_tokens": token_count,
            "generation_seconds": round(elapsed, 4),
            "tokens_per_second": round(tokens_per_second, 3),
            "kv_cache_used": kv_cache_used,
        })

    except Exception as exc:
        print("Generation error:", repr(exc))
        return jsonify({
            "error": "Generation failed.",
            "details": str(exc),
        }), 500


# ============================================================
# Local execution
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))

    # Flask's development server is useful for local testing.
    # On Render, use Gunicorn as the start command instead.
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
    )
