import os
import json
import re
import threading
import time

import numpy as np
from flask import Flask, request, jsonify


# ============================================================
# Configuration
# ============================================================

MODEL_FILE = os.environ.get(
    "MODEL_FILE",
    "tiny_gpt_pc.npz"
)

MAX_PROMPT_CHARS = 8000

DEFAULT_TEMPERATURE = 0.75
DEFAULT_TOP_K = 30
DEFAULT_TOP_P = 0.90
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_NEW_TOKENS = 60

MAX_NEW_TOKENS_LIMIT = 180

MODEL_LOCK = threading.Lock()


# ============================================================
# Tokenizer
# ============================================================

class HybridTokenizer:

    def __init__(self, tokens):

        self.tokens = list(tokens)

        self.token_to_id = {
            token: i
            for i, token in enumerate(self.tokens)
        }

        self.id_to_token = {
            i: token
            for i, token in enumerate(self.tokens)
        }

        self.vocab_size = len(self.tokens)

        self.special_tokens = {
            "<START>",
            "<USER>",
            "<ASSISTANT>",
            "<END>",
            "<UNK>"
        }

        self.pattern = re.compile(
            r"<START>|<USER>|<ASSISTANT>|<END>|<UNK>|"
            r"[A-Za-z]+(?:'[A-Za-z]+)?|"
            r"[0-9]+(?:\.[0-9]+)?|"
            r"\s+|"
            r"[^A-Za-z0-9\s]"
        )

    def encode(self, text):

        pieces = self.pattern.findall(text)

        result = []

        for piece in pieces:

            if piece in self.token_to_id:

                result.append(
                    self.token_to_id[piece]
                )

                continue

            if re.fullmatch(
                r"[A-Za-z]+(?:'[A-Za-z]+)?",
                piece
            ):

                lower = piece.lower()

                if lower in self.token_to_id:

                    result.append(
                        self.token_to_id[lower]
                    )

                    continue

            for char in piece:

                if char in self.token_to_id:

                    result.append(
                        self.token_to_id[char]
                    )

                else:

                    result.append(
                        self.token_to_id.get(
                            "<UNK>",
                            0
                        )
                    )

        return result

    def decode(self, ids):

        output = []

        for token_id in ids:

            token = self.id_to_token.get(
                int(token_id),
                ""
            )

            if token in self.special_tokens:
                continue

            output.append(token)

        return "".join(output)


# ============================================================
# TinyGPT inference model
# ============================================================

class TinyGPT:

    def __init__(self, data):

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        meta_raw = data["meta_json"]

        if isinstance(meta_raw, np.ndarray):
            meta_raw = meta_raw.item()

        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")

        if isinstance(meta_raw, str):
            self.meta = json.loads(meta_raw)
        else:
            self.meta = meta_raw

        self.block_size = int(
            self.meta["block_size"]
        )

        self.embed_size = int(
            self.meta["embed_size"]
        )

        self.num_heads = int(
            self.meta["num_heads"]
        )

        self.num_layers = int(
            self.meta["num_layers"]
        )

        self.vocab_size = int(
            self.meta["vocab_size"]
        )

        self.head_dim = (
            self.embed_size //
            self.num_heads
        )

        self.tokenizer = HybridTokenizer(
            self.meta["tokens"]
        )

        # ----------------------------------------------------
        # Load parameters
        # ----------------------------------------------------

        self.params = {}

        for key in data.files:

            if key.startswith("param_"):

                name = key[len("param_"):]

                self.params[name] = data[
                    key
                ].astype(
                    np.float32,
                    copy=False
                )

        # ----------------------------------------------------
        # Special tokens
        # ----------------------------------------------------

        self.start_id = (
            self.tokenizer.token_to_id.get(
                "<START>"
            )
        )

        self.user_id = (
            self.tokenizer.token_to_id.get(
                "<USER>"
            )
        )

        self.assistant_id = (
            self.tokenizer.token_to_id.get(
                "<ASSISTANT>"
            )
        )

        self.end_id = (
            self.tokenizer.token_to_id.get(
                "<END>"
            )
        )

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        self.causal_mask = np.triu(
            np.ones(
                (
                    self.block_size,
                    self.block_size
                ),
                dtype=bool
            ),
            k=1
        )

        # ----------------------------------------------------
        # Fused QKV matrices
        #
        # The original model has separate Wq/Wk/Wv.
        # We concatenate them once during startup so that
        # every forward pass uses ONE matrix multiplication
        # instead of three.
        # ----------------------------------------------------

        self.qkv_weights = []

        for layer in range(
            self.num_layers
        ):

            prefix = f"layer{layer}_"

            Wq = self.params[
                prefix + "Wq"
            ]

            Wk = self.params[
                prefix + "Wk"
            ]

            Wv = self.params[
                prefix + "Wv"
            ]

            self.qkv_weights.append(
                np.concatenate(
                    [
                        Wq,
                        Wk,
                        Wv
                    ],
                    axis=1
                )
            )

        print("TinyGPT model loaded.")

        print(
            "Vocabulary:",
            self.vocab_size
        )

        print(
            "Embedding:",
            self.embed_size
        )

        print(
            "Layers:",
            self.num_layers
        )

        print(
            "Heads:",
            self.num_heads
        )

        print(
            "Context:",
            self.block_size
        )

        print(
            "Parameters:",
            len(self.params)
        )

        print(
            "Optimizations: "
            "fused QKV + KV cache"
        )

    # ========================================================
    # RMSNorm
    # ========================================================

    @staticmethod
    def rms_norm(
        x,
        weight,
        eps=1e-5
    ):

        mean_square = np.mean(
            x * x,
            axis=-1,
            keepdims=True
        )

        return (
            x /
            np.sqrt(
                mean_square + eps
            )
        ) * weight

    # ========================================================
    # GELU
    # ========================================================

    @staticmethod
    def gelu(x):

        return 0.5 * x * (
            1.0 +
            np.tanh(
                np.sqrt(2.0 / np.pi)
                *
                (
                    x +
                    0.044715 *
                    x * x * x
                )
            )
        )

    # ========================================================
    # Prefill
    #
    # Processes the original prompt once and builds the
    # K/V cache.
    # ========================================================

    def prefill(self, token_ids):

        token_ids = np.asarray(
            token_ids,
            dtype=np.int64
        )

        T = len(token_ids)

        if T > self.block_size:

            token_ids = token_ids[
                -self.block_size:
            ]

            T = self.block_size

        C = self.embed_size

        token_embedding = self.params[
            "token_embedding"
        ]

        position_embedding = self.params[
            "position_embedding"
        ]

        x = (
            token_embedding[token_ids]
            +
            position_embedding[
                np.arange(T)
            ]
        )

        # Each element contains the complete K/V history
        # for one transformer layer.

        cache = []

        for layer in range(
            self.num_layers
        ):

            prefix = f"layer{layer}_"

            # ------------------------------------------------
            # Attention normalization
            # ------------------------------------------------

            x_residual = x

            h = self.rms_norm(
                x,
                self.params[
                    prefix + "norm1"
                ]
            )

            # ------------------------------------------------
            # Fused QKV
            # ------------------------------------------------

            qkv = h @ self.qkv_weights[
                layer
            ]

            Q = qkv[
                :, :C
            ]

            K = qkv[
                :, C:2 * C
            ]

            V = qkv[
                :, 2 * C:
            ]

            Q = Q.reshape(
                T,
                self.num_heads,
                self.head_dim
            ).transpose(
                1,
                0,
                2
            )

            K = K.reshape(
                T,
                self.num_heads,
                self.head_dim
            ).transpose(
                1,
                0,
                2
            )

            V = V.reshape(
                T,
                self.num_heads,
                self.head_dim
            ).transpose(
                1,
                0,
                2
            )

            # Save K/V for future tokens.

            cache.append({
                "k": K.copy(),
                "v": V.copy()
            })

            # ------------------------------------------------
            # Attention
            # ------------------------------------------------

            scores = (
                Q @
                K.transpose(
                    0,
                    2,
                    1
                )
            ) / np.sqrt(
                self.head_dim
            )

            mask = self.causal_mask[
                :T,
                :T
            ]

            scores = np.where(
                mask[None, :, :],
                -1e9,
                scores
            )

            scores -= np.max(
                scores,
                axis=-1,
                keepdims=True
            )

            weights = np.exp(
                scores
            )

            weights /= (
                np.sum(
                    weights,
                    axis=-1,
                    keepdims=True
                )
                + 1e-9
            )

            attention = (
                weights @ V
            )

            attention = (
                attention
                .transpose(
                    1,
                    0,
                    2
                )
                .reshape(
                    T,
                    C
                )
            )

            x = (
                x_residual +
                attention @ self.params[
                    prefix + "Wo"
                ]
            )

            # ------------------------------------------------
            # Feed-forward
            # ------------------------------------------------

            x_residual = x

            h = self.rms_norm(
                x,
                self.params[
                    prefix + "norm2"
                ]
            )

            h = (
                h @ self.params[
                    prefix + "W1"
                ]
            )

            h += self.params[
                prefix + "b1"
            ]

            h = self.gelu(h)

            h = (
                h @ self.params[
                    prefix + "W2"
                ]
            )

            h += self.params[
                prefix + "b2"
            ]

            x = (
                x_residual +
                h
            )

        # ----------------------------------------------------
        # Final logits
        # ----------------------------------------------------

        x = self.rms_norm(
            x,
            self.params[
                "final_norm"
            ]
        )

        logits = (
            x @ token_embedding.T
        )

        return logits, cache, T

    # ========================================================
    # Decode ONE new token using KV cache
    # ========================================================

    def decode_token(
        self,
        token_id,
        position,
        cache
    ):

        C = self.embed_size

        token_embedding = self.params[
            "token_embedding"
        ]

        position_embedding = self.params[
            "position_embedding"
        ]

        # ----------------------------------------------------
        # Single-token embedding
        # ----------------------------------------------------

        x = (
            token_embedding[
                int(token_id)
            ]
            +
            position_embedding[
                position
            ]
        )

        x = x.reshape(
            1,
            C
        )

        for layer in range(
            self.num_layers
        ):

            prefix = f"layer{layer}_"

            x_residual = x

            h = self.rms_norm(
                x,
                self.params[
                    prefix + "norm1"
                ]
            )

            # ------------------------------------------------
            # Fused QKV
            # ------------------------------------------------

            qkv = (
                h @
                self.qkv_weights[
                    layer
                ]
            )

            Q = qkv[
                :, :C
            ]

            K = qkv[
                :, C:2 * C
            ]

            V = qkv[
                :, 2 * C:
            ]

            Q = Q.reshape(
                self.num_heads,
                self.head_dim
            )

            K = K.reshape(
                self.num_heads,
                self.head_dim
            )

            V = V.reshape(
                self.num_heads,
                self.head_dim
            )

            # ------------------------------------------------
            # Append only ONE new K/V entry
            # ------------------------------------------------

            cache[layer]["k"] = np.concatenate(
                [
                    cache[layer]["k"],
                    K[:, None, :]
                ],
                axis=1
            )

            cache[layer]["v"] = np.concatenate(
                [
                    cache[layer]["v"],
                    V[:, None, :]
                ],
                axis=1
            )

            keys = cache[layer]["k"]
            values = cache[layer]["v"]

            # ------------------------------------------------
            # Attention against cached history
            # ------------------------------------------------

            scores = np.einsum(
                "hd,hnd->hn",
                Q,
                keys
            )

            scores /= np.sqrt(
                self.head_dim
            )

            scores -= np.max(
                scores,
                axis=-1,
                keepdims=True
            )

            weights = np.exp(
                scores
            )

            weights /= (
                np.sum(
                    weights,
                    axis=-1,
                    keepdims=True
                )
                + 1e-9
            )

            attention = np.einsum(
                "hn,hnd->hd",
                weights,
                values
            )

            attention = attention.reshape(
                1,
                C
            )

            x = (
                x_residual +
                attention @ self.params[
                    prefix + "Wo"
                ]
            )

            # ------------------------------------------------
            # Feed-forward
            # ------------------------------------------------

            x_residual = x

            h = self.rms_norm(
                x,
                self.params[
                    prefix + "norm2"
                ]
            )

            h = (
                h @ self.params[
                    prefix + "W1"
                ]
            )

            h += self.params[
                prefix + "b1"
            ]

            h = self.gelu(h)

            h = (
                h @ self.params[
                    prefix + "W2"
                ]
            )

            h += self.params[
                prefix + "b2"
            ]

            x = (
                x_residual +
                h
            )

        # ----------------------------------------------------
        # Final logits
        # ----------------------------------------------------

        x = self.rms_norm(
            x,
            self.params[
                "final_norm"
            ]
        )

        logits = (
            x @ token_embedding.T
        )

        return logits[0]


# ============================================================
# Load model
# ============================================================

# This is intentionally the SAME file locating method
# from your original working server.

if not os.path.exists(
    MODEL_FILE
):

    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}\n"
        "Make sure tiny_gpt_pc.npz is in the repository."
    )

print(
    f"Loading TinyGPT model from "
    f"{MODEL_FILE}..."
)

model_data = np.load(
    MODEL_FILE,
    allow_pickle=True
)

model = TinyGPT(
    model_data
)


# ============================================================
# Sampling
# ============================================================

def sample_token(
    logits,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    generated_ids,
    rng
):

    logits = logits.astype(
        np.float64,
        copy=True
    )

    # --------------------------------------------------------
    # Repetition penalty
    # --------------------------------------------------------

    if repetition_penalty > 1.0:

        recent = set(
            generated_ids[-64:]
        )

        for token_id in recent:

            if (
                0 <= token_id
                < len(logits)
            ):

                if logits[token_id] > 0:

                    logits[token_id] /= (
                        repetition_penalty
                    )

                else:

                    logits[token_id] *= (
                        repetition_penalty
                    )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    logits /= max(
        float(temperature),
        0.05
    )

    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    if (
        top_k > 0
        and
        top_k < len(logits)
    ):

        indices = np.argpartition(
            logits,
            -top_k
        )[-top_k:]

        filtered = np.full_like(
            logits,
            -np.inf
        )

        filtered[indices] = (
            logits[indices]
        )

        logits = filtered

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    max_logit = np.max(
        logits
    )

    probabilities = np.exp(
        logits - max_logit
    )

    total = np.sum(
        probabilities
    )

    if (
        total <= 0
        or
        not np.isfinite(total)
    ):

        probabilities = (
            np.ones_like(
                probabilities
            )
            /
            len(probabilities)
        )

    else:

        probabilities /= total

    # --------------------------------------------------------
    # Top-P
    # --------------------------------------------------------

    if (
        0.0 < top_p < 1.0
    ):

        sorted_indices = np.argsort(
            probabilities
        )[::-1]

        sorted_probs = (
            probabilities[
                sorted_indices
            ]
        )

        cumulative = np.cumsum(
            sorted_probs
        )

        remove = (
            cumulative > top_p
        )

        if len(remove) > 0:
            remove[0] = False

        probabilities[
            sorted_indices[remove]
        ] = 0.0

        total = np.sum(
            probabilities
        )

        if total > 0:

            probabilities /= total

    return int(
        rng.choice(
            len(probabilities),
            p=probabilities
        )
    )


# ============================================================
# Generation
# ============================================================

def generate(
    prompt,
    temperature=DEFAULT_TEMPERATURE,
    top_k=DEFAULT_TOP_K,
    top_p=DEFAULT_TOP_P,
    repetition_penalty=DEFAULT_REPETITION_PENALTY,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    seed=None
):

    tokenizer = model.tokenizer

    # --------------------------------------------------------
    # EXACT TinyGPT v8 prompt format
    # --------------------------------------------------------

    full_prompt = (
        "<START>\n"
        "<USER> "
        + prompt +
        "\n"
        "<ASSISTANT> "
    )

    token_ids = tokenizer.encode(
        full_prompt
    )

    if not token_ids:

        return "", 0

    # --------------------------------------------------------
    # Limit prompt to model context
    # --------------------------------------------------------

    if len(token_ids) > model.block_size:

        token_ids = token_ids[
            -model.block_size:
        ]

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # PREFILL
    #
    # This is the only point where the entire prompt goes
    # through the attention mechanism.
    # --------------------------------------------------------

    logits, cache, prompt_length = (
        model.prefill(
            token_ids
        )
    )

    generated_ids = []
    answer_ids = []

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    for _ in range(
        min(
            int(max_new_tokens),
            MAX_NEW_TOKENS_LIMIT
        )
    ):

        next_logits = logits[-1]

        next_token = sample_token(
            next_logits,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            generated_ids,
            rng
        )

        # ----------------------------------------------------
        # Stop tokens
        # ----------------------------------------------------

        if (
            model.end_id is not None
            and
            next_token == model.end_id
        ):
            break

        if (
            model.user_id is not None
            and
            next_token == model.user_id
        ):
            break

        if (
            model.start_id is not None
            and
            next_token == model.start_id
        ):
            break

        if (
            model.assistant_id is not None
            and
            next_token == model.assistant_id
        ):
            break

        generated_ids.append(
            next_token
        )

        answer_ids.append(
            next_token
        )

        # ----------------------------------------------------
        # Stop if context is full
        # ----------------------------------------------------

        position = (
            prompt_length +
            len(generated_ids)
        )

        if position >= model.block_size:
            break

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Only the NEW token goes through the model.
        # The previous K/V values remain cached.
        # ----------------------------------------------------

        logits = model.decode_token(
            next_token,
            position,
            cache
        )

    answer = tokenizer.decode(
        answer_ids
    )

    return (
        answer.strip(),
        len(answer_ids)
    )


# ============================================================
# Flask
# ============================================================

app = Flask(
    __name__
)


# ============================================================
# Optional API key
# ============================================================

def check_api_key():

    expected = os.environ.get(
        "API_KEY"
    )

    if not expected:
        return True

    authorization = request.headers.get(
        "Authorization",
        ""
    )

    if authorization.startswith(
        "Bearer "
    ):

        supplied = authorization[
            7:
        ]

        return supplied == expected

    supplied = request.headers.get(
        "X-API-Key",
        ""
    )

    return supplied == expected


# ============================================================
# Root
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    return jsonify({

        "name":
            "TinyGPT API",

        "status":
            "online",

        "model_loaded":
            True,

        "model_file":
            MODEL_FILE,

        "kv_cache":
            True,

        "fused_qkv":
            True,

        "endpoints": {

            "GET /":
                "API information",

            "GET /health":
                "Health check",

            "GET /info":
                "Model information",

            "POST /generate":
                "Generate TinyGPT response"
        }
    })


# ============================================================
# Health
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "model_loaded":
            True,

        "model_file":
            MODEL_FILE,

        "kv_cache":
            True,

        "fused_qkv":
            True
    })


# ============================================================
# Info
# ============================================================

@app.route(
    "/info",
    methods=["GET"]
)
def info():

    metadata = {}

    for key, value in (
        model.meta.items()
    ):

        if isinstance(
            value,
            np.generic
        ):

            value = value.item()

        metadata[key] = value

    return jsonify({

        "model":
            metadata,

        "server": {

            "model_file":
                MODEL_FILE,

            "kv_cache":
                True,

            "fused_qkv":
                True,

            "max_prompt_chars":
                MAX_PROMPT_CHARS,

            "default_temperature":
                DEFAULT_TEMPERATURE,

            "default_top_k":
                DEFAULT_TOP_K,

            "default_top_p":
                DEFAULT_TOP_P,

            "default_repetition_penalty":
                DEFAULT_REPETITION_PENALTY,

            "default_max_new_tokens":
                DEFAULT_MAX_NEW_TOKENS
        }
    })


# ============================================================
# Generate endpoint
# ============================================================

@app.route(
    "/generate",
    methods=["POST"]
)
def generate_endpoint():

    request_start = (
        time.perf_counter()
    )

    if not check_api_key():

        return jsonify({
            "error":
                "Unauthorized"
        }), 401

    data = request.get_json(
        silent=True
    )

    if not isinstance(
        data,
        dict
    ):

        return jsonify({
            "error":
                "Request body must be JSON."
        }), 400

    prompt = data.get(
        "prompt"
    )

    if not isinstance(
        prompt,
        str
    ):

        return jsonify({
            "error":
                "'prompt' must be a string."
        }), 400

    prompt = prompt.strip()

    if not prompt:

        return jsonify({
            "error":
                "Prompt cannot be empty."
        }), 400

    if len(prompt) > MAX_PROMPT_CHARS:

        return jsonify({
            "error":
                (
                    "Prompt is too long. "
                    f"Maximum is "
                    f"{MAX_PROMPT_CHARS} characters."
                )
        }), 400

    try:

        temperature = float(
            data.get(
                "temperature",
                DEFAULT_TEMPERATURE
            )
        )

        top_k = int(
            data.get(
                "top_k",
                DEFAULT_TOP_K
            )
        )

        top_p = float(
            data.get(
                "top_p",
                DEFAULT_TOP_P
            )
        )

        repetition_penalty = float(
            data.get(
                "repetition_penalty",
                DEFAULT_REPETITION_PENALTY
            )
        )

        max_new_tokens = int(
            data.get(
                "max_new_tokens",
                DEFAULT_MAX_NEW_TOKENS
            )
        )

        seed = data.get(
            "seed",
            None
        )

        if seed is not None:

            seed = int(
                seed
            )

        # ----------------------------------------------------
        # Clamp values
        # ----------------------------------------------------

        temperature = min(
            max(
                temperature,
                0.05
            ),
            3.0
        )

        top_k = min(
            max(
                top_k,
                0
            ),
            model.vocab_size
        )

        top_p = min(
            max(
                top_p,
                0.05
            ),
            1.0
        )

        repetition_penalty = min(
            max(
                repetition_penalty,
                1.0
            ),
            2.0
        )

        max_new_tokens = min(
            max(
                max_new_tokens,
                1
            ),
            MAX_NEW_TOKENS_LIMIT
        )

    except (
        ValueError,
        TypeError
    ):

        return jsonify({
            "error":
                "Invalid generation parameters."
        }), 400

    # ========================================================
    # Generation
    # ========================================================

    try:

        with MODEL_LOCK:

            generation_start = (
                time.perf_counter()
            )

            answer, generated_tokens = (
                generate(
                    prompt=prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=
                        repetition_penalty,
                    max_new_tokens=
                        max_new_tokens,
                    seed=seed
                )
            )

            generation_seconds = (
                time.perf_counter()
                -
                generation_start
            )

        total_seconds = (
            time.perf_counter()
            -
            request_start
        )

        if generation_seconds > 0:

            tokens_per_second = (
                generated_tokens /
                generation_seconds
            )

        else:

            tokens_per_second = 0.0

        return jsonify({

            "response":
                answer,

            "generated_tokens":
                generated_tokens,

            "generation_seconds":
                round(
                    generation_seconds,
                    4
                ),

            "tokens_per_second":
                round(
                    tokens_per_second,
                    2
                ),

            "total_seconds":
                round(
                    total_seconds,
                    4
                ),

            "kv_cache_used":
                True,

            "fused_qkv":
                True,

            "model_file":
                MODEL_FILE
        })

    except Exception as exc:

        print(
            "Generation error:",
            repr(exc)
        )

        return jsonify({

            "error":
                "Generation failed.",

            "details":
                str(exc)

        }), 500


# ============================================================
# Local execution
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "10000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        threaded=False
            )
