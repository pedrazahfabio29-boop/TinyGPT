import os
import time
import threading

import numpy as np
from flask import Flask, request, jsonify


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "model.npz")

DEFAULT_TEMPERATURE = 0.75
DEFAULT_TOP_K = 30
DEFAULT_TOP_P = 0.90
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_NEW_TOKENS = 60

MAX_ALLOWED_TOKENS = 180

MODEL_LOCK = threading.Lock()


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# TOKENIZER
# ============================================================

class HybridTokenizer:
    def __init__(self, data):
        self.word_to_id = {}
        self.id_to_word = {}

        for key in data.files:
            if key.startswith("word_to_id_"):
                word = key[len("word_to_id_"):]
                self.word_to_id[word] = int(data[key])

        for key in data.files:
            if key.startswith("id_to_word_"):
                idx = int(key[len("id_to_word_"):])
                value = data[key]

                if isinstance(value, np.ndarray):
                    value = value.item()

                self.id_to_word[idx] = str(value)

        self.START = self.word_to_id.get("<START>", 0)
        self.USER = self.word_to_id.get("<USER>", 1)
        self.ASSISTANT = self.word_to_id.get("<ASSISTANT>", 2)
        self.END = self.word_to_id.get("<END>", 3)
        self.UNK = self.word_to_id.get("<UNK>", 4)

        self.vocab_size = max(
            len(self.word_to_id),
            max(self.id_to_word.keys(), default=0) + 1
        )

    def encode(self, text):
        words = text.strip().split()

        result = []

        for word in words:
            result.append(
                self.word_to_id.get(word, self.UNK)
            )

        return result

    def decode(self, ids):
        words = []

        for idx in ids:
            idx = int(idx)

            word = self.id_to_word.get(idx)

            if word is None:
                continue

            words.append(word)

        return " ".join(words)


# ============================================================
# MODEL
# ============================================================

class TinyGPT:

    def __init__(self, data):
        self.data = data

        self.embed_size = int(
            data["embed_size"]
            if "embed_size" in data
            else 128
        )

        self.num_heads = int(
            data["num_heads"]
            if "num_heads" in data
            else 4
        )

        self.num_layers = int(
            data["num_layers"]
            if "num_layers" in data
            else 4
        )

        self.block_size = int(
            data["block_size"]
            if "block_size" in data
            else 128
        )

        self.vocab_size = int(
            data["vocab_size"]
            if "vocab_size" in data
            else 3000
        )

        self.head_dim = self.embed_size // self.num_heads

        self.token_embedding = data["token_embedding"]
        self.position_embedding = data["position_embedding"]

        self.layers = []

        for i in range(self.num_layers):

            prefix = f"layer_{i}_"

            layer = {
                "ln1_g": data[prefix + "ln1_g"],
                "ln2_g": data[prefix + "ln2_g"],

                "Wq": data[prefix + "Wq"],
                "Wk": data[prefix + "Wk"],
                "Wv": data[prefix + "Wv"],
                "Wo": data[prefix + "Wo"],

                "W1": data[prefix + "W1"],
                "W2": data[prefix + "W2"],
            }

            self.layers.append(layer)

        self.final_norm_g = data["final_norm_g"]
        self.lm_head = data["lm_head"]

    # --------------------------------------------------------
    # RMS NORM
    # --------------------------------------------------------

    @staticmethod
    def rms_norm(x, g, eps=1e-5):

        variance = np.mean(
            x * x,
            axis=-1,
            keepdims=True
        )

        return (
            x / np.sqrt(variance + eps)
        ) * g

    # --------------------------------------------------------
    # GELU
    # --------------------------------------------------------

    @staticmethod
    def gelu(x):

        return 0.5 * x * (
            1.0 +
            np.tanh(
                np.sqrt(2.0 / np.pi) *
                (
                    x +
                    0.044715 *
                    np.power(x, 3)
                )
            )
        )

    # --------------------------------------------------------
    # FULL FORWARD
    # --------------------------------------------------------

    def forward(self, token_ids):

        token_ids = np.asarray(
            token_ids,
            dtype=np.int32
        )

        seq_len = len(token_ids)

        if seq_len > self.block_size:
            token_ids = token_ids[-self.block_size:]
            seq_len = len(token_ids)

        x = self.token_embedding[token_ids]

        positions = np.arange(
            seq_len,
            dtype=np.int32
        )

        x = x + self.position_embedding[positions]

        for layer in self.layers:

            residual = x

            h = self.rms_norm(
                x,
                layer["ln1_g"]
            )

            q = h @ layer["Wq"]
            k = h @ layer["Wk"]
            v = h @ layer["Wv"]

            q = q.reshape(
                seq_len,
                self.num_heads,
                self.head_dim
            ).transpose(1, 0, 2)

            k = k.reshape(
                seq_len,
                self.num_heads,
                self.head_dim
            ).transpose(1, 0, 2)

            v = v.reshape(
                seq_len,
                self.num_heads,
                self.head_dim
            ).transpose(1, 0, 2)

            scores = (
                q @ k.transpose(0, 2, 1)
            ) / np.sqrt(self.head_dim)

            mask = np.triu(
                np.ones(
                    (seq_len, seq_len),
                    dtype=bool
                ),
                k=1
            )

            scores[:, mask] = -1e9

            scores -= np.max(
                scores,
                axis=-1,
                keepdims=True
            )

            attention = np.exp(scores)

            attention /= (
                np.sum(
                    attention,
                    axis=-1,
                    keepdims=True
                ) + 1e-9
            )

            out = attention @ v

            out = out.transpose(
                1, 0, 2
            ).reshape(
                seq_len,
                self.embed_size
            )

            x = residual + (
                out @ layer["Wo"]
            )

            residual = x

            h = self.rms_norm(
                x,
                layer["ln2_g"]
            )

            h = self.gelu(
                h @ layer["W1"]
            )

            x = residual + (
                h @ layer["W2"]
            )

        x = self.rms_norm(
            x,
            self.final_norm_g
        )

        logits = x @ self.lm_head

        return logits

    # --------------------------------------------------------
    # KV-CACHE FOR ONE TOKEN AT A TIME
    # --------------------------------------------------------

    def forward_cached(self, token_id, position, cache):

        token_id = int(token_id)

        x = self.token_embedding[
            token_id
        ].astype(np.float32)

        x = x + self.position_embedding[
            position
        ]

        x = x[None, :]

        for layer_index, layer in enumerate(self.layers):

            residual = x

            h = self.rms_norm(
                x,
                layer["ln1_g"]
            )

            q = h @ layer["Wq"]
            k = h @ layer["Wk"]
            v = h @ layer["Wv"]

            q = q.reshape(
                self.num_heads,
                self.head_dim
            )

            k = k.reshape(
                self.num_heads,
                self.head_dim
            )

            v = v.reshape(
                self.num_heads,
                self.head_dim
            )

            cache[layer_index]["k"].append(k)
            cache[layer_index]["v"].append(v)

            keys = np.stack(
                cache[layer_index]["k"],
                axis=1
            )

            values = np.stack(
                cache[layer_index]["v"],
                axis=1
            )

            scores = np.einsum(
                "hd,hnd->hn",
                q,
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

            attention = np.exp(scores)

            attention /= (
                np.sum(
                    attention,
                    axis=-1,
                    keepdims=True
                ) + 1e-9
            )

            out = np.einsum(
                "hn,hnd->hd",
                attention,
                values
            )

            out = out.reshape(
                1,
                self.embed_size
            )

            x = residual + (
                out @ layer["Wo"]
            )

            residual = x

            h = self.rms_norm(
                x,
                layer["ln2_g"]
            )

            h = self.gelu(
                h @ layer["W1"]
            )

            x = residual + (
                h @ layer["W2"]
            )

        x = self.rms_norm(
            x,
            self.final_norm_g
        )

        logits = x @ self.lm_head

        return logits[0]


# ============================================================
# MODEL LOADING
# ============================================================

print("=" * 60)
print("Loading TinyGPT...")
print("=" * 60)

MODEL_DATA = np.load(
    MODEL_PATH,
    allow_pickle=True
)

TOKENIZER = HybridTokenizer(
    MODEL_DATA
)

MODEL = TinyGPT(
    MODEL_DATA
)

print(
    f"Vocabulary: {TOKENIZER.vocab_size}"
)

print(
    f"Embedding size: {MODEL.embed_size}"
)

print(
    f"Layers: {MODEL.num_layers}"
)

print(
    f"Heads: {MODEL.num_heads}"
)

print(
    f"Context size: {MODEL.block_size}"
)

print("=" * 60)
print("TinyGPT loaded successfully")
print("=" * 60)


# ============================================================
# SAMPLING
# ============================================================

def sample_token(
    logits,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    generated_ids
):

    logits = logits.astype(
        np.float64,
        copy=True
    )

    # --------------------------------------------------------
    # Repetition penalty
    # --------------------------------------------------------

    if repetition_penalty != 1.0:

        for token_id in set(generated_ids):

            if token_id < len(logits):

                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    temperature = max(
        float(temperature),
        1e-5
    )

    logits /= temperature

    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    if top_k > 0 and top_k < len(logits):

        indices = np.argpartition(
            logits,
            -top_k
        )[-top_k:]

        filtered = np.full_like(
            logits,
            -np.inf
        )

        filtered[indices] = logits[indices]

        logits = filtered

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    max_logit = np.max(logits)

    probabilities = np.exp(
        logits - max_logit
    )

    probabilities /= (
        np.sum(probabilities) + 1e-12
    )

    # --------------------------------------------------------
    # Top-P
    # --------------------------------------------------------

    if 0.0 < top_p < 1.0:

        sorted_indices = np.argsort(
            probabilities
        )[::-1]

        sorted_probs = probabilities[
            sorted_indices
        ]

        cumulative = np.cumsum(
            sorted_probs
        )

        remove = cumulative > top_p

        if np.any(remove):

            first = np.argmax(remove)

            remove[:first] = False

            probabilities[
                sorted_indices[remove]
            ] = 0.0

            probabilities /= (
                np.sum(probabilities) + 1e-12
            )

    return int(
        np.random.choice(
            len(probabilities),
            p=probabilities
        )
    )


# ============================================================
# GENERATION
# ============================================================

def generate(
    question,
    max_new_tokens=60,
    temperature=0.75,
    top_k=30,
    top_p=0.90,
    repetition_penalty=1.05,
    seed=None
):

    if seed is not None:
        np.random.seed(
            int(seed)
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # TinyGPT v8 was trained on:
    #
    # <START>
    # <USER> question
    # <ASSISTANT> answer
    # <END>
    #
    # It is NOT trained on a multi-turn transcript.
    # --------------------------------------------------------

    prompt_ids = [
        TOKENIZER.START,
        TOKENIZER.USER
    ]

    prompt_ids.extend(
        TOKENIZER.encode(question)
    )

    prompt_ids.append(
        TOKENIZER.ASSISTANT
    )

    # Keep the prompt inside the model context.

    if len(prompt_ids) >= MODEL.block_size:

        prompt_ids = prompt_ids[
            -MODEL.block_size:
        ]

    # --------------------------------------------------------
    # Build KV cache
    # --------------------------------------------------------

    cache = []

    for _ in range(MODEL.num_layers):

        cache.append({
            "k": [],
            "v": []
        })

    # --------------------------------------------------------
    # Process prompt once
    # --------------------------------------------------------

    logits = None

    for position, token_id in enumerate(prompt_ids):

        logits = MODEL.forward_cached(
            token_id,
            position,
            cache
        )

    generated = []

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    for _ in range(
        min(
            int(max_new_tokens),
            MAX_ALLOWED_TOKENS
        )
    ):

        token_id = sample_token(
            logits,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            generated
        )

        # Stop tokens

        if token_id == TOKENIZER.END:
            break

        if token_id == TOKENIZER.USER:
            break

        if token_id == TOKENIZER.START:
            break

        if token_id == TOKENIZER.ASSISTANT:
            break

        generated.append(
            token_id
        )

        position = (
            len(prompt_ids) +
            len(generated) -
            1
        )

        if position >= MODEL.block_size:
            break

        logits = MODEL.forward_cached(
            token_id,
            position,
            cache
        )

    answer = TOKENIZER.decode(
        generated
    )

    return answer.strip(), len(generated)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None
    })


# ============================================================
# GENERATE API
# ============================================================

@app.route("/generate", methods=["POST"])
def generate_endpoint():

    start_time = time.perf_counter()

    try:

        data = request.get_json(
            silent=True
        ) or {}

        prompt = data.get(
            "prompt",
            ""
        )

        if not isinstance(
            prompt,
            str
        ):

            return jsonify({
                "error": "prompt must be a string"
            }), 400

        prompt = prompt.strip()

        if not prompt:

            return jsonify({
                "error": "prompt is empty"
            }), 400

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

        max_new_tokens = max(
            1,
            min(
                max_new_tokens,
                MAX_ALLOWED_TOKENS
            )
        )

        temperature = max(
            0.01,
            min(
                temperature,
                2.0
            )
        )

        top_k = max(
            0,
            min(
                top_k,
                TOKENIZER.vocab_size
            )
        )

        top_p = max(
            0.01,
            min(
                top_p,
                1.0
            )
        )

        repetition_penalty = max(
            1.0,
            min(
                repetition_penalty,
                2.0
            )
        )

        # Only one generation at a time.
        # This prevents multiple requests from
        # competing for the small CPU/RAM available
        # on Render Free.

        with MODEL_LOCK:

            generation_start = time.perf_counter()

            answer, generated_tokens = generate(
                question=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed
            )

            generation_seconds = (
                time.perf_counter()
                - generation_start
            )

        total_seconds = (
            time.perf_counter()
            - start_time
        )

        tokens_per_second = (
            generated_tokens /
            generation_seconds
            if generation_seconds > 0
            else 0.0
        )

        return jsonify({

            "response": answer,

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
                True
        })

    except Exception as e:

        print(
            "Generation error:",
            repr(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        threaded=False
    )
