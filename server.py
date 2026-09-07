import os
import json
import re
import threading

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
DEFAULT_MAX_NEW_TOKENS = 180

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

        # The original TinyGPT tokenizer uses a similar
        # word/punctuation/whitespace pattern.
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

            # Exact special token
            if piece in self.token_to_id:
                result.append(self.token_to_id[piece])
                continue

            # Words are lowercased when possible
            if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", piece):
                lower = piece.lower()

                if lower in self.token_to_id:
                    result.append(self.token_to_id[lower])
                    continue

            # Direct token
            if piece in self.token_to_id:
                result.append(self.token_to_id[piece])
                continue

            # Character fallback
            for char in piece:
                if char in self.token_to_id:
                    result.append(self.token_to_id[char])
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

        tokens = self.meta["tokens"]

        self.tokenizer = HybridTokenizer(tokens)

        # ----------------------------------------------------
        # Load every saved parameter
        # ----------------------------------------------------

        self.params = {}

        for key in data.files:
            if key.startswith("param_"):
                name = key[len("param_"):]
                self.params[name] = data[key].astype(
                    np.float32,
                    copy=False
                )

        # ----------------------------------------------------
        # Special token IDs
        # ----------------------------------------------------

        self.start_id = self.tokenizer.token_to_id.get(
            "<START>"
        )

        self.user_id = self.tokenizer.token_to_id.get(
            "<USER>"
        )

        self.assistant_id = self.tokenizer.token_to_id.get(
            "<ASSISTANT>"
        )

        self.end_id = self.tokenizer.token_to_id.get(
            "<END>"
        )

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        self.causal_mask = (
            np.triu(
                np.ones(
                    (
                        self.block_size,
                        self.block_size
                    ),
                    dtype=bool
                ),
                k=1
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
            "Context:",
            self.block_size
        )
        print(
            "Parameters:",
            len(self.params)
        )

    # --------------------------------------------------------
    # RMSNorm
    # --------------------------------------------------------

    def rms_norm(self, x, weight, eps=1e-5):
        mean_square = np.mean(
            x * x,
            axis=-1,
            keepdims=True
        )

        x = x / np.sqrt(
            mean_square + eps
        )

        return x * weight

    # --------------------------------------------------------
    # GELU
    # --------------------------------------------------------

    def gelu(self, x):
        return 0.5 * x * (
            1.0 +
            np.tanh(
                np.sqrt(2.0 / np.pi)
                * (
                    x +
                    0.044715 * x * x * x
                )
            )
        )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(self, token_ids):
        token_ids = np.asarray(
            token_ids,
            dtype=np.int64
        )

        T = len(token_ids)

        if T > self.block_size:
            token_ids = token_ids[-self.block_size:]
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

        # ----------------------------------------------------
        # Transformer layers
        # ----------------------------------------------------

        for layer in range(self.num_layers):

            prefix = f"layer{layer}_"

            # -------------------------------
            # Attention normalization
            # -------------------------------

            norm1 = self.params[
                prefix + "norm1"
            ]

            h = self.rms_norm(
                x,
                norm1
            )

            Wq = self.params[
                prefix + "Wq"
            ]

            Wk = self.params[
                prefix + "Wk"
            ]

            Wv = self.params[
                prefix + "Wv"
            ]

            Wo = self.params[
                prefix + "Wo"
            ]

            Q = h @ Wq
            K = h @ Wk
            V = h @ Wv

            head_dim = C // self.num_heads

            Q = Q.reshape(
                T,
                self.num_heads,
                head_dim
            ).transpose(1, 0, 2)

            K = K.reshape(
                T,
                self.num_heads,
                head_dim
            ).transpose(1, 0, 2)

            V = V.reshape(
                T,
                self.num_heads,
                head_dim
            ).transpose(1, 0, 2)

            # Attention scores
            scores = (
                Q @ K.transpose(0, 2, 1)
            ) / np.sqrt(head_dim)

            # IMPORTANT:
            # This intentionally follows the mask behavior
            # used by the original TinyGPT v8 implementation
            # so that inference remains compatible with the
            # model that was actually trained.
            mask = self.causal_mask[
                :T,
                :T
            ]

            scores = np.where(
                mask[None, :, :],
                -1e9,
                scores
            )

            # Stable softmax
            scores = (
                scores
                - np.max(
                    scores,
                    axis=-1,
                    keepdims=True
                )
            )

            weights = np.exp(scores)

            weights /= np.sum(
                weights,
                axis=-1,
                keepdims=True
            ) + 1e-9

            attention = (
                weights @ V
            )

            attention = attention.transpose(
                1,
                0,
                2
            ).reshape(
                T,
                C
            )

            x = x + (
                attention @ Wo
            )

            # -------------------------------
            # Feed-forward
            # -------------------------------

            norm2 = self.params[
                prefix + "norm2"
            ]

            h = self.rms_norm(
                x,
                norm2
            )

            W1 = self.params[
                prefix + "W1"
            ]

            b1 = self.params[
                prefix + "b1"
            ]

            W2 = self.params[
                prefix + "W2"
            ]

            b2 = self.params[
                prefix + "b2"
            ]

            h = h @ W1 + b1
            h = self.gelu(h)
            h = h @ W2 + b2

            x = x + h

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.rms_norm(
            x,
            self.params["final_norm"]
        )

        # Weight tying:
        # output projection uses token_embedding.T
        logits = (
            x @ token_embedding.T
        )

        return logits


# ============================================================
# Load model
# ============================================================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}\n"
        "Make sure tiny_gpt_pc.npz is in the repository."
    )

print(
    f"Loading TinyGPT model from {MODEL_FILE}..."
)

model_data = np.load(
    MODEL_FILE,
    allow_pickle=True
)

model = TinyGPT(model_data)


# ============================================================
# Generation
# ============================================================

def softmax(logits):
    logits = logits - np.max(logits)

    probabilities = np.exp(logits)

    total = np.sum(probabilities)

    if total <= 0 or not np.isfinite(total):
        return np.ones_like(
            probabilities
        ) / len(probabilities)

    return probabilities / total


def generate(
    prompt,
    temperature=0.70,
    top_k=30,
    top_p=0.90,
    repetition_penalty=1.05,
    max_new_tokens=180,
    seed=None
):
    tokenizer = model.tokenizer

    # Same general prompt format used during TinyGPT training
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
        return ""

    rng = np.random.default_rng(seed)

    generated = list(token_ids)

    for _ in range(max_new_tokens):

        context = generated[
            -model.block_size:
        ]

        logits = model.forward(
            context
        )

        next_logits = logits[-1].copy()

        # ----------------------------------------
        # Repetition penalty
        # ----------------------------------------

        if repetition_penalty > 1.0:
            recent = set(
                generated[-64:]
            )

            for token_id in recent:
                if 0 <= token_id < len(next_logits):

                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= (
                            repetition_penalty
                        )
                    else:
                        next_logits[token_id] *= (
                            repetition_penalty
                        )

        # ----------------------------------------
        # Temperature
        # ----------------------------------------

        temperature = max(
            float(temperature),
            1e-5
        )

        next_logits /= temperature

        # ----------------------------------------
        # Top-K
        # ----------------------------------------

        if top_k > 0 and top_k < len(next_logits):

            indices = np.argpartition(
                next_logits,
                -top_k
            )[-top_k:]

            filtered = np.full_like(
                next_logits,
                -np.inf
            )

            filtered[indices] = (
                next_logits[indices]
            )

            next_logits = filtered

        # ----------------------------------------
        # Convert to probabilities
        # ----------------------------------------

        probabilities = softmax(
            next_logits
        )

        # ----------------------------------------
        # Top-P
        # ----------------------------------------

        if top_p < 1.0:

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

            # Always keep at least one token
            if len(remove) > 0:
                remove[0] = False

            probabilities[
                sorted_indices[remove]
            ] = 0.0

            total = probabilities.sum()

            if total > 0:
                probabilities /= total

        # ----------------------------------------
        # Sample
        # ----------------------------------------

        next_token = int(
            rng.choice(
                len(probabilities),
                p=probabilities
            )
        )

        generated.append(
            next_token
        )

        # ----------------------------------------
        # Stop at END
        # ----------------------------------------

        if (
            model.end_id is not None
            and next_token == model.end_id
        ):
            break

    # Decode only the generated answer
    answer = tokenizer.decode(
        generated
    )

    # Remove everything before ASSISTANT
    marker = "<ASSISTANT>"

    if marker in answer:
        answer = answer.split(
            marker,
            1
        )[1]

    # Don't let generated USER sections leak out
    if "<USER>" in answer:
        answer = answer.split(
            "<USER>",
            1
        )[0]

    if "<END>" in answer:
        answer = answer.split(
            "<END>",
            1
        )[0]

    return answer.strip()


# ============================================================
# Flask API
# ============================================================

app = Flask(__name__)


def check_api_key():
    expected = os.environ.get(
        "API_KEY"
    )

    # API key is optional.
    # If no API_KEY is configured on Render,
    # the API is public.
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
            len("Bearer "):
        ]

        return supplied == expected

    supplied = request.headers.get(
        "X-API-Key",
        ""
    )

    return supplied == expected


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "TinyGPT API",
        "status": "online",
        "model_loaded": True,
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /info": "Model information",
            "POST /generate": "Generate a TinyGPT response"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True
    })


@app.route("/info", methods=["GET"])
def info():

    metadata = {}

    for key, value in model.meta.items():

        # Convert NumPy values to normal Python values
        if isinstance(value, np.generic):
            value = value.item()

        metadata[key] = value

    return jsonify({
        "model": metadata,
        "server": {
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "default_max_new_tokens": DEFAULT_MAX_NEW_TOKENS
        }
    })


@app.route("/generate", methods=["POST"])
def generate_endpoint():

    if not check_api_key():
        return jsonify({
            "error": "Unauthorized"
        }), 401

    data = request.get_json(
        silent=True
    )

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be JSON."
        }), 400

    prompt = data.get(
        "prompt"
    )

    if not isinstance(prompt, str):
        return jsonify({
            "error": "'prompt' must be a string."
        }), 400

    prompt = prompt.strip()

    if not prompt:
        return jsonify({
            "error": "Prompt cannot be empty."
        }), 400

    if len(prompt) > MAX_PROMPT_CHARS:
        return jsonify({
            "error": (
                f"Prompt is too long. "
                f"Maximum is {MAX_PROMPT_CHARS} characters."
            )
        }), 400

    try:
        temperature = float(
            data.get(
                "temperature",
                0.70
            )
        )

        top_k = int(
            data.get(
                "top_k",
                30
            )
        )

        top_p = float(
            data.get(
                "top_p",
                0.90
            )
        )

        repetition_penalty = float(
            data.get(
                "repetition_penalty",
                1.05
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
            seed = int(seed)

        # Safety/validity limits
        temperature = min(
            max(temperature, 0.05),
            3.0
        )

        top_k = min(
            max(top_k, 0),
            model.vocab_size
        )

        top_p = min(
            max(top_p, 0.05),
            1.0
        )

        repetition_penalty = min(
            max(repetition_penalty, 1.0),
            2.0
        )

        max_new_tokens = min(
            max(max_new_tokens, 1),
            180
        )

    except (ValueError, TypeError):
        return jsonify({
            "error": "Invalid generation parameters."
        }), 400

    try:

        # Serialize generation requests.
        # This keeps memory use predictable and avoids
        # simultaneous NumPy generations fighting over RNG/
        # CPU resources.
        with MODEL_LOCK:

            answer = generate(
                prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                seed=seed
            )

        return jsonify({
            "prompt": prompt,
            "response": answer
        })

    except Exception as exc:

        print(
            "Generation error:",
            repr(exc)
        )

        return jsonify({
            "error": "Generation failed.",
            "details": str(exc)
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
        port=port
              )
