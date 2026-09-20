import requests
import time

# ============================================================
# TinyGPT Remote Chat - Pydroid 3
# ============================================================

SERVER_URL = "https://tinygpt-ppjt.onrender.com"

GENERATE_URL = SERVER_URL + "/generate"
HEALTH_URL = SERVER_URL + "/health"


# ============================================================
# RECOMMENDED GENERATION SETTINGS
# ============================================================

TEMPERATURE = 0.75
TOP_K = 30
TOP_P = 0.90
REPETITION_PENALTY = 1.05

MAX_NEW_TOKENS = 60

REQUEST_TIMEOUT = 180


# ============================================================
# ASK TINYGPT
# ============================================================

def ask_tinygpt(message):

    message = message.strip()

    if not message:
        return ""

    payload = {
        "prompt": message,

        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,

        "repetition_penalty":
            REPETITION_PENALTY,

        "max_new_tokens":
            MAX_NEW_TOKENS
    }

    try:

        start = time.perf_counter()

        response = requests.post(
            GENERATE_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        elapsed = time.perf_counter() - start

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "response",
            ""
        )

        print()
        print("TinyGPT:")
        print(answer)

        print()
        print(
            f"[HTTP {elapsed:.2f}s]"
        )

        if "generated_tokens" in data:

            print(
                f"[Generated "
                f"{data['generated_tokens']} tokens]"
            )

        if "generation_seconds" in data:

            print(
                f"[Generation time: "
                f"{data['generation_seconds']:.2f}s]"
            )

        if "tokens_per_second" in data:

            print(
                f"[Generation speed: "
                f"{data['tokens_per_second']:.2f} tokens/sec]"
            )

        return answer

    except requests.exceptions.Timeout:

        print()
        print("Request timed out.")
        print(
            "The Render server may be starting up."
        )

        return ""

    except requests.exceptions.RequestException as e:

        print()
        print("Request failed:")
        print(e)

        return ""

    except Exception as e:

        print()
        print("Unexpected error:")
        print(e)

        return ""


# ============================================================
# PING SERVER
# ============================================================

def ping_server():

    print()
    print("Checking server...")

    try:

        start = time.perf_counter()

        response = requests.get(
            HEALTH_URL,
            timeout=30
        )

        elapsed = time.perf_counter() - start

        print(
            f"HTTP {response.status_code}"
        )

        print(
            f"Response time: "
            f"{elapsed:.2f}s"
        )

        try:
            print(
                response.json()
            )
        except:
            print(
                response.text
            )

    except Exception as e:

        print()
        print("Ping failed:")
        print(e)


# ============================================================
# MAIN CHAT
# ============================================================

print("=" * 50)
print("TinyGPT Remote Chat")
print("=" * 50)

print()
print("Server:")
print(SERVER_URL)

print()
print("Commands:")
print("/quit   - exit")
print("/ping   - check server")
print("/clear  - clear screen")

print()
print(
    "TinyGPT v8 uses one-question-at-a-time"
)
print(
    "generation. Previous messages are NOT"
)
print(
    "sent back to the model."
)

print("=" * 50)


while True:

    try:

        user_input = input("\nYou: ")

    except KeyboardInterrupt:

        print()
        print("Exiting...")

        break

    except EOFError:

        break

    user_input = user_input.strip()

    if not user_input:
        continue


    # --------------------------------------------------------
    # COMMANDS
    # --------------------------------------------------------

    if user_input.lower() == "/quit":

        print("Goodbye!")

        break


    if user_input.lower() == "/ping":

        ping_server()

        continue


    if user_input.lower() == "/clear":

        print("\033[2J\033[H")

        continue


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    ask_tinygpt(
        user_input
    )
