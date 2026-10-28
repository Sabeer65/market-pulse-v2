import ollama

from marketpulse.config import get_settings


def installed_model_names(client: ollama.Client) -> set[str]:
    """Return the names of all models the server has installed."""
    return {m.model for m in client.list().models}


def is_installed(wanted: str, installed: set[str]) -> bool:
    """Match 'nomic-embed-text' against 'nomic-embed-text:latest'."""
    if ":" not in wanted:
        wanted = f"{wanted}:latest"
    return wanted in installed


def check_embedding(client: ollama.Client, model: str, expected_dim: int = 768) -> bool:
    """Embed one sentence and verify the vector has the expected length."""
    response = client.embed(model=model, input="MarketPulse analyzes market trends.")
    vector = response.embeddings[0]
    print(f"  embedding: got {len(vector)} numbers, first three: {vector[:3]}")
    return len(vector) == expected_dim


def check_generation(client: ollama.Client, model: str, num_ctx: int) -> bool:
    """Ask one trivial question and verify a non-empty, clean answer comes back."""
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly one word: ready"}],
        think=True,
        options={"num_ctx": num_ctx},
    )
    answer = response.message.content.strip()
    tok_s = response.eval_count / (response.eval_duration / 1e9)
    print(f"  generation: {answer!r} at {tok_s:.0f} tok/s")
    return bool(answer) and "</think>" not in answer


def main() -> None:
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_host)

    try:
        installed = installed_model_names(client)
    except Exception as exc:
        print(f"FAIL: cannot reach Ollama at {settings.ollama_host}: {type(exc).__name__}")
        return

    print(f"Server reachable at {settings.ollama_host}")
    for wanted in (settings.llm_model, settings.embed_model):
        status = "OK" if is_installed(wanted, installed) else "MISSING"
        print(f"  {status}: {wanted}")

    print("Embedding check:")
    ok = check_embedding(client, settings.embed_model)
    print(f"  {'PASS' if ok else 'FAIL'}")

    print("Generation check:")
    ok = check_generation(client, settings.llm_model, settings.num_ctx)
    print(f"  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()