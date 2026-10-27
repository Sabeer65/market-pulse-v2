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


if __name__ == "__main__":
    main()