import asyncio

from claude_code_sdk import ClaudeCodeOptions, ClaudeSDKClient


async def explain_failure(log_text: str) -> str:
    """
    Programmatisk fejlanalyse (read-only).
    Returnerer kort forklaring + forslag (ingen skriv/exec).
    """
    async with ClaudeSDKClient(
        options=ClaudeCodeOptions(
            system_prompt=(
                "Du er senior Python/ML-dev i et AI trading bot-projekt. "
                "Forklar fejl og foreslå MINIMAL patch + regression-test."
            ),
            allowed_tools=["Read"],  # kun læsning
            permission_mode="plan",
            max_turns=2,
        )
    ) as client:
        await client.query(f"Analyser testlog og foreslå fix:\n{log_text[:8000]}")
        chunks = []
        async for msg in client.receive_response():
            if getattr(msg, "content", None):
                for b in msg.content:
                    if getattr(b, "text", None):
                        chunks.append(b.text)
        return "".join(chunks)


if __name__ == "__main__":
    import sys

    text = sys.stdin.read() if not sys.stdin.isatty() else "Ingen log indlæst."
    out = asyncio.run(explain_failure(text))
    print(out)
