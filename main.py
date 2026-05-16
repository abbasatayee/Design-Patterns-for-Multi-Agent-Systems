import os

from dotenv import load_dotenv

from agents import build_orchestrator, build_router

load_dotenv()


def _print_reply(result: dict) -> None:
    message = result["messages"][-1]
    if message.content:
        print(message.content)
        return
    for block in getattr(message, "content_blocks", []) or []:
        if block.get("type") == "text":
            print(block.get("text", ""))


def main():
    model = os.getenv("OLLAMA_MODEL", "ollama:minimax-m2.5:cloud")
    pattern = os.getenv("PATTERN", "router").lower()
    user_message = os.getenv(
        "USER_MESSAGE",
        "What's the weather in Berlin and what is the crypto price of bitcoin in euros?",
    )

    if pattern == "orchestrator":
        print(f"Pattern: orchestrator\n")
        orchestrator = build_orchestrator(model)
        result = orchestrator.invoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )
        _print_reply(result)
        return

    print(f"Pattern: router\n")
    router = build_router(model)
    result = router.invoke({"query": user_message})
    print("Routed to:", [c["source"] for c in result.get("classifications", [])])
    print()
    print(result["final_answer"])


if __name__ == "__main__":
    main()
