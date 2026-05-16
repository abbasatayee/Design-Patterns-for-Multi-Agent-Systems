import os

from dotenv import load_dotenv

from agents import build_router

load_dotenv()


def main():
    model = os.getenv("OLLAMA_MODEL", "ollama:minimax-m2.5:cloud")
    router = build_router(model)

    user_message = os.getenv(
        "USER_MESSAGE",
        "What's the weather in Berlin and what is the crypto price of bitcoin in euros?",
    )

    result = router.invoke({"query": user_message})

    print("Routed to:", [c["source"] for c in result.get("classifications", [])])
    print()
    print(result["final_answer"])


if __name__ == "__main__":
    main()
