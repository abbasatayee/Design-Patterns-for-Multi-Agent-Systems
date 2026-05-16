"""Router pattern: classify → parallel specialist agents → synthesize."""

import json
import operator
import re
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from tools import FINANCE_TOOLS, FUN_TOOLS, GEO_TOOLS, KNOWLEDGE_TOOLS

AgentName = Literal["geo", "knowledge", "finance", "fun"]

GEO_PROMPT = (
    "You are a geography & travel specialist. Use your tools for weather, "
    "timezones, ISS tracking, and country information. Be factual and concise."
)
KNOWLEDGE_PROMPT = (
    "You are a research specialist. Use Wikipedia, the dictionary, and "
    "fact APIs. Cite what you found; do not make things up."
)
FINANCE_PROMPT = (
    "You are a finance specialist. Use tools for live crypto prices and "
    "currency conversion. Always include numbers from tool output."
)
FUN_PROMPT = (
    "You are a fun & utilities specialist. Use tools for jokes, activities, "
    "space photos, dice, passwords, and date math. Keep it light."
)

CLASSIFIER_PROMPT = """Analyze the user query and pick which specialists to consult.
For each relevant specialist, write a focused sub-question optimized for that domain.

Specialists (only include relevant ones):
- geo: weather, timezones, ISS location, country facts
- knowledge: Wikipedia, dictionary definitions, random facts
- finance: cryptocurrency prices, currency conversion
- fun: jokes, activity ideas, NASA photos, dice, passwords, date countdowns

Respond with ONLY valid JSON (no markdown), shape:
{"classifications": [{"source": "geo", "query": "..."}, {"source": "finance", "query": "..."}]}

Omit specialists that do not apply."""

SYNTHESIZER_PROMPT = """Synthesize specialist results into one clear answer for the user.
- Combine sources without repeating yourself
- Keep numbers and facts from the specialists
- Be friendly and concise"""


class AgentInput(TypedDict):
    query: str


class AgentOutput(TypedDict):
    source: str
    result: str


class Classification(TypedDict):
    source: AgentName
    query: str


class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str


class ClassificationItem(BaseModel):
    source: AgentName = Field(description="Specialist to invoke")
    query: str = Field(description="Targeted sub-question for that specialist")


class ClassificationResult(BaseModel):
    classifications: list[ClassificationItem] = Field(
        description="Specialists to invoke with tailored sub-questions"
    )


def _extract_reply(result: dict) -> str:
    message = result["messages"][-1]
    if message.content:
        return message.content
    blocks = getattr(message, "content_blocks", None) or []
    return " ".join(
        block.get("text", "") for block in blocks if block.get("type") == "text"
    )


def _parse_classifications(text: str, original_query: str) -> list[Classification]:
    """Parse classifier JSON; fall back to keyword routing for models without structured output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            items = data.get("classifications", data if isinstance(data, list) else [])
            valid: list[Classification] = []
            for item in items:
                source = item.get("source")
                if source in ("geo", "knowledge", "finance", "fun"):
                    valid.append(
                        {"source": source, "query": item.get("query", original_query)}
                    )
            if valid:
                return valid
        except json.JSONDecodeError:
            pass

    q = original_query.lower()
    routes: list[Classification] = []
    rules: list[tuple[AgentName, tuple[str, ...]]] = [
        ("geo", ("weather", "temperature", "iss", "country", "capital", "timezone", "where is")),
        (
            "knowledge",
            ("wikipedia", "define", "definition", "what is", "who is", "fact about", "meaning of"),
        ),
        ("finance", ("crypto", "bitcoin", "ethereum", "price", "convert", "currency", "usd", "eur")),
        ("fun", ("joke", "bored", "activity", "nasa", "dice", "password", "days until", "roll")),
    ]
    for source, keywords in rules:
        if any(k in q for k in keywords):
            routes.append({"source": source, "query": original_query})
    return routes or [{"source": "geo", "query": original_query}]


def _build_specialists(model: str) -> dict[AgentName, object]:
    return {
        "geo": create_agent(model=model, tools=GEO_TOOLS, system_prompt=GEO_PROMPT),
        "knowledge": create_agent(
            model=model, tools=KNOWLEDGE_TOOLS, system_prompt=KNOWLEDGE_PROMPT
        ),
        "finance": create_agent(
            model=model, tools=FINANCE_TOOLS, system_prompt=FINANCE_PROMPT
        ),
        "fun": create_agent(model=model, tools=FUN_TOOLS, system_prompt=FUN_PROMPT),
    }


def build_router(model: str):
    """LangGraph router: classify query, fan-out via Send, synthesize results."""
    router_llm = init_chat_model(model)
    specialists = _build_specialists(model)

    def classify_query(state: RouterState) -> dict:
        try:
            structured = router_llm.with_structured_output(ClassificationResult)
            result = structured.invoke(
                [
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": state["query"]},
                ]
            )
            classifications = [
                {"source": c.source, "query": c.query} for c in result.classifications
            ]
        except Exception:
            response = router_llm.invoke(
                [
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": state["query"]},
                ]
            )
            classifications = _parse_classifications(
                response.content or "", state["query"]
            )
        return {"classifications": classifications}

    def route_to_agents(state: RouterState) -> list[Send]:
        if not state["classifications"]:
            return [Send("geo", {"query": state["query"]})]
        return [
            Send(c["source"], {"query": c["query"]}) for c in state["classifications"]
        ]

    def _query_agent(source: AgentName, state: AgentInput) -> dict:
        result = specialists[source].invoke(
            {"messages": [{"role": "user", "content": state["query"]}]}
        )
        return {
            "results": [{"source": source, "result": _extract_reply(result)}]
        }

    def query_geo(state: AgentInput) -> dict:
        return _query_agent("geo", state)

    def query_knowledge(state: AgentInput) -> dict:
        return _query_agent("knowledge", state)

    def query_finance(state: AgentInput) -> dict:
        return _query_agent("finance", state)

    def query_fun(state: AgentInput) -> dict:
        return _query_agent("fun", state)

    def synthesize_results(state: RouterState) -> dict:
        if not state["results"]:
            return {"final_answer": "No specialist returned results."}

        formatted = [
            f"**{r['source'].title()} specialist:**\n{r['result']}"
            for r in state["results"]
        ]
        response = router_llm.invoke(
            [
                {"role": "system", "content": SYNTHESIZER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Original question: {state['query']}\n\n"
                        + "\n\n".join(formatted)
                    ),
                },
            ]
        )
        return {"final_answer": response.content}

    workflow = (
        StateGraph(RouterState)
        .add_node("classify", classify_query)
        .add_node("geo", query_geo)
        .add_node("knowledge", query_knowledge)
        .add_node("finance", query_finance)
        .add_node("fun", query_fun)
        .add_node("synthesize", synthesize_results)
        .add_edge(START, "classify")
        .add_conditional_edges(
            "classify", route_to_agents, ["geo", "knowledge", "finance", "fun"]
        )
        .add_edge("geo", "synthesize")
        .add_edge("knowledge", "synthesize")
        .add_edge("finance", "synthesize")
        .add_edge("fun", "synthesize")
        .add_edge("synthesize", END)
        .compile()
    )
    return workflow
