import logging
import os
from typing import List

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.agents.agent import AgentExecutor
from langgraph.graph import StateGraph, END
import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union
from fastapi import FastAPI, Request, Form, Response, Depends
from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

import os


app = FastAPI()

import os
from dotenv import load_dotenv

load_dotenv()  # will search for .env file in local folder and load variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

logging.basicConfig(level=logging.INFO)

llm = ChatOpenAI(model="gpt-4", temperature=0.5)
tavily_tool = TavilySearchResults(max_results=5)
tools = [tavily_tool]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant that uses web search to find information on the topic given below",
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# researchstate dictionary


class ResearchState(dict):
    topic: str
    plan: str
    links: List[str]
    contents: str
    summary: str
    approved: bool


# method for research planner node 1
def research_planner(state: ResearchState):
    topic = state["topic"]
    plan = f"Research plan for '{topic}': 1) Write background 2) Research objectives"
    logging.info("Plan created.")
    return {**state, "plan": plan}


# method for infomation gatherer planner node 2


def information_gatherer(state: ResearchState):
    topic = state["topic"]
    logging.info(f"Gathering info on: {topic}")
    result = agent_executor.invoke({"input": f"Find recent web sources about: {topic}"})
    links = []
    if isinstance(result, dict) and "output" in result:
        output = result["output"]
        links = output.split("\n")[:3]  # naive extraction
    logging.info(f"Links gathered: {links}")
    return {**state, "links": links}


# method for content analyzer node 3
def content_analyzer(state: ResearchState):
    if not state.get("links"):
        raise ValueError("No links to analyze.")
    contents = "\n".join([f"Summary of {link}" for link in state["links"]])
    logging.info("Content analyzed.")
    return {**state, "contents": contents}


# method for report generator node 4
def report_generator(state: ResearchState):
    summary = f"Report on '{state['topic']}':\n{state['contents']}"
    logging.info("Report generated.")
    return {**state, "summary": summary}


def human_review(state: ResearchState):
    print("Do you approve this report? (yes/no)")
    approval = input()
    approved = approval.lower().strip() == "yes"
    return {**state, "approved": approved}


def error_recovery(state: ResearchState):
    logging.warning("Recovering from error...")
    return {**state, "links": [], "contents": "Fallback content."}


def main():
    builder = StateGraph(ResearchState)

    builder.add_node("Planner", RunnableLambda(research_planner))
    builder.add_node("Gatherer", RunnableLambda(information_gatherer))
    builder.add_node("Analyzer", RunnableLambda(content_analyzer))
    builder.add_node("Reporter", RunnableLambda(report_generator))
    builder.add_node("Review", RunnableLambda(human_review))
    builder.add_node("Recovery", RunnableLambda(error_recovery))

    builder.set_entry_point("Planner")

    builder.add_edge("Planner", "Gatherer")
    builder.add_edge("Gatherer", "Analyzer")

    builder.add_conditional_edges(
        "Analyzer",
        lambda state: "Recovery" if not state.get("contents") else "Reporter",
        {"Recovery": "Reporter", "Reporter": "Review"},
    )

    builder.add_conditional_edges(
        "Review",
        lambda state: END if state.get("approved") else "Gatherer",
        {END: END, "Gatherer": "Analyzer"},
    )

    graph = builder.compile()

    initial_state = ResearchState(topic="AI in modern world")
    final_state = graph.invoke(initial_state)

    print("\n📄 Final Report:")
    print(final_state)


if __name__ == "__main__":
    main()
