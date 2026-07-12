"""
wealthdesk/nodes.py
-------------------
Node functions for the WealthDesk graph.

Each node is a plain Python function:
  - Input : the full WealthDeskState (read-only)
  - Output: a dict containing ONLY the keys this node changed
             (LangGraph merges it into the state automatically)
"""
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .config import SYSTEM_PROMPT
from .state import WealthDeskState
from .tools import llm


# ---------------------------------------------------------------------------
# TODO 4 of 5 -- respond node
# ---------------------------------------------------------------------------
# Implement the respond() function so it:
#
#   1. Builds a messages list:
#        messages = [
#            SystemMessage(content=SYSTEM_PROMPT),
#            HumanMessage(content=state["customer_message"]),
#        ]
#
#   2. Calls the LLM inside a try / except block:
#        result = llm.invoke(messages)
#
#   3. On success  → return {"response": result.content}
#      On exception → print the error with a [WealthDesk] prefix
#                      and return a safe fallback string so the
#                      agent never crashes mid-conversation.
#
# ---------------------------------------------------------------------------

def respond(state: WealthDeskState) -> dict:
    messages= [
            SystemMessage(content=SYSTEM_PROMPT)
        ]
    history= state.get('history',[])
    for turn in history:
        if(turn["role"]=="user"):
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result=llm.invoke(messages)
        response_text=result.content
    except Exception as e:
        print("LLM error:" ,e)
        return {"response":"Agent is unavailable"}
    new_history=history+[{"role":"user","content":state["customer_message"]},
    {"role":"assistant","content":response_text}]
    return {"response":response_text,"history":new_history}

