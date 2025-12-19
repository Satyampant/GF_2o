from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import get_router_chain
from ai_companion.settings import settings

async def router_node(state: AICompanionState) -> str:
    chain = get_router_chain()
    result = await chain.ainvoke(messages=state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :])
    return {"workflow": result.response_type}