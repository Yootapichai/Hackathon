from .supply_chain_agent import SupplyChainAgent
from agent.general_agent import general_agent
from agent.knowledge_agent import knowledge_agent
from agent.intent_agent import router_agent

def handle_router(intent:str, question:str, chat_history:list = None):
    
    if 'inventory_question' in intent:
      agent = SupplyChainAgent()
    elif 'general_question' in intent:
        return general_agent(question)
    elif 'knowledge_question' in intent:
        return knowledge_agent(question, chat_history=[])
    else:
        return "Sorry, I couldn't understand your question. Please try again with a different query." 


question = "what is Butyl"
result = handle_router(router_agent(question), question, chat_history=[])

print(result)