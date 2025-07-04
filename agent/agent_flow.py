from utils.supply_chain_agent import SupplyChainAgent
from agent.general_agent import general_agent
from agent.knowledge_agent import knowledge_agent
from agent.intent_agent import router_agent

def handle_router(intent:str, question:str, chat_history:list = None, supply_chain_agent=None, thread_id=None):
    print(f"Intent on handle_router: {intent}")
    if 'inventory_question' in intent:
        if supply_chain_agent is None:
            agent = SupplyChainAgent()
            response = agent.process_query(question)
        else:
            response = supply_chain_agent.process_query(question, thread_id=thread_id)
        return response
    elif 'general_question' in intent:
        return general_agent(question)
    elif 'knowledge_question' in intent:
        return knowledge_agent(question, chat_history=[])
    else:
        return "Sorry, I couldn't understand your question. Please try again with a different query." 

if __name__ == "__main__":

    question = "what is Butyl"
    result = handle_router(router_agent(question), question, chat_history=[])

    print(result)