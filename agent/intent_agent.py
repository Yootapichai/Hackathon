import requests

from agent.general_agent import general_agent

def router_agent(question:str):
    payload = {
        "user_query": question,
        "chat_history": []
    }
    response = requests.post("https://chanoot001.app.n8n.cloud/webhook/0698e909-a440-4648-87c7-29d27348caf5", json=payload)
    print(response.json())
    question, intent = response.json()[0]['output'].split("Intent:",1)
    _, question = question.split("Question: ",1)
    print (f"Question: {question.strip()}")
    print (f"Intent: {intent.strip()}")
    return question.strip(), intent.strip()
  
def handle_router(intent:str):
    
    if 'inventory_question' in intent:
        pass #TODO: connect to Text2SQL agent
    elif 'general_question' in intent:
        return general_agent(intent)
    elif 'knowledge_question' in intent:
        pass #TODO: connect to RAG Agent
    else:
        return "Sorry, I couldn't understand your question. Please try again with a different query." 

router_agent("how are you doing today?")
