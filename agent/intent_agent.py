import requests

def router_agent(question:str):
    payload = {
        "user_query": question,
        "chat_history": []
    }
    response = requests.post("https://chanoot001.app.n8n.cloud/webhook/0698e909-a440-4648-87c7-29d27348caf5", json=payload)
    # print(f"router_agent Question: {question}")
    # print(f"From router_agent:")
    # print(response.json())
    return response.json()[0]['output']