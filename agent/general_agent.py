import requests

def general_agent(question):
    payload = {
        "user_query": question,
    }
    response = requests.post("https://chanoot001.app.n8n.cloud/webhook/general-001", json=payload)
    print(response.json())
    # print(response.json()[0]['output'])
    return response.json()[0]['output']

# general_agent("how can did warehouse management system help in inventory management?")