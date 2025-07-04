import requests

from agent.query_tools.notification import warehouse_capacity

def forecaset_agent():
    df = warehouse_capacity()
    payload = {
        "user_query":df.to_json(),
    }
    response = requests.post("https://chanoot001.app.n8n.cloud/webhook/forecast", json=payload)

    print(response.text)
    # print(response.json()[0]['output'])
    # return response.json()[0]['output']

forecaset_agent()