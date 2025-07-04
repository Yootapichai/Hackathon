import requests

from agent.query_tools.notification import warehouse_capacity

def forecast_agent():
    df = warehouse_capacity()
    payload = {
        "user_query":df.to_json(),
    }
    response = requests.post("https://chanoot001.app.n8n.cloud/webhook/forecast", json=payload)

    print(response.text)
    # Return the actual response content
    return response.json()[0]['output']

if __name__ == "__main__":
    forecast_agent()