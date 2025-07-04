import requests
import pandas as pd

def analyst_thoughput(question:str = ""):
    payload = {
        "query": question,
    }
    response =requests.post("https://chanoot001.app.n8n.cloud/webhook/monthly_thoughput", json=payload)
    df = pd.DataFrame(response.json())
    print(df)
    return response.json()

if __name__ == "__main__":
    analyst_thoughput()