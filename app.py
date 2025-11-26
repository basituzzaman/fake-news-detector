import requests

def get_model_prediction(text):
    API_URL = "https://api-inference.huggingface.co/models/Pulk17/Fake-News-Detection"
    headers = {"Authorization": "Bearerhf_mgrDOWteWZuaFRapQLSYumYbyWarHQjjSV"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        label = result[0]["label"]
        score = result[0]["score"]
        return label, score
    else:
        return "Unknown", 0.0
