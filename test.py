import requests

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
  "Authorization": "Bearer ",
  "Content-Type": "application/json"
}

data = {
  "model": "openai/gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "Say hello in one sentence"}
  ]
}

response = requests.post(url, headers=headers, json=data)

print(response.status_code)
print(response.json())