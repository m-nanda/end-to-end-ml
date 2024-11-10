from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello():
  """Basic Flask with simple routing"""
  return "<h1>Hello, Telkom University Surabaya!</h1>"

@app.route("/v1/hello/<name>", methods=['GET'])
def hello_v1(name: str) -> str: 
  """
  Example using a route parameter.

  Returns:
    str: Simple greeting message.
  """
  return f"Hello, {name}!"

@app.route("/v2/hello", methods=['GET', 'POST'])
def hello_v2() -> str:
  """
  Receives a JSON request body with a 'name' field.

  Returns:
    str: Simple greeting message.
  """
  data = request.get_json()
  name = data.get('name')
  return f"Hello, {name}!"

if __name__ == "__main__":
  app.run(host="0.0.0.0")