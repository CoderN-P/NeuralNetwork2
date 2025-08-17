from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/api/predict', methods=['POST'])
def echo():
    """
    Echoes back the JSON data sent in the POST request.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    return jsonify(data), 200


@app.route('/api/health', methods=['GET'])
def health():
    """
    Returns a simple health check response.
    """
    return jsonify({"status": "healthy"}), 200