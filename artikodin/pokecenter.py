from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json

    print(f'{data}')

    # For demonstration, just returning the received data
    output = {'Hello': data}

    # Return the prediction as JSON response
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)