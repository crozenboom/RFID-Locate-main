from flask import Flask, request # type: ignore

app = Flask(__name__)

@app.route('/rfid', methods=['POST'])
def receive_data():
    data = request.get_data(as_text=True)
    print("Received data:\n", data)
    return "OK", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050)