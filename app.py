from flask import Flask,jsonify,request

#importing classifier file
from classifier import getPrediction

app=Flask(__name__)
@app.route("/predictDigit",methods=["POST"])

def predictData():
    image=request.files.get("digit")
    prediction=getPrediction(image)
    return jsonify({
        "prediction":prediction
    },200)

if (__name__=="__main__"):
    app.run(debug=True)