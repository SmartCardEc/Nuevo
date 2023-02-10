from flask import Flask, render_template, request
import pandas as pd

import os
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

app = Flask(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./GOOGLE_APPLICATION_CREDENTIALS.json"

def predict_text_classification_single_label_sample(
    project: str,
    endpoint_id: str,
    content: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    instance = predict.instance.TextClassificationPredictionInstance(
        content=content,
    ).to_value()
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)

    predictions = response.predictions
    for prediction in predictions:
        x = dict(prediction)
    
    df = pd.DataFrame.from_dict(x)
    return df



# [END aiplatform_predict_text_classification_single_label_sample]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():  
    return render_template('login.html')

@app.route('/contacto', methods=["POST"])
def resultado():
    frase = request.form.get("frase")

    x = predict_text_classification_single_label_sample(
        project="14746404045",
        endpoint_id="8564848134700662784",
        location="us-central1",
        content=frase)


    x.sort_values(by='confidences', ascending=False, inplace=True)
    return render_template('login.html', tables=[x.to_html(classes='data')], titles=x.columns.values)





if __name__ == '__main__':
    app.run(debug=True, port=5000)
