from flask import Flask, request, jsonify
from inference_pipeline import *
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route("/v1/predict", methods=["GET", "POST"])
def predict_by_index() -> jsonify:
    """
    Predicts based on a specific index in the dataset. The user provides an index,
    which is used to extract a row from the dataset. The model name can be specified
    or defaulted to 'SVM_FE'.

    Expected JSON input:
    {
        "index": int,
        "model": str (optional)
    }

    Returns:
        JSON response with the prediction, probability, and prediction string.
    """
    try:
        # Parse user input
        user_input = request.get_json()
        index = user_input["index"]
        model_name = user_input.get("model", "SVM_FE")

        # fetching data
        raw_data = fetch_data()
        raw_data = raw_data.loc[index].to_frame().T

        # predict data
        prediction, prediction_proba, prediction_str = prediction_pipeline({}, model_name, raw_data)

        # Return the result as JSON
        return jsonify(
            prediction=int(prediction[0]), 
            prediction_proba=prediction_proba, 
            prediction_str=prediction_str
        )
    
    except KeyError as e:
        return jsonify({"error": f"Missing required input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v2/predict", methods=["GET", "POST"])
def predict_new_data() -> jsonify:
    """
    Predicts based on user-provided input data. The user sends the raw input data
    along with an optional model name. If no model is provided, it defaults to 'SVM_FE'.

    Expected JSON input:
    {
        "data": Dict[str, Any],
        "model": str (optional)
    }

    Returns:
        JSON response with the prediction, probability, and prediction string.
    """
    try:
        # parse user input
        user_input = request.get_json()
        raw_input_from_user = user_input["data"]
        model_name = user_input.get("model", "SVM_FE")

        # predict input
        prediction, prediction_proba, prediction_str = prediction_pipeline(raw_input_from_user, model_name=model_name)
        
        # Return the result as JSON
        return jsonify(
            prediction=int(prediction[0]), 
            prediction_proba=prediction_proba, 
            prediction_str=prediction_str
        )
    
    except KeyError as e:
        return jsonify({"error": f"Missing required input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5002)