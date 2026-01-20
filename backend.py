from flask import Flask, jsonify
from datetime import datetime
import numpy as np
import subprocess
import base64
import json
import os
import cv2


app = Flask(__name__)

IMAGE_PATH = "/home/nekodu-02/ee491/nesnes/491_project/491_project/inputs" 
RESULT_JSON = "/home/nekodu-02/ee491/nesnes/491_project/491_project/outputs/latest_result.json"

@app.route("/status")
def status():
    return "running", 200


@app.route("/force_classification")
def force_classification():
    try:
        if not os.path.exists(IMAGE_PATH):
            return jsonify({"error": "Image not found"}), 404

   
        result = subprocess.run(
            ["python3",
             "/home/nekodu-02/ee491/nesnes/491_project/491_project/classification_module/classify.py",
             "--input", IMAGE_PATH],
            capture_output=True, text=True
        )

        
        raw_output = result.stdout.strip().lower()
        last_line = raw_output.split("\n")[-1].strip()

        if "ants" in last_line:
            prediction = "ants"
        elif "bees" in last_line:
            prediction = "bees"
        else:
            prediction = "unknown"

        
        with open(IMAGE_PATH, "rb") as f:
            b64img = base64.b64encode(f.read()).decode("utf-8")

        result_data = {
            "prediction": prediction,
            "base64_slices": [b64img]
        }

        
        with open(RESULT_JSON, "w") as fp:
            json.dump(result_data, fp)

        return jsonify(result_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

from flask import request

@app.route("/upload_and_classify", methods=["POST"])
def upload_and_classify():
    try:
        
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        
        file.save(IMAGE_PATH)

        
        result = subprocess.run(
            ["python3",
             "/home/nekodu-02/ee491/nesnes/491_project/491_project/classification_module/classify.py",
             "--input", IMAGE_PATH],
            capture_output=True, text=True
        )

        raw_output = result.stdout.strip().lower()
        last_line = raw_output.split("\n")[-1].strip()

        if "ants" in last_line:
            prediction = "ants"
        elif "bees" in last_line:
            prediction = "bees"
        else:
            prediction = "unknown"

        
        with open(IMAGE_PATH, "rb") as f:
            b64img = base64.b64encode(f.read()).decode("utf-8")

        result_data = {
            "prediction": prediction,
            "base64_slices": [b64img]
        }

        
        with open(RESULT_JSON, "w") as fp:
            json.dump(result_data, fp)

        
        return jsonify(result_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/upload_dicom_to_server", methods=["POST"])
def upload_dicom_to_server():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"

        save_path = os.path.join(IMAGE_PATH, filename)
        file.save(save_path)

        return jsonify({
            "status": "success",
            "saved_as": filename
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/slice_and_classify_latest", methods=["POST"])
def slice_and_classify_latest():
    try:
       
        npz_files = [
            f for f in os.listdir(IMAGE_PATH)
            if f.endswith(".npz")
        ]

        if not npz_files:
            return jsonify({"error": "No NPZ file found"}), 404

        npz_files.sort()
        latest_npz = npz_files[-1]
        npz_path = os.path.join(IMAGE_PATH, latest_npz)

        
        npz_data = np.load(npz_path, allow_pickle=True)

        if "volume" not in npz_data:
            return jsonify({"error": "NPZ must contain 'volume'"}), 400

        volume = npz_data["volume"] 

        slice_results = []

       
        for idx, slice_img in enumerate(volume):
            slice_name = f"slice_{idx:03d}.png"
            slice_path = os.path.join(IMAGE_PATH, slice_name)

            
            slice_img_norm = cv2.normalize(
                slice_img,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX
            ).astype("uint8")

            cv2.imwrite(slice_path, slice_img_norm)

            
            result = subprocess.run(
                [
                    "python3",
                    "/home/nekodu-02/ee491/nesnes/491_project/491_project/classification_module/classify.py",
                    "--input",
                    slice_path
                ],
                capture_output=True,
                text=True
            )
            

            
            raw_output = result.stdout.strip().lower()

            if raw_output == "ants":
                prediction = "ants"
            elif raw_output == "bees":
                prediction = "bees"
            else:
                prediction = "unknown"

            
            with open(slice_path, "rb") as img_file:
                slice_b64 = base64.b64encode(img_file.read()).decode("utf-8")

            slice_results.append({
                "slice_index": idx,
                "prediction": prediction,
                "image_base64": slice_b64   
            })

        
        result_data = {
            "source_npz": latest_npz,
            "num_slices": len(slice_results),
            "results": slice_results
        }

        with open(RESULT_JSON, "w") as fp:
            json.dump(result_data, fp, indent=2)

        
        return jsonify({
            "status": "completed",
            "num_slices": len(slice_results)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

      

@app.route("/get_latest_classification")
def get_latest_classification():
    try:
        if not os.path.exists(RESULT_JSON):
            return jsonify({"error": "No result found"}), 404

        with open(RESULT_JSON, "r") as fp:
            result_data = json.load(fp)

        return jsonify({
            "source_npz": result_data.get("source_npz"),
            "num_slices": result_data.get("num_slices"),
            "slices": result_data.get("results", [])
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=44155)