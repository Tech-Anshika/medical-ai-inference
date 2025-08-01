import os
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical_ai_server")

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "./models")

class MedicalModels:
    def __init__(self):
        self.device = torch.device("cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Sirf Pneumonia model load karo
        self.pneumonia_model = self.load_pytorch_model(
            os.path.join(MODEL_PATH, "swasthsetu_pneumonia_model.pth"),
            "Pneumonia"
        )

        logger.info("🎉 Pneumonia model attempted to load")

    def load_pytorch_model(self, path, name):
        try:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            state_dict = torch.load(path, map_location=self.device)

            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                model.load_state_dict(state_dict["state_dict"], strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)

            model.to(self.device)
            model.eval()
            logger.info(f"✅ PyTorch model loaded: {name} from {path}")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load PyTorch model {name} from {path}: {e}")
            return None

    def predict(self, image_data, analysis_type):
        try:
            if len(image_data) > 5_000_000:
                return {"error": "Payload too large"}

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            if analysis_type != "chest_xray":
                return {"error": 'Invalid analysis type. Only "chest_xray" supported in this trimmed version.'}

            if self.pneumonia_model is None:
                return {"error": "Pneumonia model not loaded"}

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pneumo_output = self.pneumonia_model(input_tensor)

            pneumonia_prob = torch.softmax(pneumo_output, dim=1)[0][1].item()

            if pneumonia_prob > 0.55:
                diagnosis = "PNEUMONIA DETECTED"
                confidence = pneumonia_prob
            else:
                diagnosis = "NORMAL"
                confidence = 1.0 - pneumonia_prob

            return {
                "result": diagnosis,
                "confidence": confidence,
                "scores": {"Pneumonia": pneumonia_prob},
                "message": "Analysis by Pneumonia model",
            }

        except Exception as e:
            logger.exception("Prediction error")
            return {"error": str(e)}

medical_models = MedicalModels()

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json(force=True)
        image_data = data.get("image")
        analysis_type = data.get("type", "chest_xray")

        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        logger.info(f"Running inference for type: {analysis_type}")
        result = medical_models.predict(image_data, analysis_type)

        if "error" in result:
            logger.warning(f"Inference error: {result['error']}")
            return jsonify(result), 500

        logger.info(f"Inference result: {result['result']} confidence={result['confidence']}")
        return jsonify(result)
    except Exception as e:
        logger.exception("Endpoint error")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    logger.info("Starting Medical AI server (Pneumonia-only)...")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
