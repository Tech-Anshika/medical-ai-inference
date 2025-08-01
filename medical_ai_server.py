import os
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np
import onnxruntime
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

        # Load PyTorch models (TB and Pneumonia only)
        self.tb_model = self.load_pytorch_model(os.path.join(MODEL_PATH, "best_tb_model.pth"), "TB")
        self.pneumonia_model = self.load_pytorch_model(os.path.join(MODEL_PATH, "swasthsetu_pneumonia_model.pth"), "Pneumonia")

        # Load Malaria ONNX model
        self.malaria_onnx_session = self.load_onnx_model(os.path.join(MODEL_PATH, "malaria_resnet18.onnx"))

        logger.info("🎉 All models attempted to load (TB, Pneumonia, Malaria)")

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

    def load_onnx_model(self, path):
        try:
            session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
            logger.info(f"✅ ONNX model loaded: Malaria from {path}")
            return session
        except Exception as e:
            logger.error(f"❌ Failed to load ONNX model (Malaria) from {path}: {e}")
            return None

    def predict(self, image_data, analysis_type):
        try:
            if len(image_data) > 5_000_000:
                return {"error": "Payload too large"}

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            if analysis_type == "chest_xray":
                if any(m is None for m in [self.tb_model, self.pneumonia_model]):
                    return {"error": "Required chest X-ray models not loaded"}

                input_tensor = self.transform(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    tb_output = self.tb_model(input_tensor)
                    pneumonia_output = self.pneumonia_model(input_tensor)

                tb_prob = torch.softmax(tb_output, dim=1)[0][1].item()
                pneumonia_prob = torch.softmax(pneumonia_output, dim=1)[0][1].item()

                scores = {
                    "TB": tb_prob,
                    "Pneumonia": pneumonia_prob
                }

                if tb_prob > 0.6 and tb_prob >= pneumonia_prob:
                    diagnosis = "TB DETECTED"
                    confidence = tb_prob
                elif pneumonia_prob > 0.55:
                    diagnosis = "PNEUMONIA DETECTED"
                    confidence = pneumonia_prob
                else:
                    diagnosis = "NORMAL"
                    confidence = 1.0 - max(tb_prob, pneumonia_prob)

                return {
                    "result": diagnosis,
                    "confidence": confidence,
                    "scores": scores,
                    "message": "Analysis by Chest X-ray AI models",
                }

            elif analysis_type == "blood_smear":
                if self.malaria_onnx_session is None:
                    return {"error": "Malaria ONNX model not loaded"}

                input_name = self.malaria_onnx_session.get_inputs()[0].name
                img_resized = image.resize((224, 224))
                img_np = np.array(img_resized).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_np = (img_np - mean) / std
                img_np = np.transpose(img_np, (2, 0, 1))
                input_tensor = img_np[np.newaxis, ...]

                onnx_output = self.malaria_onnx_session.run(None, {input_name: input_tensor})
                output_array = onnx_output[0]

                if output_array.shape[-1] == 2:
                    probs = torch.softmax(torch.from_numpy(output_array[0]), dim=0).numpy()
                    malaria_prob = float(probs[1])
                else:
                    malaria_prob = float(output_array[0][1]) if output_array.ndim == 2 else float(output_array[0])

                if malaria_prob > 0.5:
                    diagnosis = "MALARIA DETECTED"
                    confidence = malaria_prob
                else:
                    diagnosis = "NORMAL"
                    confidence = 1 - malaria_prob

                return {
                    "result": diagnosis,
                    "confidence": confidence,
                    "scores": {"Malaria": malaria_prob},
                    "message": "Analysis by Malaria Blood Smear AI model",
                }

            else:
                return {"error": 'Invalid analysis type. Use "chest_xray" or "blood_smear"'}

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
    logger.info("Starting Medical AI server...")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
