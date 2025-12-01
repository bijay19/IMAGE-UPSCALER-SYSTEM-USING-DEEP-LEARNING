import os
import uuid
import numpy as np
import cv2
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from tensorflow import keras
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "super-secret-change-this"

# -------------------------------------------------
# Custom PixelShuffle layer
# -------------------------------------------------
@keras.utils.register_keras_serializable()
class PixelShuffle(keras.layers.Layer):
    def __init__(self, block_size=2, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.block_size)

    def get_config(self):
        config = super(PixelShuffle, self).get_config()
        config.update({"block_size": self.block_size})
        return config

# -------------------------------------------------
# Load models
# -------------------------------------------------
MODEL_PATHS = {
    "mae": "model_mae.keras",
    "a005": "model_a005.keras",
}

loaded_models = {}
for key, path in MODEL_PATHS.items():
    if os.path.exists(path):
        print(f"[INFO] Loading model: {path}")
        loaded_models[key] = keras.models.load_model(
            path,
            custom_objects={"PixelShuffle": PixelShuffle},
            compile=False,
        )
    else:
        print(f"[WARN] Model file not found: {path}")

# -------------------------------------------------
# Helper: run SR and compute PSNR/SSIM
# -------------------------------------------------
def run_sr_on_image(model, rgb_image, hr_image=None, lr_size=(128, 128)):
    # Resize input
    lr_img = cv2.resize(rgb_image, lr_size, interpolation=cv2.INTER_CUBIC)

    # Normalize [-1,1]
    lr_in = lr_img.astype("float32") / 255.0
    lr_in = (lr_in * 2.0) - 1.0
    lr_in = np.expand_dims(lr_in, axis=0)

    # Predict SR
    sr_pred = model.predict(lr_in, verbose=0)
    sr_img = (sr_pred[0] + 1.0) / 2.0
    sr_img = np.clip(sr_img, 0.0, 1.0)
    sr_uint8 = (sr_img * 255).astype("uint8")

    # Compute metrics (if ground truth provided)
    psnr_val, ssim_val = None, None
    if hr_image is not None:
        hr_resized = cv2.resize(hr_image, sr_uint8.shape[:2][::-1])
        psnr_val = psnr(hr_resized, sr_uint8, data_range=255)
        ssim_val = ssim(hr_resized, sr_uint8, multichannel=True)

    return lr_img, sr_uint8, psnr_val, ssim_val

# -------------------------------------------------
# Helper: compute sharpness and contrast
# -------------------------------------------------
def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.std(gray)

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check file
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["image"]
        model_choice = request.form.get("model_choice", "mae")

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if model_choice not in loaded_models:
            flash("Selected model is not loaded on server.")
            return redirect(request.url)

        if file:
            # Save upload
            ext = os.path.splitext(file.filename)[1].lower()
            uid = uuid.uuid4().hex
            in_name = f"{uid}_in{ext}"
            in_path = os.path.join(UPLOAD_FOLDER, in_name)
            file.save(in_path)

            # Read as RGB
            bgr = cv2.imread(in_path)
            if bgr is None:
                flash("Could not read the uploaded image.")
                return redirect(request.url)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Run SR
            model = loaded_models[model_choice]
            lr_img, sr_img, psnr_val, ssim_val = run_sr_on_image(model, rgb)

            # --- Compute numeric metrics (only for results) ---
            sharp_lr, sharp_sr = sharpness(lr_img), sharpness(sr_img)
            contrast_lr, contrast_sr = contrast(lr_img), contrast(sr_img)
            sharp_improve = ((sharp_sr - sharp_lr) / sharp_lr) * 100 if sharp_lr != 0 else 0
            contrast_improve = ((contrast_sr - contrast_lr) / contrast_lr) * 100 if contrast_lr != 0 else 0

            # Save outputs
            out_lr_name = f"{uid}_lr.png"
            out_sr_name = f"{uid}_sr.png"

            cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_lr_name),
                        cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_sr_name),
                        cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))

            return render_template(
                "index.html",
                lr_image=url_for("output_file", filename=out_lr_name),
                sr_image=url_for("output_file", filename=out_sr_name),
                psnr_val=psnr_val,
                ssim_val=ssim_val,
                sharp_lr=round(sharp_lr, 2),
                sharp_sr=round(sharp_sr, 2),
                contrast_lr=round(contrast_lr, 2),
                contrast_sr=round(contrast_sr, 2),
                sharp_improve=round(sharp_improve, 1),
                contrast_improve=round(contrast_improve, 1),
                chosen_model=model_choice,
                models_available=list(loaded_models.keys()),
                done=True,
            )

    # GET
    return render_template(
        "index.html",
        models_available=list(loaded_models.keys()),
        done=False,
    )

@app.route("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# -------------------------------------------------
# RUN APP
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
