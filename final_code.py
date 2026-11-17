import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import os

USER_IDS = [f"User_{i}" for i in range(1, 600)]
GENDER_LABELS = ["Male", "Female"]
FINGER_LABELS = [
    "Left_index_finger",
    "Left_little_finger",
    "Left_middle_finger",
    "Left_ring_finger",
    "Left_thumb_finger",
    "Right_index_finger",
    "Right_little_finger",
    "Right_middle_finger",
    "Right_ring_finger",
    "Right_thumb_finger"
]

def apply_contrast_stretch(img):
    result = img.copy().astype(np.float32)
    flat_img = result.flatten()
    flat_img.sort()
    in_min_idx = int(0.02 * len(flat_img))
    in_max_idx = int(0.98 * len(flat_img))
    in_min = flat_img[in_min_idx]
    in_max = flat_img[in_max_idx]
    if in_max > in_min:
        result = 255.0 * (result - in_min) / (in_max - in_min)
    result[result < 0] = 0
    result[result > 255] = 255
    return result.astype(np.uint8)

def apply_histogram_equalization(img):
    result = img.copy()
    height, width = result.shape
    pixel_count = height * width
    hist = np.zeros(256, dtype=np.int32)
    for i in range(height):
        for j in range(width):
            hist[result[i, j]] += 1
    cdf = np.zeros(256, dtype=np.float32)
    cdf[0] = hist[0] / pixel_count
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i] / pixel_count
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = round(cdf[i] * 255)
    for i in range(height):
        for j in range(width):
            result[i, j] = lut[result[i, j]]
    return result

def apply_negative(img):
    result = np.zeros_like(img)
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            result[i, j] = 255 - img[i, j]
    return result

def apply_sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    height, width = img.shape
    result = np.zeros_like(img)
    pad = 1
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            sum_val = 0
            for ki in range(-pad, pad + 1):
                for kj in range(-pad, pad + 1):
                    sum_val += img[i + ki, j + kj] * kernel[ki + pad, kj + pad]
            result[i, j] = np.clip(sum_val, 0, 255)
    result[0:pad, :] = img[0:pad, :]
    result[height - pad:height, :] = img[height - pad:height, :]
    result[:, 0:pad] = img[:, 0:pad]
    result[:, width - pad:width] = img[:, width - pad:width]
    return result

def apply_laplacian(img):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    height, width = img.shape
    lap = np.zeros_like(img, dtype=np.float32)
    pad = 1
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            sum_val = 0
            for ki in range(-pad, pad + 1):
                for kj in range(-pad, pad + 1):
                    sum_val += img[i + ki, j + kj] * kernel[ki + pad, kj + pad]
            lap[i, j] = sum_val
    lap = np.abs(lap)
    lap_max = np.max(lap)
    if lap_max > 0:
        lap = lap * (255.0 / lap_max)
    return lap.astype(np.uint8)

def apply_gamma_correction(img, gamma=1.5):
    result = np.zeros_like(img, dtype=np.float32)
    inv_gamma = 1.0 / gamma
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = np.clip(int(((i / 255.0) ** inv_gamma) * 255.0), 0, 255)
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            result[i, j] = lut[img[i, j]]
    return result.astype(np.uint8)

def apply_segmentation(img):
    hist = np.zeros(256, dtype=np.int32)
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            hist[img[i, j]] += 1
    total_pixels = height * width
    max_variance = 0
    best_threshold = 0
    for threshold in range(1, 256):
        weight1 = np.sum(hist[:threshold]) / total_pixels
        weight2 = np.sum(hist[threshold:]) / total_pixels
        if weight1 == 0 or weight2 == 0:
            continue
        mean1 = 0
        for i in range(threshold):
            mean1 += i * hist[i]
        mean1 /= (weight1 * total_pixels)
        mean2 = 0
        for i in range(threshold, 256):
            mean2 += i * hist[i]
        mean2 /= (weight2 * total_pixels)
        variance = weight1 * weight2 * ((mean1 - mean2) ** 2)
        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold
    segmented = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            segmented[i, j] = 255 if img[i, j] >= best_threshold else 0
    return segmented

def apply_optimized_fingerprint_pipeline(img):
    kernel_size = 5
    sigma = 1.0
    half_size = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - half_size
            y = j - half_size
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    height, width = img.shape
    blurred = np.zeros_like(img, dtype=np.float32)
    for i in range(half_size, height - half_size):
        for j in range(half_size, width - half_size):
            sum_val = 0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    sum_val += img[i + ki - half_size, j + kj - half_size] * kernel[ki, kj]
            blurred[i, j] = sum_val
    blurred = blurred.astype(np.uint8)
    pixel_count = height * width
    hist = np.zeros(256, dtype=np.int32)
    for i in range(height):
        for j in range(width):
            hist[blurred[i, j]] += 1
    cdf = np.zeros(256, dtype=np.float32)
    cdf[0] = hist[0] / pixel_count
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i] / pixel_count
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = round(cdf[i] * 255)
    equalized = np.zeros_like(blurred)
    for i in range(height):
        for j in range(width):
            equalized[i, j] = lut[blurred[i, j]]
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    gradient_x = np.zeros_like(equalized, dtype=np.float32)
    gradient_y = np.zeros_like(equalized, dtype=np.float32)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            sum_x = 0
            for ki in range(3):
                for kj in range(3):
                    sum_x += equalized[i + ki - 1, j + kj - 1] * sobel_x[ki, kj]
            gradient_x[i, j] = sum_x
            sum_y = 0
            for ki in range(3):
                for kj in range(3):
                    sum_y += equalized[i + ki - 1, j + kj - 1] * sobel_y[ki, kj]
            gradient_y[i, j] = sum_y
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    suppressed = np.zeros_like(gradient_magnitude)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = (gradient_direction[i, j] * 180 / np.pi) % 180
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                neighbor1 = gradient_magnitude[i, j + 1]
                neighbor2 = gradient_magnitude[i, j - 1]
            elif 22.5 <= angle < 67.5:
                neighbor1 = gradient_magnitude[i - 1, j + 1]
                neighbor2 = gradient_magnitude[i + 1, j - 1]
            elif 67.5 <= angle < 112.5:
                neighbor1 = gradient_magnitude[i - 1, j]
                neighbor2 = gradient_magnitude[i + 1, j]
            else:
                neighbor1 = gradient_magnitude[i - 1, j - 1]
                neighbor2 = gradient_magnitude[i + 1, j + 1]
            if (gradient_magnitude[i, j] >= neighbor1) and (gradient_magnitude[i, j] >= neighbor2):
                suppressed[i, j] = gradient_magnitude[i, j]
    low_threshold = 50
    high_threshold = 150
    strong_edges = (suppressed > high_threshold)
    weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)
    edges = np.zeros_like(suppressed, dtype=np.uint8)
    edges[strong_edges] = 255
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if weak_edges[i, j]:
                if any(strong_edges[i - 1:i + 2, j - 1:j + 2].flatten()):
                    edges[i, j] = 255
    return edges

def add_images(img1, img2):
    result = np.zeros_like(img1)
    height, width = img1.shape
    for i in range(height):
        for j in range(width):
            value = int(img1[i, j]) + int(img2[i, j])
            result[i, j] = min(value, 255)
    return result

def subtract_images(img1, img2):
    result = np.zeros_like(img1)
    height, width = img1.shape
    for i in range(height):
        for j in range(width):
            value = int(img1[i, j]) - int(img2[i, j])
            result[i, j] = max(value, 0)
    return result

def apply_clahe(img, clip_limit=4.0, grid_size=8):
    height, width = img.shape
    result = np.zeros_like(img)
    h_grid = height // grid_size
    w_grid = width // grid_size
    for y in range(grid_size):
        for x in range(grid_size):
            y_start = y * h_grid
            y_end = (y + 1) * h_grid if y < grid_size - 1 else height
            x_start = x * w_grid
            x_end = (x + 1) * w_grid if x < grid_size - 1 else width
            grid = img[y_start:y_end, x_start:x_end]
            grid_height, grid_width = grid.shape
            grid_pixels = grid_height * grid_width
            hist = np.zeros(256, dtype=np.int32)
            for i in range(grid_height):
                for j in range(grid_width):
                    hist[grid[i, j]] += 1
            clip_height = int(clip_limit * grid_pixels / 256)
            excess = 0
            for i in range(256):
                if hist[i] > clip_height:
                    excess += hist[i] - clip_height
                    hist[i] = clip_height
            redistrib = excess // 256
            for i in range(256):
                hist[i] += redistrib
            cdf = np.zeros(256, dtype=np.float32)
            cdf[0] = hist[0] / grid_pixels
            for i in range(1, 256):
                cdf[i] = cdf[i - 1] + hist[i] / grid_pixels
            lut = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                lut[i] = round(cdf[i] * 255)
            for i in range(grid_height):
                for j in range(grid_width):
                    result[y_start + i, x_start + j] = lut[grid[i, j]]
    return result

def apply_gabor_filter(img, theta=45, wavelength=10, sigma=5, gamma=0.5, psi=0):
    theta_rad = theta * np.pi / 180
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    half_size = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    for y in range(kernel_size):
        for x in range(kernel_size):
            x_prime = x - half_size
            y_prime = y - half_size
            x_theta = x_prime * np.cos(theta_rad) + y_prime * np.sin(theta_rad)
            y_theta = -x_prime * np.sin(theta_rad) + y_prime * np.cos(theta_rad)
            gaussian = np.exp(-(x_theta**2 + (gamma**2) * (y_theta**2)) / (2 * sigma**2))
            sinusoid = np.cos((2 * np.pi * x_theta / wavelength) + psi)
            kernel[y, x] = gaussian * sinusoid
    kernel = kernel / np.sqrt(np.sum(kernel**2))
    height, width = img.shape
    result = np.zeros_like(img, dtype=np.float32)
    for i in range(half_size, height - half_size):
        for j in range(half_size, width - half_size):
            sum_val = 0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    sum_val += img[i + ki - half_size, j + kj - half_size] * kernel[ki, kj]
            result[i, j] = sum_val
    result_min = np.min(result)
    result_max = np.max(result)
    if result_max > result_min:
        result = 255 * (result - result_min) / (result_max - result_min)
    for i in range(half_size):
        result[i, :] = img[i, :]
        result[height - i - 1, :] = img[height - i - 1, :]
    for j in range(half_size):
        result[:, j] = img[:, j]
        result[:, width - j - 1] = img[:, width - j - 1]
    return result.astype(np.uint8)

def apply_multi_gabor_filter(img, num_orientations=8):
    height, width = img.shape
    result = np.zeros_like(img, dtype=np.float32)
    for i in range(num_orientations):
        theta = i * 180 / num_orientations
        filtered = apply_gabor_filter(img, theta=theta, wavelength=8, sigma=4, gamma=0.5, psi=0)
        result = np.maximum(result, filtered)
    result_min = np.min(result)
    result_max = np.max(result)
    if result_max > result_min:
        result = 255 * (result - result_min) / (result_max - result_min)
    return result.astype(np.uint8)

class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Grayscale Image Filter & Fingerprint Classifier")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        self.image = None
        self.processed_image = None

        try:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "model",
                "fingerprint_model_final_20250509_1311.h5",
            )
            self.model = load_model(model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.model = None

        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 10), padding=6)
        self.style.configure("TCheckbutton", font=("Helvetica", 10))

        self.create_widgets()

    def create_widgets(self):
        self.frame = ttk.Frame(self.root, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(self.frame, text="Filters & Model", padding=10)
        controls.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        self.upload_btn = ttk.Button(controls, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=(0, 10), fill='x')

        self.filters = {
            "Contrast Stretch": tk.BooleanVar(),
            "Histogram Equalization": tk.BooleanVar(),
            "Negative": tk.BooleanVar(),
            "Sharpen": tk.BooleanVar(),
            "Laplacian": tk.BooleanVar(),
            "Optimized Fingerprint Pipeline": tk.BooleanVar(),
            "Gamma Correction": tk.BooleanVar(),
            "Segmentation": tk.BooleanVar(),
            "CLAHE": tk.BooleanVar(),
            "Gabor Filter": tk.BooleanVar(),
            "Multi-Gabor Filter": tk.BooleanVar()
        }

        for name, var in self.filters.items():
            ttk.Checkbutton(controls, text=name, variable=var).pack(anchor="w")

        self.combobox_operation = ttk.Combobox(controls, values=["None", "Add", "Subtract"], state="readonly")
        self.combobox_operation.set("None")
        self.combobox_operation.pack(pady=5, fill='x')

        self.run_btn = ttk.Button(controls, text="Run Filters", command=self.run_filters)
        self.run_btn.pack(pady=10, fill='x')

        self.test_btn = ttk.Button(controls, text="Test Fingerprint Model", command=self.test_model)
        self.test_btn.pack(pady=10, fill='x')

        preview_frame = ttk.Frame(self.frame)
        preview_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(0, weight=1)

        self.canvas_orig_label = ttk.Label(preview_frame, text="Original Image")
        self.canvas_orig_label.pack()
        self.canvas_orig = tk.Label(preview_frame, bd=2, relief="solid", width=250, height=250)
        self.canvas_orig.pack(pady=5)

        self.canvas_proc_label = ttk.Label(preview_frame, text="Processed Image")
        self.canvas_proc_label.pack()
        self.canvas_proc = tk.Label(preview_frame, bd=2, relief="solid", width=250, height=250)
        self.canvas_proc.pack(pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.image is None:
                messagebox.showerror("Error", "Could not load image.")
                return
            self.display_image(self.image, self.canvas_orig)

    def display_image(self, img, canvas):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((250, 250), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        canvas.img = img_tk
        canvas.configure(image=img_tk)

    def run_filters(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        try:
            result = self.image.copy()

            filter_order = [
                "Contrast Stretch",
                "Histogram Equalization",
                "CLAHE",
                "Gamma Correction",
                "Sharpen",
                "Laplacian",
                "Gabor Filter",
                "Multi-Gabor Filter",
                "Segmentation",
                "Negative",
                "Optimized Fingerprint Pipeline"
            ]

            for name in filter_order:
                if self.filters[name].get():
                    func_name = f"apply_{name.lower().replace(' ', '_')}"
                    func = globals()[func_name]
                    result = func(result)

            op = self.combobox_operation.get()
            if op == "Add":
                result = add_images(self.image, result)
            elif op == "Subtract":
                result = subtract_images(self.image, result)

            self.processed_image = result
            self.display_image(result, self.canvas_proc)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def test_model(self):
        if self.processed_image is None:
            messagebox.showwarning("No Image", "Please process an image first.")
            return
        if self.model is None:
            messagebox.showwarning("Model Not Loaded", "The fingerprint model is not loaded.")
            return
        try:
            img_resized = cv2.resize(self.processed_image, (128, 128))
            img_normalized = img_resized / 255.0
            img_input = np.expand_dims(img_normalized, axis=(0, -1))

            predictions = self.model.predict(img_input, verbose=0)

            user_probs = predictions[0].flatten()
            user_idx = np.argmax(user_probs)
            user_confidence = user_probs[user_idx]
            user_label = USER_IDS[user_idx]

            gender_probs = predictions[1].flatten()
            gender_idx = np.argmax(gender_probs)
            gender_confidence = gender_probs[gender_idx]
            gender_label = GENDER_LABELS[gender_idx]

            finger_probs = predictions[2].flatten()
            finger_idx = np.argmax(finger_probs)
            finger_confidence = finger_probs[finger_idx]
            finger_label = FINGER_LABELS[finger_idx]

            result_text = (
                f"👤 **User ID:** {user_label} ({user_confidence * 100:.2f}% confidence)\n"
                f"🚻 **Gender:** {gender_label} ({gender_confidence * 100:.2f}% confidence)\n"
                f"🖐️ **Finger Type:** {finger_label} ({finger_confidence * 100:.2f}% confidence)"
            )

            messagebox.showinfo("Model Prediction", result_text)

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()