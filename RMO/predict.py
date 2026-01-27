from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox

# ---- Callbacks ----
model = YOLO("yolov11_customdata_trained.pt")
monety_voc = {'1zl': 1, '2zl': 2, '5zl': 5}


def analyze_image():
    path = filedialog.askopenfilename(
        title="Wybierz zdjęcie",
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    if path:
        messagebox.showinfo("Selected image", path)
        run_model(mode="image", source=path)


def analyze_video():
    path = filedialog.askopenfilename(
        title="Wybierz wideo",
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )
    if path:
        messagebox.showinfo("Selected video", path)
        run_model(mode="video", source=path)


def analyze_webcam():
    run_model(mode="webcam", source=0)


def run_model(mode, source):
    print(f"YOLO | mode={mode} | source={source}")
    if mode == "image":
        results = model.predict(source=source, show=True, save=False)
        suma = 0
        for _, name in results[0].names.items():
            suma += monety_voc.get(name)
        print(suma)
    else:
        model.predict(source=source, show=False, save=True)


root = tk.Tk()
root.title("Monety")
root.geometry("300x200")

btn_image = tk.Button(root, text="Zdjęcie", command=analyze_image, width=25)
btn_video = tk.Button(root, text="Wideo", command=analyze_video, width=25)
btn_webcam = tk.Button(root, text="Webcam", command=analyze_webcam, width=25)

btn_image.pack(pady=10)
btn_video.pack(pady=10)
btn_webcam.pack(pady=10)

root.mainloop()
