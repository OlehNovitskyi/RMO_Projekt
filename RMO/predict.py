from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import Counter

model = YOLO("yolov11_customdata_trained.pt")
monety_voc = {'1zl': 1, '2zl': 2, '5zl': 5}


def analyze_image():
    path = filedialog.askopenfilename(
        title="Wybierz zdjęcie",
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    if path:
        run_model(mode="image", source=path)


def analyze_video():
    path = filedialog.askopenfilename(
        title="Wybierz wideo",
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )
    if path:
        run_model(mode="video", source=path)


def analyze_webcam():
    run_model(mode="webcam", source=0)


def run_model(mode, source):
    print(f"YOLO | mode={mode} | source={source}")
    if mode == "image":
        results = model.predict(source=source, show=False, save=True)

        r = results[0]
        class_ids = r.boxes.cls.tolist()
        counts = Counter(class_ids)
        #print(counts)
        suma = 0

        for class_id, count in counts.items():
            class_name = r.names[int(class_id)]
            value = monety_voc.get(class_name)
            suma += value * count
            print(f"{class_name}: {count}")
        print(f"SUMA: {suma}")

    elif mode == "video":
        model.predict(source=source, show=True, save=True)
    elif mode == "webcam":
        model.predict(source=source, show=True, save=True)


root = tk.Tk()
root.title("Monety")
root.geometry("300x200")

btn_image = tk.Button(root, text="Zdjęcie", command=analyze_image, width=25)
btn_video = tk.Button(root, text="Wideo", command=analyze_video, width=25)
btn_webcam = tk.Button(root, text="Webcam", command=analyze_webcam, width=25)
btn_exit = tk.Button(root, command=root.destroy, width=25)

btn_image.pack(pady=10)
btn_video.pack(pady=10)
btn_webcam.pack(pady=10)

root.mainloop()
