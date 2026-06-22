import os.path
from ultralytics import YOLO
from collections import Counter
import mimetypes
#docker build -t coin-predictor .

#docker run --shm-size=8g
#           --volume C:\Users\olegn\Workspace\ContainerSharedFoulders\RMO\Input:/app/input_dir
#           --volume C:\Users\olegn\Workspace\ContainerSharedFoulders\RMO\Output:/app/output_dir
#           coin-predictor:latest

model = YOLO("yolov11_customdata_trained.pt")

monety_voc = {'1zl': 1, '2zl': 2, '5zl': 5}

def determine_file_type_by_ext(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type:
        if mime_type.startswith('image/'):
            return "image"
        elif mime_type.startswith('video/'):
            return "video"

    return "Unknown or other file type"


def run_model(mode, source, output_path):
    print(f"YOLO | mode={mode} | source={source}")
    if mode == "image":
        results = model.predict(source=source, project=output_path, show=False, save=True)

        r = results[0]
        class_ids = r.boxes.cls.tolist()
        counts = Counter(class_ids)
        suma = 0

        for class_id, count in counts.items():
            class_name = r.names[int(class_id)]
            value = monety_voc.get(class_name)
            suma += value * count
            print(f"{class_name}: {count}")
        print(f"SUMA: {suma}")

    elif mode == "video":
        model.predict(source=source, project=output_path, show=False, save=True)

    else:
        print("Unknown mode")

def main():
    input_dir = "/app/input_dir"
    output_dir = "/app/output_dir"

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print("No input directory or no input files found")
        return

    for e in os.scandir(input_dir):
        if e.is_file():
            file_type = determine_file_type_by_ext(e.path)
            run_model(mode=file_type, source=e.path, output_path=output_dir)


if __name__ == "__main__":
    main()