from insightface.app import FaceAnalysis
from keras.models import load_model
import pickle
import cv2
import numpy as np
import time

class VideoAnonymizer:

    def __init__(self, labeled_embeddings, encoded_labels_path):
        with open(encoded_labels_path, "rb") as f:
            self.encoded_labels = pickle.load(f)

        self.embeddings = np.array(labeled_embeddings['embeddings'])
        self.labels = self.encoded_labels.fit_transform(labeled_embeddings['labels'])

        model_pack_name = 'buffalo_l'
        self.face_embedder = FaceAnalysis(name=model_pack_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_embedder.prepare(ctx_id=0)

    def anonymize_regions(self, video_path, anonymized_path, frames_regions, people_to_keep):
        start = time.time()
        video = cv2.VideoCapture(video_path)

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        fps = int(video.get(cv2.CAP_PROP_FPS))
        anonymized = cv2.VideoWriter(anonymized_path, cv2.VideoWriter_fourcc("m","p","4","v"), fps, (frame_width, frame_height))

        while(video.isOpened()):
            ret, image = video.read()

            if ret == True:
                percentage = 100 * (frame_count + 1) / total_frame_count
                filled = int(percentage)
                empty = int(100-filled)

                end = time.time()
                hours, rem = divmod(end-start, 3600)
                minutes, seconds = divmod(rem, 60)
                pretty_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
                print(f'\rElapsed time: {pretty_time} - [{"="*filled}{" "*empty}] {"{:.1f}".format(percentage)}% ', end='')

                for frame_region in frames_regions[frame_count]:
                    x, y = frame_region["box"][0][0], frame_region["box"][0][1]
                    w, h = frame_region["box"][1][0] - frame_region["box"][0][0], frame_region["box"][1][1] - frame_region["box"][0][1]

                    roi = image[y:y+h, x:x+w]
                    cv2.imwrite(f"outputs/faces/{frame_count}-{frame_region['label']}.jpg", roi)

                    if not(frame_region["label"] in people_to_keep) or frame_region["probability"] < 0.8:
                        x, y = frame_region["box"][0][0], frame_region["box"][0][1]
                        w, h = frame_region["box"][1][0] - frame_region["box"][0][0], frame_region["box"][1][1] - frame_region["box"][0][1]

                        roi = image[y:y+h, x:x+w]
                        blur = cv2.GaussianBlur(roi, (101, 101), 0)

                        image[y:y+h, x:x+w] = blur
                    else:
                        text = "{} - {}".format(frame_region["label"], frame_region["probability"]*100)

                        y = frame_region["box"][0][1] - 10 if frame_region["box"][0][1] - 10 > 10 else frame_region["box"][0][1] + 10
                        cv2.putText(image, text, (int(frame_region["box"][0][0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        cv2.rectangle(image, frame_region["box"][0], frame_region["box"][1], (0,255,0), 2)

                frame_count += 1
                anonymized.write(image)
            else:
                break

        print()
        video.release()
        anonymized.release()

    def save_regions(self, video_path, model_path, save_path):
        start = time.time()
        video = cv2.VideoCapture(video_path)
        model = load_model(model_path)
        frame_count = 0
        total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        regions = []

        while(video.isOpened()):
            ret, image = video.read()

            if ret == True:
                frame_count += 1

                percentage = 100 * frame_count / total_frame_count
                filled = int(percentage)
                empty = int(100-filled)
                end = time.time()
                hours, rem = divmod(end-start, 3600)
                minutes, seconds = divmod(rem, 60)
                pretty_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
                print(f'\rElapsed time: {pretty_time} - [{"="*filled}{" "*empty}] {"{:.1f}".format(percentage)}% ', end='')

                img_results = self.face_embedder.get(image)

                if len(img_results) != 0:
                    frame_regions = []

                    for result in img_results:
                        bbox = result.bbox
                        embedding = result.embedding.reshape(1,-1)

                        preds = model.predict(embedding, verbose=0)
                        preds = preds.flatten()

                        j = np.argmax(preds)
                        proba = preds[j]

                        top_left = (max(0, int(bbox[0])), max(0, int(bbox[1])))
                        bottom_right = (max(0, int(bbox[2])), max(0, int(bbox[3])))

                        name = self.encoded_labels.classes_[j]

                        frame_regions.append({"label": name, "probability": proba, "box": [top_left, bottom_right]})

                    regions.append(frame_regions)
                else:
                    regions.append([])


            else:
                break

        print()
        video.release()
        f = open(save_path, "wb")
        f.write(pickle.dumps(regions))
        f.close()

        return regions
