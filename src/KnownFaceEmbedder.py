import os
import cv2
import numpy as np
import pickle
import insightface
from insightface.app import FaceAnalysis

class KnownFaceEmbedder:

    def __init__(self, analyzer_model="buffalo_l"):
        self.analyzer = FaceAnalysis(name=analyzer_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.labels = []
        self.embeddings = []
        self.analyzer.prepare(ctx_id=0)

    def load_embeddings_from_file(self, filename):
        with open(filename, "rb") as f:
            labeled_embeddings = pickle.load(f)

        self.labels = labeled_embeddings["labels"]
        self.embeddings = labeled_embeddings["embeddings"]

        # necessary to perform some imread previously, otherwise it gives memory error
        img = cv2.imread("/home/los/Documents/ufsc/tcc/new_tests/datasets/Ben_Affleck/000001.jpg")
        self.analyzer.get(img)

    def generate_embedding(self, image, label):
        img_results = self.analyzer.get(image)

        if len(img_results) == 1:
            self.labels.append(label)
            self.embeddings.append(img_results[0].embedding)

            return img_results[0]

        print(f"Multiple faces detected: {len(img_results)}")
        return None

    def process_images(self, path = "datasets/images/"):
        for known_label in os.listdir(path):
            for filename in os.listdir(f"{path}/{known_label}"):
                img = cv2.imread(f"{path}/{known_label}/{filename}")
                img_data = self.generate_embedding(img, known_label)

    def process_videos(self, path = "datasets/videos/", skip_frames=10, rotate=False):
        for known_label in os.listdir(path):
            for filename in os.listdir(f"{path}/{known_label}"):
                video = cv2.VideoCapture(f"{path}/{known_label}/{filename}")
                count = 0

                while video.isOpened():
                    ret, frame = video.read()

                    if (rotate):
                        frame = cv2.rotate(frame, cv2.ROTATE_180)

                    if ret:
                        self.generate_embedding(frame, known_label)
                        img_data = self.generate_embedding(frame, known_label)
                        count += skip_frames
                        video.set(cv2.CAP_PROP_POS_FRAMES, count)


                    else:
                        video.release()
                        break

    def process(self):
        self.process_images()
        self.process_videos()

    def labeled_embeddings(self):
        return {"embeddings": self.embeddings, "labels": self.labels}

    def save_file(self, path = "outputs/processed/labeled_embeddings.pickle"):
        f = open(path, "wb")
        f.write(pickle.dumps(self.labeled_embeddings()))
        f.close()
