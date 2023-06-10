import os
import cv2
import numpy as np
import pickle
import insightface
from insightface.app import FaceAnalysis
from src.utils.ImageAugmentation import ImageAugmentation

class KnownFaceEmbedder:

    def __init__(self, analyzer_model="buffalo_l", data_augmentation=False):
        self.analyzer = FaceAnalysis(name=analyzer_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.labels = []
        self.embeddings = []
        self.data_augmentation = data_augmentation
        self.image_augmentation = ImageAugmentation()

        self.analyzer.prepare(ctx_id=0)

    def generate_embedding(self, image, label):
        img_results = self.analyzer.get(image)

        if len(img_results) == 1:
            self.labels.append(label)
            self.embeddings.append(img_results[0].embedding)

            return img_results[0]

        print(f"Multiple faces detected: {len(img_results)}")
        return None

    def process_images(self, path = "datasets/images/"):
        print("Processing images...")

        for known_label in os.listdir(path):
            print(f"  {known_label}")

            for filename in os.listdir(f"{path}/{known_label}"):
                print(f"    {filename}")

                img = cv2.imread(f"{path}/{known_label}/{filename}")
                img_data = self.generate_embedding(img, known_label)

                if img_data and self.data_augmentation:
                    for augmentation in self.image_augmentation.augment(img, img_data.bbox, img_data.kps):
                        self.generate_embedding(augmentation, known_label)
                        #cv2.imwrite(f'datasets/augmentations/{"___" if aug_result else "no_detected__"}-{filename.split(".")[0]}-{count}.jpg', augmentation)

    def process_videos(self, path = "datasets/videos/", skip_frames=10, rotate=False):
        print("Processing vídeos...")

        for known_label in os.listdir(path):
            print(f"  {known_label}")

            for filename in os.listdir(f"{path}/{known_label}"):
                print(f"    {filename}")

                video = cv2.VideoCapture(f"{path}/{known_label}/{filename}")
                count = 0

                while video.isOpened():
                    ret, frame = video.read()

                    if (rotate):
                        frame = cv2.rotate(frame, cv2.ROTATE_180)

                    if ret:
                        print(count)
                        #cv2.imwrite(f'datasets/test/{known_label}-frame-{count}.jpg', frame)
                        self.generate_embedding(frame, known_label)
                        img_data = self.generate_embedding(frame, known_label)

                        if img_data and self.data_augmentation:
                            for augmentation in self.image_augmentation.augment(frame, img_data.bbox, img_data.kps):
                                self.generate_embedding(augmentation, known_label)

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
