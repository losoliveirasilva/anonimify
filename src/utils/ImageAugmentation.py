from src.utils.ImageProcessor import ImageProcessor

class ImageAugmentation():

    def __init__(self):
        self.image_processor = ImageProcessor()

    def augment(self, image, bbox, kps):
        augmentations = []

        augmentations.append(self.image_processor.pixelate(self.image_processor.avg_blur(image)))

        eyes = self.image_processor.cover_eyes(image, bbox, kps)
        augmentations.append(eyes)
        augmentations.append(self.image_processor.pixelate(self.image_processor.avg_blur(eyes)))

        mouth = self.image_processor.cover_mouth(image, bbox, kps)
        augmentations.append(mouth)
        augmentations.append(self.image_processor.pixelate(self.image_processor.avg_blur(mouth)))

        return augmentations
