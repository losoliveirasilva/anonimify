import cv2

class ImageProcessor():

    def pixelate(self, image, pixel_size=None):
        if pixel_size:
            px, py = (pixel_size, pixel_size)
        else:
            px, py = (int(image.shape[1]/5), int(image.shape[0]/5))

        height, width = image.shape[:2]
        temp = cv2.resize(image, (px, py), interpolation=cv2.INTER_LINEAR)
        output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
        return output

    def avg_blur(self, image, kernel_size=5):
        blur = cv2.blur(image, (kernel_size, kernel_size))
        return blur

    def cover_eyes(self, image, bbox, kps):
        covered = image.copy()
        radius = int(0.2 * (bbox[2] - bbox[0]))

        for eyes in kps[:2]:
            cv2.circle(covered, (int(eyes[0]), int(eyes[1])), radius, (0,0,0), -1)

        return covered

    def cover_mouth(self, image, bbox, kps):
        covered = image.copy()

        mouth_height = int(0.1 * (bbox[3] - bbox[1])/2)
        mouth_width_increase = int(0.2 * (kps[4][0] - kps[3][0]))

        start = (int(kps[3][0] - mouth_width_increase), int(kps[3][1] - mouth_height))
        end = (int(kps[4][0] + mouth_width_increase), int(kps[4][1] + mouth_height))

        covered = cv2.rectangle(covered, start, end, (0,0,0), -1)

        return covered
