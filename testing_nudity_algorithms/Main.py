# this class is using the 'nudity' lib. its being used for detecting whether an image is porn, nudity or other human inappropriate content.
# this class takes an image path and returns its offensiveness value (between 0-1),
# it also blur the image in case its offensive
from nudity import Nudity
import matplotlib.pyplot as plt
import cv2


def check_pornographic_content(image_path):
    nudity = Nudity()
    # returns True or False
    is_offensive = nudity.has(image_path)
    # gives nudity / inappropriate content score between 0.0 - 1.0
    score = nudity.score(image_path)

    print(is_offensive)
    if score < 0.0001:
        score = 0.0
    print(score)
    return [is_offensive, score]


def blur_image(image_path):

    img = cv2.imread(image_path)
    image = cv2.blur(img, (40, 40))
    return image


if __name__ == "__main__":
    # image path
    path = r'/images_for_testing/image.jpg'
    image_to_show = cv2.imread(image_path)
    image_data = check_pornographic_content(path)
    if image_data[0]:
        image_to_show = blur_image(path)
    # show the image
    plt.imshow(image_to_show)
    plt.show()
