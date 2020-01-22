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


def check_drugs_content(image_path):
    return 0


if __name__ == "__main__":
    # image path
    path = r'/images_for_testing/p3.jpg'
    image_data = [check_pornographic_content(path), check_drugs_content(path)]
    # for result in image_data:
    #   if result[0] == true:
    blurred_image = blur_image(path)
    # to be deleted
    # show the image
    plt.imshow(blurred_image)
    plt.show()
