from nudity import Nudity

nudity = Nudity()
path = r'C:\Users\Student\PycharmProjects\ImageDetection\images_for_testing\np3.jpg'
# returns True or False
is_offensive = nudity.has(path)
# gives nudity / inappropriate content score between 0.0 - 1.0
score = nudity.score(path)

print(is_offensive)
if score < 0.00000001:
    score = 0.0
print(score)

