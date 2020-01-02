from nudity import Nudity

nudity = Nudity()
path = "C:\my_app\imageProcessing_FinalProject\\final\\111.jpg"
print(nudity.has(path))
# returns True or False

print(nudity.score(path))
# gives nudity score between 0.0 - 1.0
