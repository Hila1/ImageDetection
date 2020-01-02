from nudity import Nudity
nudity = Nudity()

path = "C:\my_app\imageProcessing_FinalProject\\final\\np_pic2.jpg"

print(nudity.has(path))
# gives you True or False

print(nudity.score(path))
# gives you nudity score 0 - 1