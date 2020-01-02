import nude
from nude import Nude
path="C:\my_app\imageProcessing_FinalProject\\view.jpg"

print(nude.is_nude(path))

n = Nude(path)
n.parse()
print("damita :", n.result, n.inspect())