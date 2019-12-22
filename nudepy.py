import nude
from nude import Nude

print(nude.is_nude('C:\my_app\python\imageProceccingClassEX4\ImageDetection\elefant.jpg'))

n = Nude('C:\my_app\python\imageProceccingClassEX4\ImageDetection\elefant.jpg')
n.parse()
print("damita :", n.result, n.inspect())