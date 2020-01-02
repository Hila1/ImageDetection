# import nude
# from nude import Nude
#
# path = "222.jpg"
# print(nude.is_nude(path))
#
# n = Nude(path)
# n.parse()
# print("damita :", n.result, n.inspect())

import Algorithmia

input = "C:\my_app\imageProcessing_FinalProject\\222.jpg"
client = Algorithmia.client('YOUR_API_KEY')
algo = client.algo('sfw/NudityDetection/1.1.6')
algo.set_options(timeout=300) # optional
print(algo.pipe(input).result)