from nnpcr import NNPCR
model = NNPCR()
model.loadModel('nnmodel.bin')
predictions = model.predict(['image1.jpg', 'image2.jpg', 'image3.jpg'])
print(predictions)