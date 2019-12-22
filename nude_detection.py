# '''from nudenet import NudeClassifier
# # import tensorflow.compat.v1 as tf
# # tf.disable_v2_behavior()
# # classifier = NudeClassifier()
# # #classifier.classify('C:\my_app\python\imageProceccingClassEX4\ImageDetection')
# # classifier.classify('n.jpg')
# #
# # # {'path_to_nude_image': {'safe': 5.8822202e-08, 'unsafe': 1.0}}'''
# #
# # #############
# # import cv2
# # from matplotlib import image
# # from opt_einsum.backends import tensorflow
# #
# # '''from nudenet import NudeDetector
# # detector = NudeDetector()
# #
# # # Performing detection
# # detector.detect('n.jpg')
# # # [{'box': [352, 688, 550, 858], 'score': 0.9603578, 'label': 'BELLY'}, {'box': [507, 896, 586, 1055], 'score': 0.94103414, 'label': 'F_GENITALIA'}, {'box': [221, 467, 552, 650], 'score': 0.8011624, 'label': 'F_BREAST'}, {'box': [359, 464, 543, 626], 'score': 0.6324697, 'label': 'F_BREAST'}]
# #
# # # Censoring an image
# # detector.censor('n.jpg', out_path='n.jpg', visualize=False)'''
# #
# #
# # ##############
# #
# # '''import nudeclient
# # # Single image prediction
# # nudeclient.predict('p_pic.jpg')
# # {'p_pic.jpg': {'safe': 5.8822202e-08, 'unsafe': 1.0}}
# #
# # # Batch predictions
# # nudeclient.predict(['p_pic.jpg', 'n.jpg'])
# # {'p_pic.jpg': {'safe': 5.8822202e-08, 'unsafe': 1.0}, 'n.jpg': {'safe': 5.8822202e-08, 'unsafe': 1.0}}'''
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # from nudenet import NudeDetector
# # import cv2
# # detector = NudeDetector()
# #
# # # Performing detection
# # image_path = r'C:\my_app\python\imageProceccingClassEX4\ImageDetection\n.jpg'
# # img = cv2.imread(image_path)
# # detector.detect(img)
# # # [{'box': [352, 688, 550, 858], 'score': 0.9603578, 'label': 'BELLY'}, {'box': [507, 896, 586, 1055], 'score': 0.94103414, 'label': 'F_GENITALIA'}, {'box': [221, 467, 552, 650], 'score': 0.8011624, 'label': 'F_BREAST'}, {'box': [359, 464, 543, 626], 'score': 0.6324697, 'label': 'F_BREAST'}]
# #
# # # Censoring an image
# # detector.censor(img, out_path='censored_image_path', visualize=False)
# # # Displaying the image
# # cv2.imshow('n', img)



from nudenet import NudeClassifier
classifier = NudeClassifier()
classifier.classify('p_pic.jpg')
# {'path_to_nude_image': {'safe': 5.8822202e-08, 'unsafe': 1.0}}