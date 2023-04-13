from retinaface.pre_trained_models import get_model
import cv2

image = cv2.imread('./images/source.png')
model = get_model("resnet50_2020-07-20", max_size=2048)
model.eval()
annotation = model.predict_jsons(image)
landmarks = annotation[0]['landmarks']

print(landmarks)