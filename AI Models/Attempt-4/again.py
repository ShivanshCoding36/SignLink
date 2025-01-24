import cv2
import numpy as np
from keras.models import model_from_json
import operator
from string import ascii_uppercase



directory = 'model/'

ct = {}
ct['blank'] = 0
# Load the model for basic gestures
json_file = open(directory + "model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights(directory + "model-bw.h5")

# Load the model for D-R-U gestures
json_file_dru = open(directory + "model-bw_dru.json", "r")
model_json_dru = json_file_dru.read()
json_file_dru.close()
loaded_model_dru = model_from_json(model_json_dru)
loaded_model_dru.load_weights(directory + "model-bw_dru.h5")

# Load the model for T-K-D-I gestures
json_file_tkdi = open(directory + "model-bw_tkdi.json", "r")
model_json_tkdi = json_file_tkdi.read()
json_file_tkdi.close()
loaded_model_tkdi = model_from_json(model_json_tkdi)
loaded_model_tkdi.load_weights(directory + "model-bw_tkdi.h5")

# Load the model for S-M-N gestures
json_file_smn = open(directory + "model-bw_smn.json", "r")
model_json_smn = json_file_smn.read()
json_file_smn.close()
loaded_model_smn = model_from_json(model_json_smn)
loaded_model_smn.load_weights(directory + "model-bw_smn.h5")


def predict(self, test_image):
    test_image = cv2.resize(test_image, (128,128))
    result = loaded_model.predict(test_image.reshape(1, 128, 128, 1))
    result_dru = loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))
    result_tkdi = loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))
    result_smn = loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))
    prediction = {}
    prediction['blank'] = result[0][0]
    inde = 1
    for i in ascii_uppercase:
        prediction[i] = result[0][inde]
        inde += 1
    # LAYER 1
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    current_symbol = prediction[0][0]
    # LAYER 2
    if(current_symbol == 'D' or current_symbol == 'R' or current_symbol == 'U'):
        prediction = {}
        prediction['D'] = result_dru[0][0]
        prediction['R'] = result_dru[0][1]
        prediction['U'] = result_dru[0][2]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]

    if(current_symbol == 'D' or current_symbol == 'I' or current_symbol == 'K' or current_symbol == 'T'):
        prediction = {}
        prediction['D'] = result_tkdi[0][0]
        prediction['I'] = result_tkdi[0][1]
        prediction['K'] = result_tkdi[0][2]
        prediction['T'] = result_tkdi[0][3]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]

    if(current_symbol == 'M' or current_symbol == 'N' or current_symbol == 'S'):
        prediction1 = {}
        prediction1['M'] = result_smn[0][0]
        prediction1['N'] = result_smn[0][1]
        prediction1['S'] = result_smn[0][2]
        prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
        if(prediction1[0][0] == 'S'):
            current_symbol = prediction1[0][0]
        else:
            current_symbol = prediction[0][0]
    if(current_symbol == 'blank'):
        for i in ascii_uppercase:
            ct[i] = 0
    ct[self.current_symbol] += 1
    if(ct[self.current_symbol] > 60):
        for i in ascii_uppercase:
            if i == current_symbol:
                continue
            tmp = ct[self.current_symbol] - self.ct[i]
            if tmp < 0:
                tmp *= -1
            if tmp <= 20:
                ct['blank'] = 0
                for i in ascii_uppercase:
                    ct[i] = 0
                return
        ct['blank'] = 0
        for i in ascii_uppercase:
            ct[i] = 0
        if current_symbol == 'blank':
            if blank_flag == 0:
                blank_flag = 1
                if len(str) > 0:
                    str += " "
                str += self.word
                word = ""
        else:
            if(len(str) > 16):
                str = ""
            blank_flag = 0
            word += current_symbol

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame.
    ret, frame = cap.read()
    predicted_class = predict(frame)
    cv2.putText(frame, '{}'.format(predicted_class), 
                                    (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Window',frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break