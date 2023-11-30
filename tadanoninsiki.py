# -*- coding: utf-8 -*-
from ultralytics import YOLO
model_file = "1_best.pt"
pic_file = "3.png"
limit_conf = 0.1

model = YOLO(model_file)
result = model.predict(pic_file, save=True, conf=limit_conf)

#print(result[0].save_dir)

txt_dir = result[0].save_dir + '\conf.txt'
with open(txt_dir, 'w', encoding='utf-8') as f:
    f.write("モデル名:" + model_file + "\n")
    f.write("conf=" + str(limit_conf) + "\n")
    f.close()