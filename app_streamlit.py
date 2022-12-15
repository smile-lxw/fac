import streamlit as st
import numpy as np
import pandas as pd
import face_recognition

import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model

# 下面的代码用来创建神经网络
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(128,)))

model.add(layers.Dense(768))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.1))

model.add(layers.Dense(512))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.1))

model.add(layers.Dense(256))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.1))

model.add(layers.Dense(128))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.1))

model.add(layers.Dense(63))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.1))

model.add(layers.Dense(4))
model.add(layers.BatchNormalization())
model.add(layers.Activation('sigmoid'))


model.load_weights('model20221128csv1-20acc-4-79-3-18-he.h5')


def main():
    st.title('性格罗盘测试')
    img_file = st.file_uploader('请上传相片')
    if img_file:
        stringio = img_file.getvalue()
        img_file_path = 'temp/' + img_file.name
        with open(img_file_path, 'wb') as f:
            f.write(stringio)
        st.image(img_file_path)
        if st.button('开始进行性格识别'):
            with st.spinner('图像识别中...'):
                img = face_recognition.load_image_file(img_file_path)
                output = face_recognition.face_encodings(img, model='large')[0] * 10
                output = output.reshape(1, 128)
                pred = model.predict(output) * 100
                W = round(pred[0, 0])
                C = round(pred[0, 1])
                L = round(pred[0, 2])
                G = round(pred[0, 3])
                data = pd.DataFrame(np.array([W, C, L, G]), index=['W', 'C', 'L', 'G'], columns=['指数值'])
                st.write(data)
            st.success('识别完成！')


if __name__ == '__main__':
    main()
