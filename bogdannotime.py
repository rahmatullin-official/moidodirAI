from collections import Counter
import cv2
import mediapipe as mp
import numpy as np
import time
import datetime
import sqlite3
import face_recognition as fr
import pickle

# from keras.models import load_model
from PIL import Image, ImageOps
import tensorflow

CAMERA_MAIN = 'http://admin:admin@192.168.1.100/snap.jpg'
CAMERA_FACES = 'http://admin:admin@192.168.1.169/snap.jpg'


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
today = datetime.datetime.today()

hands = mp.solutions.hands.Hands(max_num_hands=2)  # Объект ИИ для определения ладони
draw = mp.solutions.drawing_utils  # Для рисование ладони
now = datetime.datetime.now()

# Загрузка модели -----------------------------------------------------------------
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# ---------------------------------------------------------------------------------

# Глобальные переменные-------------------------
facesp = list()

zn = {  # словарь поз для нейронки, !не для "учёт поз"!
    "0": "pose1",
    "1": "pose2",
    "2": "pose3",
    "3": "pose4",
    "4": "pose5",
    "5": "soap",
    "6": "non"}


# ----------------------------------------------




def detect_person_in_video(cap_face, namesk, coding): # функция для обнаружения лица и определения личности
    global facesp
    cap_face = cv2.VideoCapture(CAMERA_FACES)  # Камера, смотрящая за руками
    ret, image = cap_face.read()
    result = None
    # image = cv2.resize(image, xy)
    if ret: # если кадр получен
        locations = fr.face_locations(image, model="hog")
        encodings = fr.face_encodings(image, locations) # кодировки получаемого лица
        match = None
        for face_encoding in encodings:
            result = fr.compare_faces(coding, face_encoding)  # список, который показывает с какими лицами схоже получаемое лицо

            res = list(fr.face_distance(coding, face_encoding))  # список со значением, на сколько получаемое лицо схоже с добавленными
            # print(res)
            # print(max(res))
            ind = res.index(min(res))

            if result[ind] == True:
                match = namesk[ind]
                facesp.append(match)
                print(f"match {match}")
            else:
                print("DONT KNOW((")

        # cv2.imshow("detect_person", image)
        cv2.waitKey(1)


def persons_name(): # функция для определения имени
    global facesp
    sl = Counter(facesp)  # словарь с количеством повторений имен
    slkeys = list(sl.keys())  # имена
    slvalues = list(sl.values())  # сколько раз определило
    print(facesp)
    if len(facesp) > 0:
        return slkeys[slvalues.index(max(slvalues))]
    else:
        return "Неопознанный"


def coefficient(quality):  # функция считает итоговый коэффициент качества мытья рук
    it = 0
    if quality['soap']:
        it += 0.40
    if quality['pose1']:
        it += 0.12
    if quality['pose2']:
        it += 0.12
    if quality['pose3']:
        it += 0.12
    if quality['pose4']:
        it += 0.12
    if quality['pose5']:
        it += 0.12
    return round(it, 2)


def answer(pred):  # Возвращает красивый ответ нейросети
    global zn
    it = np.argmax(pred)
    if np.max(pred) < 0.50:  # коэфициент уверенности в ответе нейронки
        return 'no_sure'
    return zn[str(it)]


# def cv2pil(img):  # делает из cv2 фото - PIL фото для нейолнки
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     im_pil = Image.fromarray(img)
#     return im_pil


def neyroset(img):  # Основная функция нейросети, определят позу рук
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    return prediction

def main(): # основная часть
    global id, face_result, countface, facesp
    handst = False
    sec = None
    start_time = None

    # ------------------------------------------------------------------
    cap_hands = cv2.VideoCapture(CAMERA_MAIN)  # Камера, смотрящая за руками
    cap_face = cv2.VideoCapture(CAMERA_FACES)  # Камера, смотрящая за лицом
    # ------------------------------------------------------------------

    data = pickle.loads(open("data_encodings.pickle", "rb").read())
    informdict = pickle.loads(open("informdict_encodings.pickle", "rb").read())
    namesk = list(data.keys())  # список всех ключей-имен
    coding = list(data.values())  # список со всеми кодировками
    # xy = (image.shape[1] // 2, image.shape[0] // 2)

    quality = {  # словарь отображает какие поз рук присутствовали во время мытья, называется - "учет поз"
        'pose1': False,
        'pose2': False,
        'pose3': False,
        'pose4': False,
        'pose5': False,
        'soap': False,
        'no_sure': False
    }
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5) as hands:
        ii = 0
        while True:
            cap_hands = cv2.VideoCapture(CAMERA_MAIN)  # Камера, смотрящая за руками
            success, image = cap_hands.read()
            # image = image[0:720, 450:1280]  # обрезка
            if not success:
                print("Ignoring empty camera frame.")  # кадр не получен
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            # image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            backgr = np.zeros([image.shape[0], image.shape[1]], dtype='uint8')
            backgr = cv2.cvtColor(backgr, cv2.COLOR_BGR2RGB)

            if results.multi_hand_landmarks:  # Руки в кадре
                print(detect_person_in_video(cap_face, namesk, coding))

                start_time = None
                if sec is None:
                    sec = time.time()
                    # first = False
                    handst = True  # руки были

                print(pred)
                quality[pred] = True

            elif handst:  # Если руки были
                if start_time is None:
                    print("Время пошло")
                    start_time = time.time()
                    # first = True
                # print("ЖДЕТ")
                print(detect_person_in_video(cap_face, namesk, coding))
                pred = neyroset(image)
                print(pred)
                if pred == 'soap':

                    quality[pred] = True
                if not (start_time is None):
                    if time.time() - start_time > 4:  # если рук в кадре нет дольше 4 секунд
                        id = persons_name()
                        print(id)
                        if id != "Неопознанный":
                            name_human = informdict[id][0]
                            Class = informdict[id][1]
                        else:
                            name_human = id
                            Class = id
                        print("Прошло 4 секунды")
                        now = datetime.datetime.now()
                        print(id, name_human, Class, date := now.strftime("%d.%m.%Y"),
                              ttime := now.strftime("%H:%M:%S"), cffc := int(coefficient(quality) * 100),
                              timego := round(time.time() - sec - 4))  # заносить в базу данных
                        cursor.execute(f"""INSERT INTO albums
                  VALUES ('{id}', '{name_human}', '{Class}',
                  '{date} {ttime}', '{cffc}', {timego})"""
                                       )

                        conn.commit()
                        quality = {  # человек смениося - обнуляем его "учет поз"
                            'pose1': False,
                            'pose2': False,
                            'pose3': False,
                            'pose4': False,
                            'pose5': False,
                            'soap': False,
                            'tmp': False
                        }
                        sec = None
                        handst = False
                        start_time = None
                        face_result = "Неопознанный"
                        facesp = []

            cv2.imshow('Hands', cv2.flip(image, 1))
            cv2.imwrite(r'C:\Users\SENT\PycharmProjects\pythonProject\HCCS_new\data2201\img' + str(ii) + '.jpg', image)
            ii += 1
            if cv2.waitKey(1) & 0xFF == 27:
                break


if __name__ == '__main__':
    main()