import os, pickle, cv2
import face_recognition as fr

from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

###########################################################################################

bot = Bot(token="---------")
dp = Dispatcher(bot, storage=MemoryStorage())
sp = []

async def on_startup(_):
    print("Бот онлайн")

@dp.message_handler(commands=['start'])
async def commands_start(message : types.Message):
    global kb_client
    await message.answer('Подойдите к камере и нажмите кнопку', reply_markup=kb_client)

b = KeyboardButton('/Зарегистрировать')
b1 = KeyboardButton('/Удалить')
kb_client = ReplyKeyboardMarkup(resize_keyboard=True)
kb_client.add(b).add(b1)


otm = KeyboardButton('Отмена')
kb_otm = ReplyKeyboardMarkup(resize_keyboard=True)
kb_otm.add(otm)

cap = cv2.VideoCapture('http://admin:admin@192.168.1.169/snap.jpg') # камера
###########################################################################################


try: # если такой pickle файл существует
    data = pickle.loads(open("data_encodings.pickle", "rb").read()) # загружаем словарь из pickle файла
    informdict = pickle.loads(open("informdict_encodings.pickle", "rb").read())
    id = int((str(list(data.keys())[-1]).split('_'))[-1]) # берем значение из последнего добавленного id

except: # пойдет сюда, когда программа увидит, что нет нужных pickle файлов и выдаст ошибку
    data = {} # создаем новый словарь с кодировками
    informdict = {} # создаем новый словарь со списком ФИО, класс, дата рождения
    id = 0 # счетчик id равен нулю


class FSMAdmin(StatesGroup):
    IMG = State()
    FIO = State()
    Class = State()
    dateb = State()

class FSMDelete(StatesGroup):
    id_delete = State()


@dp.message_handler(commands=['Удалить'], state=None)
async def delete(message : types.Message):
    global kb_otm
    await FSMDelete.id_delete.set()
    await message.answer(informdict)
    await message.answer('Введите ID ученика. В таком формате id_XXXX', reply_markup=kb_otm)

@dp.message_handler(state=FSMDelete.id_delete)
async def deleted(message : types.Message, state=FSMContext):
    global data, informdict, kb_client
    if message.text == "Отмена":
        await state.finish()
        await message.answer('Отмена', reply_markup=kb_client)
    else:
        nm = informdict[message.text]
        del data[message.text]
        del informdict[message.text]
        with open(f"data_encodings.pickle", "wb") as file:
            file.write(pickle.dumps(data))
        with open(f"informdict_encodings.pickle", "wb") as file:
            file.write(pickle.dumps(informdict))
        os.remove(f'facesdb/{message.text}.png')
        print(data)
        print(informdict)
        await message.answer(f'{nm} удален')
        
        await message.answer(f"Измененный словарь:")
        await message.answer(f"{informdict}", reply_markup=kb_client)
        nm = None
        await state.finish()


@dp.message_handler(commands=['Зарегистрировать'], state=None)
async def delete(message : types.Message):
    await FSMAdmin.IMG.set()
    await message.answer("Подойди к камере")


@dp.message_handler(state=FSMAdmin.IMG)
async def load_IMG(message : types.Message, state=FSMContext):
    global kb_client, id, data
    datasp = []
    cap_face = cv2.VideoCapture('http://admin:admin@192.168.1.169/snap.jpg')  # Камера, смотрящая за руками
    ret, img = cap_face.read()
    locations = fr.face_locations(img, model="hog") # получаем список с координатами лица
    if len(locations) != 0: # если список с координатами НЕ пустой
        top, right, bottom, left = locations[0] # задаем откуда до куда будет вырезаться лицо
        top -= 100 # корректируем значения по верху, чтобы вырезалось больше лица
        left -= 25 # корректируем значения по левому, чтобы вырезалось больше лица
        if top > 0 and left > 0: # лицо влезает в кадр
            bottom += 45 # корректируем значения по низу, чтобы вырезалось больше лица
            right += 25 # корректируем значения по правому, чтобы вырезалось больше лица
            await message.answer('Лицо в кадре', reply_markup=kb_client)
            print("Лицо в кадре")
            print(message.text)
            img = img[top:bottom, left:right] # вырезали лицо
            encodings = fr.face_encodings(img) # получаем кодировку лица
            if len(encodings) > 0: # если кодировка получена
                id += 1
                cv2.imwrite(f"facesdb/id_{id:04}.png", img) # Сохроняем фото в папку с лицами
                await message.answer_photo(open(f"facesdb/id_{id:04}.png", 'rb'))
                
                data[f"id_{id:04}"] = encodings[0]
                print(data)
                with open(f"data_encodings.pickle", "wb") as file:
                    file.write(pickle.dumps(data)) # сохроняет измененный словарь в pickle файл
                await FSMAdmin.next()
                await message.answer("Введите ФИО ученика")
            else:
                await message.answer('Кодировка не получена', reply_markup=kb_client)
        else:
            await message.answer('Лицо НЕ полностью в кадре', reply_markup=kb_client)
    else:
        await message.answer('Лицо НЕ в кадре', reply_markup=kb_client)


@dp.message_handler(state=FSMAdmin.FIO)
async def load_FIO(message : types.Message, state=FSMContext):
    global sp
    sp.append(message.text)
    await FSMAdmin.next()
    await message.answer('Введите класс ученика')


@dp.message_handler(state=FSMAdmin.Class)
async def load_Class(message : types.Message, state=FSMContext):
    global sp
    sp.append(message.text)
    await FSMAdmin.next()
    await message.answer('Введите дату рождения ученика')


@dp.message_handler(state=FSMAdmin.dateb)
async def load_dateb(message : types.Message, state=FSMContext):
    global id, informdict, sp, kb_client
    sp.append(message.text)
    informdict[f"id_{id:04}"] = sp
    with open(f"informdict_encodings.pickle", "wb") as file:
        file.write(pickle.dumps(informdict))
    await message.reply(sp)
    sp = []
    print(informdict)
    await state.finish()
    await message.answer('Ученик добавлен', reply_markup=kb_client)


executor.start_polling(dp, skip_updates=True, on_startup=on_startup)
