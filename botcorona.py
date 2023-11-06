import os
import paths
import telebot
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split

bot = telebot.TeleBot(paths.bot_hash)

global img_size
img_size = (75,75)

model = tf.keras.models.load_model(paths.model_path)

###################################################################### 
#mensagem de inicio
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(
        message.chat.id,
        "Bem vindo ao bot de classificação utilizando radiografia do tórax, para utilizar, basta enviar uma imagem, sem compressão de uma radiografia do tórax ao bot e ele te retornará se o paciente possui COVID, infecção pulmonar ou nenhuma doença"
    )

######################################################################
#classificação
def classificacao(message, file_name):
    img = tf.keras.utils.load_img(
        file_name, color_mode='rgb', target_size=(img_size),
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names=["com COVID", "sem COVID, mas possui infecção", "saudável"]

    bot.send_message(
        message.chat.id,
        "O pulmão da imagem {} está {} (certeza de {:.2f}%)"
        .format(message.document.file_name, class_names[np.argmax(score)], 100*np.max(score))
    )

######################################################################
#handler recebimento de foto
def set_path(id, name):
    path = paths.dump
    path = path + str(id) + "-" + name
    return path

@bot.message_handler(content_types=['document','photo'])
def addfile(message):
    if hasattr(message.document, 'file_name'):
        file_name = message.document.file_name
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        file_name = set_path(message.chat.id, file_name)
        with open(file_name, 'wb') as new_file:
            new_file.write(downloaded_file)
        classificacao(message,file_name)
        os.remove(file_name)
    else:
        bot.send_message(message.chat.id, "enviar imagem sem compressao")

######################################################################
print('Bot executando')
bot.polling()