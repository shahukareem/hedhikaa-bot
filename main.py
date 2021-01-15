import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from fastai.vision import *

def start(update, context):
    update.message.reply_text(
        "Welcome to Hedhikaa Bot. Send a photo of a hedhikaa (Bajiyaa, Gulha, Boakibaa and Masroshi) and this classifier will try to tell what's in the photo \n "
    )


def main():
    load_model()
    updater = Updater(token="token", use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", start))

    dp.add_handler(MessageHandler(Filters.photo, predict))

    updater.start_polling()
    updater.idle()


def load_model():
    global model
    path = os.path.dirname(__file__)
    model = load_learner(path, 'model/hedhika-classifier.pkl')


def predict(update, context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')

    preds, idx, output = model.predict('user_photo.jpg')[0]
    prediction = dict({model.data.classes[i]: round(to_np(p) * 100, 2) for i, p in enumerate(output) if p > 0.2})
    update.message.reply_text(prediction)


if __name__ == '__main__':
    main()


