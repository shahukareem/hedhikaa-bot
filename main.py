import logging
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from fastai.vision import *
from dotenv import load_dotenv
from os import getenv

path = os.path.dirname(__file__)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)
load_dotenv()
token = getenv("bot_token")
url = token = getenv("bot_url")
bot = telegram.Bot(token=token)
model = load_learner(path, 'model/hedhika-classifier.pkl')


def start(update, context):
    update.message.reply_text(
        "Welcome to Hedhikaa Bot. Send a photo of hedhikaa (Bajiyaa, Gulha, Boakibaa or Masroshi) and this classifier will tell what's in the photo \n "
    )


def main():
    updater = Updater(token=token, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", start))

    dp.add_handler(MessageHandler(Filters.photo, predict))

    updater.start_webhook(listen="0.0.0.0",
                          port=int(port),
                          url_path=token)
    updater.bot.setWebhook('https://hedhikaa-bot.herokuapp.com/' + token)
    updater.idle()


def predict(update, context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
    update.message.reply_text(
        'Got your photo 👍 please wait!'
    )

    img = open_image('user_photo.jpg')

    preds, idx, output = model.predict(img)
    prediction = dict({model.data.classes[i]: round(to_np(p) * 100, 2) for i, p in enumerate(output) if p > 0.2})
    for key, value in prediction.items():
        result = "Likely " + key + " with " + str(value) + " percentage"
        update.message.reply_text(result)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    main()


