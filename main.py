import logging
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from fastai.vision import *
from dotenv import load_dotenv
import os
from uuid import uuid4

path = os.path.dirname(__file__)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

load_dotenv()
token = os.getenv("bot_token")
url = os.getenv("bot_url")
port = int(os.getenv("PORT", 5000))
is_webhook = (os.getenv("enable_webhook", "False") == "True")  # for testing, set to false to use polling

logging.info(f"Loading model from {path}")
model = load_learner(path, 'model/hedhika-classifier.pkl')
bot = telegram.Bot(token=token)


def start(update, context):
    update.message.reply_text(
        "Welcome to Hedhikaa Bot. Send a photo of hedhikaa (Bajiyaa, Gulha, Boakibaa or Masroshi) and this classifier will tell what's in the photo \n "
    )


def predict(update, context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()

    # generate random filname and download
    f_name = f"{uuid4()}.jpg"
    photo_file.download(f_name)

    # buzz the user
    logger.info("Photo of %s: %s", user.first_name, f_name)
    update.message.reply_text(
        'Got your photo 👍 please wait!'
    )

    # process the image
    img = open_image(f_name)

    preds, idx, output = model.predict(img)
    prediction = dict({model.data.classes[i]: round(to_np(p) * 100, 2) for i, p in enumerate(output) if p > 0.2})
    for key, value in prediction.items():
        result = "Likely " + key + " with " + str(value) + " percentage"
        update.message.reply_text(result)

    # remove the temp file
    os.remove(f_name)


def main():
    updater = Updater(token=token, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", start))

    dp.add_handler(MessageHandler(Filters.photo, predict))

    if is_webhook:
        hook_url = f"{url}{token}"
        updater.start_webhook(
            listen="127.0.0.1",
            port=port,
            url_path=token,
            webhook_url=hook_url)
        updater.bot.set_webhook(hook_url)
        updater.idle()
    else:
        updater.start_polling()
        updater.idle()


if __name__ == '__main__':
    main()


