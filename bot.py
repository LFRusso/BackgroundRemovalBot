from telegram.ext import Updater, CommandHandler
import logging
import os
import numpy as np
from unet import crop as unet

MODEL = unet.unet_model.unet()
MODEL.load_weights("unet/unet_people.hdf5")

def start(update, context):
    message = "Hi, @{}! Type /help to see the commands \o/.".format(update.effective_user.username)
    context.bot.send_message(chat_id=update.effective_chat.id, text=message)

def help(update, context):
    message = """
    Comandos: 
    /help: Display commands
    /crop: Use the command replying to a photo with a persont in it
    """    
    context.bot.send_message(chat_id=update.effective_chat.id, text=message)

# Gets you stick bugged
def crop(update, context):
    media = update.message.reply_to_message.photo[0]
    if (media == None): return
    if (media.file_size > 18412): return
    
    context.bot.send_message(chat_id=update.effective_chat.id, text="Loading...")

    media_id = media.file_id
    imgFile = context.bot.getFile(media_id)

    fname = media_id
    imgFile.download(f"tmp/{fname}.jpg")
    unet.crop_img(fname, MODEL)
    os.remove(f"tmp/{fname}.jpg")
    context.bot.sendPhoto(chat_id=update.effective_chat.id, photo=open(f"tmp/out-{fname}.png", 'rb'))
    os.remove(f"tmp/out-{fname}.png")
    return



def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")


    with open('apikey', 'r') as file:
        key = file.readline()
    TOKEN = key[:-1]
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("crop", crop))

    updater.start_polling()
    logging.info("=== It's alive! ===")
    updater.idle()
    logging.info("=== Oh no, It's dying! ===")


if __name__ == "__main__":
    main()