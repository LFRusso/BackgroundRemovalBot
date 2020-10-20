from telegram.ext import Updater, CommandHandler, ConversationHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import logging
import os
import numpy as np
from u2net import crop as u2net
import torch
import os

#MODEL = unet.unet_model.unet()
#MODEL.load_weights("unet/unet_people.hdf5")

MODEL = u2net.U2NET(3,1)
MODEL.load_state_dict(torch.load("u2net/u2net.pth", map_location=torch.device('cpu')))
MODEL.eval()

MEDIA = range(1)

def start(update, context):
    message = "Hi, @{}! Type /help to see the commands \o/.".format(update.effective_user.username)
    context.bot.send_message(chat_id=update.effective_chat.id, text=message)

def help(update, context):
    message = """
    Comandos: 
    /help: Display commands
    /crop: Use the command replying to a photo to remove its background
    """    
    context.bot.send_message(chat_id=update.effective_chat.id, text=message)

def crop(update, context):
    keyboard = [[InlineKeyboardButton(text="Remove the background", callback_data='crop')]]
    markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
    text_id = context.bot.send_message(chat_id=update.effective_chat.id, text="Send a photo and click the button below:", reply_markup=markup)
    context.bot_data['text_id'] = text_id.message_id

    return MEDIA

def get_photo(update, context):
    media = update.message.photo
    if (media == None): return
    if (media.file_size > 84120): 
        context.bot.send_message(chat_id=update.effective_chat.id, text="Photo is too big, try downscaling it")
        return
    
    context.bot_data['media'] = media
    
    return ConversationHandler.END

def crop_query(update, context):
    query = update.callback_query
    query.answer()

    if query.data == 'crop':
        context.bot.edit_message_reply_markup(chat_id=query.message.chat_id, message_id=query.message.message_id)
        text_id = context.bot_data['text_id']
        context.bot.delete_message(chat_id=update.effective_chat.id, message_id=text_id.message_id)
        context.bot.send_message(chat_id=update.effective_chat.id, text="Loading...")

        media = context.bot_data['media']
        media_id = media.file_id
        imgFile = context.bot.getFile(media_id)

        fname = media_id
        imgFile.download(f"tmp/{fname}.jpg")
        u2net.crop_img(fname, MODEL)
        os.remove(f"tmp/{fname}.jpg")
        context.bot.sendDocument(chat_id=update.effective_chat.id, document=open(f"tmp/out-{fname}.png", 'rb'))
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
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("crop", crop)],
        states={MEDIA: get_photo},
        fallbacks=CallbackQueryHandler(crop_query)
    )
    dp.add_handler(conv_handler)

    updater.start_polling()
    logging.info("=== It's alive! ===")
    updater.idle()
    logging.info("=== Oh no, It's dying! ===")


if __name__ == "__main__":
    main()
