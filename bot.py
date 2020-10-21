from telegram.ext import Updater, CommandHandler, ConversationHandler, CallbackQueryHandler, MessageHandler, Filters
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

# Starting the bot
def start(update, context):
    message = "Hi, @{}! Type /help to see the commands \o/.".format(update.effective_user.username)
    context.bot.send_message(chat_id=update.effective_chat.id, text=message)

    return

# Displays available commands
def help(update, context):
    message = """
    Commands:
    /help: Display commands
    /crop: Send a photo then click the button to remove its background
    """    
    context.bot.send_message(chat_id=update.effective_chat.id, text=message)

    return

# Calling the crop function, which then goes to the get_photo function
def crop(update, context):
    message = "Send a photo"
    context.bot.send_message(chat_id=update.effective_chat.id, text=message)
    
    return MEDIA

# Saves the photo sent by the user and goes to the crop_query function, which processes the image
def get_photo(update, context):
    media = context.bot.get_file(update.message.photo[-1]) # Gets photo sent by user
    
    # Checking if a photo was sent and, if so, its size
    if (media == None): return
    if (media.file_size > 84120): 
        error = "Photo is too big, try downscaling it"
        context.bot.send_message(chat_id=update.effective_chat.id, text=error)
        return
    
    keyboard = [[InlineKeyboardButton(text="Remove", callback_data='crop')]]
    markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    prompt = "Click the button below to start the background removal"
    text = context.bot.send_message(chat_id=update.effective_chat.id, text=prompt, reply_markup=markup)
    context.bot_data['text_id'] = text.message_id # Saving the text id for posterior exclusion
    context.bot_data['media'] = media # Saving the photo data for further processing
    
    return MEDIA

# Processing image sent by user
def crop_query(update, context):
    query = update.callback_query
    query.answer()

    if query.data == 'crop':
        context.bot.edit_message_reply_markup(chat_id=query.message.chat_id, message_id=query.message.message_id) # Deleting the button
        text_id = context.bot_data['text_id']
        context.bot.delete_message(chat_id=update.effective_chat.id, message_id=text_id) # Deleting the message attached to the button
        context.bot.send_message(chat_id=update.effective_chat.id, text="Loading...")

        # Saving and processing the image
        media_id = context.bot_data['media'].file_id
        imgFile = context.bot.getFile(media_id)

        fname = media_id
        imgFile.download(f"tmp/{fname}.jpg")
        u2net.crop_img(fname, MODEL) # Cropping the background
        os.remove(f"tmp/{fname}.jpg") # Removing original image
        context.bot.sendDocument(chat_id=update.effective_chat.id, document=open(f"tmp/out-{fname}.png", 'rb'))
        os.remove(f"tmp/out-{fname}.png") # Removing cropped image
    
    return ConversationHandler.END


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
    
    # Handler for the crop function, calling what's needed in a definite order
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("crop", crop)], # User types '/crop'
        states={MEDIA: [MessageHandler(Filters.photo, get_photo)]}, # User sends an image
        fallbacks=[CallbackQueryHandler(crop_query)] # Image is processed
    )
    dp.add_handler(conv_handler)

    updater.start_polling()
    logging.info("=== It's alive! ===")
    updater.idle()
    logging.info("=== Oh no, It's dying! ===")


if __name__ == "__main__":
    main()
