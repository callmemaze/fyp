from gtts import gTTS

def speech_to_text():
    mytext = "Valid qr code, proceeding"
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("sound/qr.mp3")
