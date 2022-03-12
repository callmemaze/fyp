from gtts import gTTS
from playsound import playsound

name = "Dipesh"

def speech_to_text(text):
    mytext = "Welcome,{} unlocking door.The door will remain open for the next 5 seconds".format(name)
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("sound/welcome.mp3")

playsound('sound/welcome.mp3')