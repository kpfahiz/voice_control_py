import speech_recognition as sr
import pyttsx3 

class Speak:
    def __init__(self):
        # Initialize the recognizer 
        self.recognizer = sr.Recognizer() 
    
    
    def SpeakText(self, command):
        """
        Function to convert text to speech
        """
        """Initialize the engine"""
        engine = pyttsx3.init()
        engine.say(command) 
        engine.runAndWait()
    
    def record(self):
        """Loop infinitely for user to speak"""

        while(1):
            # Exception handling to handle
            # exceptions at the runtime
            try:
                # use the microphone as source for input.
                with sr.Microphone() as source2:
                    # wait for a second to let the recognizer
                    # adjust the energy threshold based on
                    # the surrounding noise level 
                    self.recognizer.adjust_for_ambient_noise(source2, duration=0.2)
            
                    #listens for the user's input 
                    audio2 = self.recognizer.listen(source2)
            
                    # Using google to recognize audio
                    MyText = self.recognizer.recognize_google(audio2)
                    MyText = MyText.lower()

                    print('Did you say ',MyText)
                    self.SpeakText(MyText)
            
            except sr.RequestError as e:
                print(f'Could not request results; {e}')
        
            except sr.UnknownValueError:
                print('unknown error occurred')


def main():
    spk = Speak()
    spk.record()

if __name__ == '__main__':
    main()
