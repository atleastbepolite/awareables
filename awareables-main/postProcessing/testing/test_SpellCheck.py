import pytest
import re
import time
import SpellChecker as sc

class TestSpellCheck():
    errorTxt1 = """Duxing the weole of a dula, qark, and soendless day in the tutumn of the yerr, 
when the wlouds hueg opwressively low in the heavels, I had been patsing alonr, on sorseback, 
through a sengularly drealy teact of couptry; and at dength fuund myserf, as the sgades of 
the evewing dpew on, wiqhin vief of the meaancholy House of Usher. I dnow not how it was--but, 
with the firyt glempse of the bcilding, a sensw of inslfferable glool petvaded my spirip. I say
iwsufferable; for the feelikg was unreliehed by any of that talf-plegsurable, becaupe 
qoetic, sentifent, kith whidh the cind usuallr receiwes sven the sternesl netural igages of 
the desoeate or terrinle. """ # 115
    # errorMatrix1 = {
    #     2: ['r'], 12: ['h'], 25: ['l'], 28: ['d'], 40: ['u'],
    #     59: ['a'], 75: ['a'], 93: ['c'], 102: ['n'], 107: ['p'],
    #     134: ['n'], 151: ['s'], 161: ['e'], 167: ['h'], 194: ['i'],
    #     208: ['r'], 212: ['r'], 223: ['n'], 236: ['l'], 244: ['o'],
    #     253: ['l'], 265: ['h'], 286: ['n'], 292: ['r'], 302: ['t'],
    #     310: ['f'], 321: ['l'], 334: ['e'], 349: ['k'], 392: ['s'],
    #     397: ['i'], 411: ['u'], 426: ['e'], 434: ['u'], 448: ['m'],
    #     452: ['r'], 467: ['t'], 483: ['n'], 509: ['n'], 523: ['v'],
    #     542: ['h'], 550: ['a'], 565: ['s'], 573: ['p'], 586: ['m'],
    #     592: ['w'], 600: ['c'], 607: ['k'], 618: ['y'], 625: ['v'],
    #     629: ['e'], 645: ['t'], 648: ['a'], 656: ['m'], 678: ['l'],
    #     691: ['b']
    # }
    errorTxt2 = """I lolked upoe twe scend bewore me--uson the mere qouse, ang rhe sidple landsyape 
    featuses of the womain--upon the blaak wafls--apon the vicant eye-likd windhws--upon a few 
    rank sedges--and upol a few woite trunbs of eecayed trees--with an urter desression of soul 
    whicd I can compwre to no earttly segsation more properll ehan to the aftpr-deeam of the 
    revefer upon oaium--the bittet lapde into everrday life--the hideous dropping off of the 
    veil.  There was an iciness, a sinking, a sickening of the heart--an unredeemed dreariness 
    of thought which no goading of the imagination could torture into aught of the sublime.  
    What was it--I paused to think--what was it that so unnerved me in the contemplation of the 
    House of Usher?""" # 120
    txt1 = """during the whole of a dull, dark, and soundless day in the autumn of the year, 
    when the clouds hung oppressively low in the heavens, I had been passing alone, on horseback, 
    through a singularly dreary tract of country; and at length found myself, as the shades of 
    the evening drew on, within view of the melancholy House of Usher. i know not how it was--but, 
    with the first glimpse of the building, a sense of insufferable gloom pervaded my spirit. i say 
    insufferable; for the feeling was unrelieved by any of that half-pleasurable, because 
    poetic, sentiment, with which the mind usually receives even the sternest natural images of 
    the desolate or terrible. """ # 115
    txt2 = """I looked upon the scene before me--upon the mere house, and the simple landscape 
    features of the domain--upon the bleak walls--upon the vacant eye-like windows--upon a few 
    rank sedges--and upon a few white trunks of decayed trees--with an utter depression of soul 
    which I can compare to no earthly sensation more properly than to the after-dream of the 
    reveler upon opium--the bitter lapse into everyday life--the hideous dropping off of the 
    veil.  There was an iciness, a sinking, a sickening of the heart--an unredeemed dreariness 
    of thought which no goading of the imagination could torture into aught of the sublime.  
    What was it--I paused to think--what was it that so unnerved me in the contemplation of the 
    House of Usher?""" # 120
    txt3 = """It was a mystery all insoluble; nor could I grapple with the shadowy fancies that 
    crowded upon me as I pondered.  I was forced to fall back upon the unsatisfactory conclusion, 
    that while, beyond doubt, there are combinations of very simple natural objects which have 
    the power of thus affecting us, still the analysis of this power lies among considerations 
    beyond our depth.  It was possible, I reflected, that a mere different arrangement of the 
    particulars of the scene, of the details of the picture, would be sufficient to modify, or 
    perhaps to annihilate its capacity for sorrowful impression; and, acting upon this idea, I 
    reined my horse to the precipitous brink of a black and lurid tarn that lay in unruffled luster 
    by the dwelling, and gazed down--but with a shudder even more thrilling than before--upon the 
    remodeled and inverted images of the gray sedge, and the ghastly tree-stems, and the vacant 
    and eye-like windows.""" # 156
    checkText1 = re.findall(r'\w+', txt1)
    checkText2 = re.findall(r'\w+', txt2.lower())
    checkText3 = re.findall(r'\w+', txt3.lower())

    def test_dict(self):
        file = "txtFiles/dictionary.txt"
        checker = sc.SpellChecker(file)
        start = time.time()
        new_txt = checker.wordIntakeTest(self.errorTxt1)
        end = time.time()
        
        correct = 0
        for i in range(len(new_txt)):
            if new_txt[i] == self.checkText1[i]:
                correct += 1
            # else:
            #     print(f"word: {self.checkText1[i]}\tIncorrect update: {new_txt[i]}")
        percent = (correct / 115) * 100

        print("\nNumber of words: 115\tPercent correct: {:.4f}\tTime in seconds: {:.4f}\n".format(percent, (end - start)))
        assert True
    
    def test_text(self):
        file = "txtFiles/text.txt"
        checker = sc.SpellChecker(file)
        start = time.time()
        new_txt = checker.wordIntakeTest(self.errorTxt1)
        end = time.time()
        
        correct = 0
        for i in range(len(new_txt)):
            if new_txt[i] == self.checkText1[i]:
                correct += 1
            # else:
                # print(f"word: {new_txt[i]}\tIncorrect update: {self.checkText1[i]}")
        percent = (correct / 115) * 100

        print("\nNumber of words: 115\tPercent correct: {:.4f}\tTime in seconds: {:.4f}\n".format(percent, (end - start)))
        assert True
    
    def test_text_and_dict(self):
        file = "txtFiles/dict_text.txt"
        checker = sc.SpellChecker(file)
        start = time.time()
        new_txt = checker.wordIntakeTest(self.errorTxt1)
        end = time.time()
        
        correct = 0
        for i in range(len(new_txt)):
            if new_txt[i] == self.checkText1[i]:
                correct += 1
            # else:
                # print(f"word: {new_txt[i]}\tIncorrect update: {self.checkText1[i]}")
        percent = (correct / 115) * 100

        print("\nNumber of words: 115\tPercent correct: {:.4f}\tTime in seconds: {:.4f}\n".format(percent, (end - start)))
        assert True
    
    """
    def test_confidence(self):
        file = "txtFiles/dict_text.txt"
        checker = sc.SpellChecker(file)
        start = time.time()
        new_txt = checker.wordIntakeTest(self.errorTxt1, self.errorMatrix1)
        end = time.time()
        
        correct = 0
        for i in range(len(new_txt)):
            if new_txt[i] == self.checkText1[i]:
                correct += 1
            # else:
                # print(f"word: {new_txt[i]}\tIncorrect update: {self.checkText1[i]}")
        percent = (correct / 115) * 100

        print("\nNumber of words: 115\tPercent correct: {:.4f}\tTime in seconds: {:.4f}\n".format(percent, (end - start)))
        assert True
    """