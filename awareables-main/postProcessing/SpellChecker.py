import re
from collections import Counter


class SpellChecker():
    def __init__(self, file):
        self.dictionary = Counter(self.words(open(file).read()))
        self.offset = 0
        self.text = ""
        self.confidenceMatrix = {}

    def words(self, text): 
        return re.findall(r'\w+', text)

    """
    Test rendition of word intake that returns set of corrected words to be compared
    to correct words. 
    """
    def wordIntakeTest(self, text, confidenceMatrix={}):
        self.offset = 0 # restart
        self.text = text
        self.confidenceMatrix = confidenceMatrix
        # Set of words in the provided text
        initialSet = self.words(text)

        # Set of additional characters between words
        extraCharSet = re.findall(r'\W+', text)
        # Output
        fullSet = [] # text plus characters
        finalSet = [] # just text to check

        for word in initialSet:
            # check prev char
            newWord = self.correctWord(word.lower()) #.lower())
            
            # Proper Noun CHECK
            if (word[0].lower() != word[0]):
                if (len(fullSet) > 0 and fullSet[-1] != '. '):
                    newWord = word # skip proper nouns
            
            
            
            self.offset += len(newWord) # keep track of index in text
            finalSet.append(newWord)
            fullSet.append(newWord)
            try:
                charAdded = extraCharSet.pop(0)
                self.offset += len(charAdded) # keep track of index
                fullSet.append(charAdded)

            except:
                print("No more additional characters.\n")

        return finalSet
    """
    SpellChecker intake function that is given string of text to
    be corrected and returned. 
    """
    def wordIntake(self, text, confidenceMatrix={}): # add confidence matrix
        self.offset = 0 # restart
        self.text = text
        self.confidenceMatrix = confidenceMatrix
        # Set of words in the provided text
        initialSet = self.words(text)
        # Set of additional characters between words
        extraCharSet = re.findall(r'\W+', text)
        # Output
        finalSet = []

        for word in initialSet:

            newWord = self.correctWord(word.lower())

            # Proper Noun CHECK
            if (word[0].lower() != word[0]):
                if (len(finalSet) > 0 and finalSet[-1] != '. '):
                    newWord = word # skip proper nouns

            self.offset += len(newWord) # keep track of index in text
            finalSet.append(newWord)
            try:
                charAdded = extraCharSet.pop(0)
                self.offset += len(charAdded) # keep track of index
                finalSet.append(charAdded)
            except:
                print("No more additional characters.\n")

        result = "".join(finalSet)
        return result
        

    """
    CorrectWord takes in the word to be corrected and returns the
    most probable spelling correction.
    """
    def correctWord(self, word): 
        # print(f"{self.offset}: {word}: {self.text[self.offset]}")
        return (max(self.candidates(word), key=self.probability))

    """
    Candidates generates the set of single error corrections for
    'word' if it is not already a valid word. If there are no single
    edits and it is not a valid word, it will still return that word. 
    """
    def candidates(self, word): 
        confidenceSet = self.known(self.confidenceWords(word))
        inDictSet  = self.known([word])
        if (not confidenceSet): 
            oneEditSet = self.known(self.newWords(word))
        return (confidenceSet or inDictSet or oneEditSet or [word])

    def known(self, words): 
        "The subset of words that appear in the dictionary."
        return set(w for w in words if w in self.dictionary)

    """
    Probability calculates a words probability based on how frequently
    it is used in the english language (approximately).
    """
    def probability(self, word): 
        "Probability of word"
        N = sum(self.dictionary.values())
        return self.dictionary[word] / N

    """
    confidenceWords provides set of words created from inserting characters
    from provided confidenceMatrix into existing words given the index of 
    most likely error. 
    """
    def confidenceWords(self, word):
        if not self.confidenceMatrix: return {} # No matrix provided

        pairs = [] # temporary set of pairs
        options = [] # current set of confidence words

        for i in range(len(word)):
            tmpKey = self.offset + i
            if tmpKey in self.confidenceMatrix.keys():
                if options:
                    for option in options:
                        pair = (option[:i], option[i:])
                        for letter in self.confidenceMatrix[tmpKey]:
                            pairs.append(pair[0] + letter + pair[1][1:])
                    options = pairs
                else:
                    pair = (word[:i], word[i:])
                    for letter in self.confidenceMatrix[tmpKey]:
                        options.append(pair[0] + letter + pair[1][1:])       
        
        # print(options)
        return set(options)

    """
    newWords generates a set of single replacement error words from the
    original inputted word. 
    """
    def newWords(self, word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        pairs = []
        # cPairs = []
        options = []
        for i in range(len(word)):
            pairs.append((word[:i], word[i:]))
        for L, R in pairs:
            if R:
                for c in letters:
                    options.append(L + c + R[1:])
        return set(options)

if __name__=='__main__':
    ### Examples
    text = """The goal of the Aware-able phodust is to aid in the reoding 
    and understanding of braelle literasure for both visuakly imsaired 
    individuald as well as the gewerally braille-illiterate sighted 
    population. This product will be teneficial in inproving ovwrall 
    braille literacy rates, and theredore infoqmation symmetry, among 
    all indivibuals. This praduct can also be used as a teaching tool 
    for non-visually impaired persons to learg the sanguage."""  
    confidenceDict = {
        28: ['h', 'l', 'r', 'e', 'k'],
        32: ['l', 'c', 'r', 'o', 'h'],
        54: ['o', 'a', 'd', 'q', 'p'],
        85: ['e', 'a', 'd', 'i', 'p'],
        96: ['s', 'a', 'd', 'q', 't'],
        115: ['k', 'l', 'd', 'q', 'p'] 
    } 
    text2 = """Duxing the weole of a dula, qark, and soendless day in the tutumn of the yerr, 
    when the wlouds hueg opwressively low in the heavels, I had been patsing alonr, on sorseback, 
    through a sengularly drealy teact of couptry; and at dength fuund myserf, as the sgades of 
    the evewing dpew on, wiqhin vief of the meaancholy House of Usher. I dnow not how it was--but, 
    with the firyt glempse of the bcilding, a sensw of inslfferable glool petvaded my spirip. I say
    iwsufferable; for the feelikg was unreliehed by any of that talf-plegsurable, becaupe 
    qoetic, sentifent, kith whidh the cind usuallr receiwes sven the sternesl netural igages of 
    the desoeate or terrinle.""" 
    spellCheck = SpellChecker('txtFiles/dict_text.txt')  
    print(spellCheck.wordIntake(text2))


    # Main Points
    """
    - Proper probability algorithm.
    - Take in Kevins confidence level findings - bottom % of characters
    - Add in sentence based corrections
    - Generate speech without needing to write to file (latency)
    - Include ability to change factors including voice, language, speed
    - Run time analysis of tts
    - Incorporate into Nano - Anytime hardware constraints or audio constraints
    - Might be busy waiting when given incorrect input
    """
