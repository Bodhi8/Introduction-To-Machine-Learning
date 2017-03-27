#!/usr/bin/python

# from nltk.stem.snowball import SnowballStemmer
import string

# from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """

    myReturnString = ''
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()
    print("all_text - begin .............................")
    print(all_text)
    print("all_text - end .............................\n")
    # print("type(all_text) - {}\n".format(type(all_text)))
    # type(all_text) - <class 'str'>

    ### split off metadata
    
    content = all_text.split("X-FileName:")
    print("len(content) - {}\n".format(len(content)))

    # print("content[0] - {}".format(content[0]))
    # print("type(content[0]) - {}".format(type(content[0])))
    #        type(content[0]) - <class 'str'>

    
    # content[1] - With original punctuation from email 
    print("content[1] - begin .....")
    print(content[1])
    print("content[1] - end .....\n")
    # print("type(content[1]) - {}".format(type(content[1])))
    #        type(content[1]) - <class 'str'>
       
    words = ""
    if len(content) > 1:
        ### remove punctuation
        # text_string = content[1].translate(string.maketrans("", ""), string.punctuation) # no older Python
        
        # print("string.punctuation - {}\n".format(string.punctuation))
        #      string.punctuation - !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~  # all of these - None 
        # print("type(string.punctuation) - {}\n".format(type(string.punctuation)))
        #        type(string.punctuation) - <class 'str'>
        
        # print('str.maketrans("", "", string.punctuation) - ')
        # print(str.maketrans("", "", string.punctuation))
        # {64: None, 124: None, 125: None, 91: None, 92: ....
        # Python documentation - dictionary mapping Unicode ordinals (integers) or characters (strings of length 1) to Unicode ordinals, strings (of arbitrary lengths) or None.
        # print("type(str.maketrans("", "", string.punctuation)) - {}\n".format(type(str.maketrans("", "", string.punctuation))))
        #        type(str.maketrans(, , string.punctuation)) - <class 'dict'>

        text_string = content[1].translate(str.maketrans("", "", string.punctuation))
        
        # Without original punctuation from email
        print("text_string (punctuation stripped out) - ")
        print(text_string)
        print()
        # print("type(text_string) - {}\n".format(type(text_string)))
        #        type(text_string) - <class 'str'>

        ### project part 2: comment out the line below
        words = text_string
        print("words - ")
        print(words)
        print()
        # print("type(words) - {}\n".format(type(words)))
        #        type(words) - <class 'str'>

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        
        mySplitOutput = text_string.split()
        print("mySplitOutput - {}\n".format(mySplitOutput))
        #      mySplitOutput - ['Hi', 'Everyone', 'If', 'you', 'can', 'read', 'this', 'message', 'youre', 'properly', 'using', 'parseOutText', 'Please', 'proceed', 'to', 'the', 'next', 'part', 'of', 'the', 'project']
        # print("type(mySplitOutput) - {}\n".format(type(mySplitOutput)))
        #        type(mySplitOutput) - <class 'list'>
        
        # done AFTER stemmimg 
        # vectorizer = CountVectorizer()
        myStemmer = SnowballStemmer('english')
        # print("myStemmer - {}".format(myStemmer))
        #       myStemmer - <nltk.stem.snowball.SnowballStemmer object at 0x10b4b57f0>
        # print("type(myStemmer) - {}\n".format(type(myStemmer)))
        #        type(myStemmer) - <class 'nltk.stem.snowball.SnowballStemmer'>
        for myWord in mySplitOutput:
            # print("myWord - {}".format(myWord))
            # print("type(myWord) - {}\n".format(type(myWord)))
            #        type(myWord) - <class 'str'>
            myStemmedWord = myStemmer.stem(myWord)
            # print("myStemmedWord - {}\n".format(myStemmedWord))
            # print("type(myStemmedWord) - {}\n".format(type(myStemmedWord)))
            #        type(myStemmedWord) - <class 'str'>
            print("{} - {}".format(myWord, myStemmedWord))
            myReturnString = myReturnString + myStemmedWord + ' '
        print()
            
    return myReturnString
    # return words

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    
    #print text
    print("text returned from parseOutText(ff) - ")
    print(text)
    # right quiz answer part 1 
    # Hi Everyone  If you can read this message youre properly using parseOutText  Please proceed to the next part of the project
    # right quiz answer

    # right quiz answer part 2     
    # hi everyon if you can read this messag your proper use parseouttext pleas proceed to the next part of the project 

if __name__ == '__main__':
    main()

