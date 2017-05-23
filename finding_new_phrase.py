from utlis.glove import tf_glove

'''
Use Word Vector way to get new phrase. 

First: we get all the word vectors of all the words, use a big crops;

Second: we get a sub word vectors from a mini crops which contains the new information updated;

Third: For any words pair W1, W2 we get the distance of those two pair in big corps and updated crops;

Forth: If the distance if more closed to the distance in big crops, we could classify this two words as a new
phrase.
'''