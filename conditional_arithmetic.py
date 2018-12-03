from bisect import bisect

import sys
from math import floor, ceil
from sys import stdout as so
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import operator
import time

char_to_int = {'\n': 0, ' ': 1, '!': 2, '"': 3, '#': 4, '$': 5, '%': 6, '&': 7,
                   "'": 8, '(': 9, ')': 10, '*': 11, ',': 12, '-': 13, '.': 14, '/': 15,
                   '0': 16, '1': 17, '2': 18, '3': 19, '4': 20, '5': 21, '6': 22, '7': 23,
                   '8': 24, '9': 25, ':': 26, ';': 27, '?': 28, '@': 29, 'A': 30, 'B': 31,
                   'C': 32, 'D': 33, 'E': 34, 'F': 35, 'G': 36, 'H': 37, 'I': 38, 'J': 39,
                   'K': 40, 'L': 41, 'M': 42, 'N': 43, 'O': 44, 'P': 45, 'Q': 46, 'R': 47,
                   'S': 48, 'T': 49, 'U': 50, 'V': 51, 'W': 52, 'X': 53, 'Y': 54, 'Z': 55,
                   '[': 56, ']': 57, '_': 58, 'a': 59, 'b': 60, 'c': 61, 'd': 62, 'e': 63,
                   'f': 64, 'g': 65, 'h': 66, 'i': 67, 'j': 68, 'k': 69, 'l': 70, 'm': 71,
                   'n': 72, 'o': 73, 'p': 74, 'q': 75, 'r': 76, 's': 77, 't': 78, 'u': 79,
                   'v': 80, 'w': 81, 'x': 82, 'y': 83, 'z': 84, '|': 85}

def load_LSTM(n):
    start = time.time()
    
    # define the LSTM model
    model = Sequential()
    model.add(CuDNNLSTM(256, input_shape=(n, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(86, activation='softmax'))
    
    # load the network weights
    filename = "weights-improvement-capital-cheat-20-1.5695.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
#    print("The time to load the LSTM model is:",time.time()-start)
    
    return model

def get_probabilities(prior_sequence,model,n_vocab,is_encoding):
    
    start = time.time()
    
    int_to_char =  dict((v,k) for k,v in char_to_int.items())
    
    prior_sequence_num = []
    
    for search_letter in prior_sequence.decode():
        prior_sequence_num.append(char_to_int[search_letter])

    z = numpy.reshape(prior_sequence_num, (1, len(prior_sequence_num), 1))
    z = z / float(n_vocab)
    prediction = model.predict(z, verbose=0)
    prediction_dict = {}
    
    for i in range(len(prediction[0])):
        prediction_dict[ord(int_to_char[i])] = prediction[0][i]
    
    prediction_dict = dict(sorted(prediction_dict.items(), key=operator.itemgetter(1), reverse = True))
    
    p = prediction_dict
    
    if not is_encoding:
        alphabet = list(p)
        

    # Compute cumulative probability as in Shannon-Fano
    f = [0]
    for a in p:
        f.append(f[-1]+p[a])
    f.pop()
    
    if is_encoding:
        f = dict([(a,mf) for a,mf in zip(p,f)])
    
    
#    print("The time to find the probability using the LSTM is:",time.time()-start)

    if is_encoding:
        return p,f
    else:
        p = list(p.values())
        return p,f,alphabet

def encode(x,p):
    start = time.time()
    '''
    The inputs are described below:
        x: The binary array of characters of the text. E.g.
        
            b'        HAMLET\n\n\n        DRAMATIS PERSONAE\n\n\nCLAUDIUS'
        
        p: The dictionary of probabilities with keys being the binary
           representation of the character, which can be found by using ord().
           E.g.
           
           p = {10: 0.029197397591758073, 32: 0.2731417752210936, 33: 0.0014538323697467627,
                38: 2.4150039364564164e-05, 39: 0.0046657876052337965, 40: 7.728012596660533e-05,
                41: 7.728012596660533e-05, 44: 0.015852085838899917, 45: 0.0020382633223692153,
                .
                .
                .
                120: 0.0008549113935055714, 121: 0.014958534382411043, 122: 0.00024633040151855447, 
                124: 0.000231840377899816}
           
           Note that the entries are sorted by their key value, the binary 
           representation of the character in the alphabet.
           
           >>for char in b"Hamlet":
           >>    print(char)
           
           Output:
                72
                97
                109
                108
                101
                116
                
            This can be checked by typing ord("H"), which returns 72.
    '''
    
    precision = 48
    
    one = int(2**precision - 1)
    quarter = int(ceil(one/4))
    half = 2*quarter
    threequarters = 3*quarter

    length_trained_LSTM = 30
    
    '''
    chars has the following shape:
        [10, 32, 33, 38, 39, 40, 41, 44, 45, 46, 58, 59, 63, 65, 66, 67, 68,
         69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
         86, 87, 89, 90, 91, 93, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
         107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 
         121, 122, 124]
        
    This is essentially the same as the key values of the p vector above.
    '''
    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(x)))
    
    '''
    char_to_int has the following shape: {ord(character): character_index}
        {10: 0, 32: 1, 33: 2, 38: 3, 39: 4, 40: 5, 41: 6, 44: 7, 45: 8, 46: 9,
        58: 10, 59: 11, 63: 12, 65: 13, 66: 14, 67: 15, 68: 16, 69: 17, 70: 18,
        71: 19, 72: 20, 73: 21, 74: 22, 75: 23, 76: 24, 77: 25, 78: 26, 79: 27,
        80: 28, 81: 29, 82: 30, 83: 31, 84: 32, 85: 33, 86: 34, 87: 35, 89: 36,
        90: 37, 91: 38, 93: 39, 97: 40, 98: 41, 99: 42, 100: 43, 101: 44,
        102: 45, 103: 46, 104: 47, 105: 48, 106: 49, 107: 50, 108: 51, 109: 52,
        110: 53, 111: 54, 112: 55, 113: 56, 114: 57, 115: 58, 116: 59, 117: 60,
        118: 61, 119: 62, 120: 63, 121: 64, 122: 65, 124: 66}
        
    int_to_chars has the following shape: {character_index: ord(character)}
        {0: 10, 1: 32, 2: 33, 3: 38, 4: 39, 5: 40, 6: 41, 7: 44, 8: 45, 9: 46,
        10: 58, 11: 59, 12: 63, 13: 65, 14: 66, 15: 67, 16: 68, 17: 69, 18: 70,
        19: 71, 20: 72, 21: 73, 22: 74, 23: 75, 24: 76, 25: 77, 26: 78, 27: 79,
        28: 80, 29: 81, 30: 82, 31: 83, 32: 84, 33: 85, 34: 86, 35: 87, 36: 89,
        37: 90, 38: 91, 39: 93, 40: 97, 41: 98, 42: 99, 43: 100, 44: 101,
        45: 102, 46: 103, 47: 104, 48: 105, 49: 106, 50: 107, 51: 108, 52: 109,
        53: 110, 54: 111, 55: 112, 56: 113, 57: 114, 58: 115, 59: 116, 60: 117,
        61: 118, 62: 119, 63: 120, 64: 121, 65: 122, 66: 124}
        
    Note: these are maps between the integer "labels" are the value of the index
          of the character in the list of characters found (in this case 67
          characters were found) and their binary represenation, and viceversa.
    '''
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    
    # Number of characters that the LSTM was trained with
    n_vocab = 86 
    
    #load the neural network weights saved after training
    model = load_LSTM(n=length_trained_LSTM)
    
    y = [] # initialise output list
    lo,hi = 0,one # initialise lo and hi to be [0,1.0)
    straddle = 0 # initialise the straddle counter to 0
    
    p = dict([(a,p[a]) for a in p if p[a]>0]) # eliminate zero probabilities
    '''
        p is the pdf and has the following shape:
            {10: 0.029197397591758073, 32: 0.2731417752210936, 33: 0.0014538323697467627,
            38: 2.4150039364564164e-05, 39: 0.0046657876052337965, 40: 7.728012596660533e-05,
            41: 7.728012596660533e-05, 44: 0.015852085838899917, 45: 0.0020382633223692153,
            .
            .
            .
            117: 0.019774052231705138, 118: 0.005747709368766271, 119: 0.012949251107279305,
            120: 0.0008549113935055714, 121: 0.014958534382411043, 122: 0.00024633040151855447,
            124: 0.000231840377899816}
        '''
    
    # Compute cumulative probability as in Shannon-Fano
    f = [0]
    for a in p:
        f.append(f[-1]+p[a])
    f.pop()
    f = dict([(a,mf) for a,mf in zip(p,f)])
    '''
    f is the cumulative probability and has the following shape:
        {10: 0, 32: 0.029197397591758073, 33: 0.3023391728128517, 38: 0.3037930051825985,
         39: 0.30381715522196306, 40: 0.30848294282719685, 41: 0.30856022295316343,
         44: 0.30863750307913, 45: 0.3244895889180299, 46: 0.32652785224039915,
         58: 0.3328165224909317, 59: 0.33548268683677956, 63: 0.33829375141881485,
         .
         .
         .
         119: 0.9707591323373854, 120: 0.9837083834446647, 121: 0.9845632948381703,
         122: 0.9995218292205813, 124: 0.9997681596220999}
    '''
    
    for k in range(length_trained_LSTM):
        # arithmetic coding is slower than vl_encode, so we display a "progress bar"
        # to let the user know that we are processing the file and haven't crashed...
        if k % 100 == 0:
            so.write('Arithmetic encoded %d%%    \r' % int(floor(k/len(x)*100)))
            so.flush()

        # 1) calculate the interval range to be the difference between hi and lo and 
        # add 1 to the difference. The added 1 is necessary to avoid rounding issues
        lohi_range =  hi - lo + 1

        # 2) narrow the interval end-points [lo,hi) to the new range [f,f+p]
        # within the old interval [lo,hi], being careful to round 'innwards' so
        # the code remains prefix-free (you want to use the functions ceil and
        # floor). This will require two instructions. Note that we start computing
        # the new 'lo', then compute the new 'hi' using the scaled probability as
        # the offset from the new 'lo' to the new 'hi'
        lo = lo + int(ceil(f[x[k]]*lohi_range))
        hi = lo + int(floor((p[x[k]])*lohi_range))

        if (lo == hi):
            raise NameError('Zero interval!')

        # Now we need to re-scale the interval if its end-points have bits in common,
        # and output the corresponding bits where appropriate. We will do this with an
        # infinite loop, that will break when none of the conditions for output / straddle
        # are fulfilled
        while True:
            
            if hi < half: # if lo < hi < 1/2
                # stretch the interval by 2 and output a 0 followed by 'straddle' ones (if any)
                # and zero the straddle after that. In fact, HOLD OFF on doing the stretching:
                # we will do the stretching at the end of the if statement
                y.append(0) # append a zero to the output list y
                y.extend([1]*straddle) # extend by a sequence of 'straddle' ones
                straddle = 0 # zero the straddle counter
            elif lo >= half: # if hi > lo >= 1/2
                # stretch the interval by 2 and substract 1, and output a 1 followed by 'straddle'
                # zeros (if any) and zero straddle after that. Again, HOLD OFF on doing the stretching
                # as this will be done after the if statement, but note that 2*interval - 1 is equivalent
                # to 2*(interval - 1/2), so for now just substract 1/2 from the interval upper and lower
                # bound (and don't forget that when we say "1/2" we mean the integer "half" we defined
                # above: this is an integer arithmetic implementation!
                y.append(1) # append a 1 to the output list y
                y.extend([0]*straddle) # extend 'straddle' zeros
                straddle = 0 # reset the straddle counter
                lo -= half
                hi -= half # substract half from lo and hi
            elif lo >= quarter and hi < threequarters: # if 1/4 < lo < hi < 3/4
                # we can increment the straddle counter and stretch the interval around
                # the half way point. This can be impemented again as 2*(interval - 1/4),
                # and as we will stretch by 2 after the if statement all that needs doing
                # for now is to subtract 1/4 from the upper and lower bound
                straddle += 1 # increment straddle
                lo -= quarter
                hi -= quarter # subtract 'quarter' from lo and hi
            else:
                break # we break the infinite loop if the interval has reached an un-stretchable state
            # now we can stretch the interval (for all 3 conditions above) by multiplying by 2
            lo *= 2 # multiply lo by 2
            hi = 2*hi + 1 # multiply hi by 2 and add 1 (I DON'T KNOW WHY +1 IS NECESSARY BUT IT IS. THIS IS MAGIC.
                # A BOX OF CHOCOLATES FOR ANYONE WHO GIVES ME A WELL ARGUED REASON FOR THIS... It seems
                # to solve a minor precision problem.)
    
    '''
    At this point, the output vector y looks like this:
        y = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
             1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
             1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0,
             1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0,
             1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0,
             1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
        
        len(y) = 128
        
    for only a length-thirty input sequence
    '''
    
    for k in range(length_trained_LSTM,len(x)): # for every symbol
    
#        A = "e are two dogs in the red hous"
        A = x[k-length_trained_LSTM:k]
        '''
        For 30, in the first iteration, A looks like this
        
            A = b'        HAMLET\n\n\n        DRAMA'
            
            len(A) = 30
        '''
        
        p,f = get_probabilities(prior_sequence=A, 
                                model=model, 
                                n_vocab=n_vocab,
                                is_encoding=True)
        '''
        p looks like this:
        p = {0: 0.5235542, 11: 0.22898121, 38: 0.06381601, 1: 0.039520435, 30: 0.03142421,
             51: 0.019769678, 48: 0.014437281, 49: 0.010225049, 8: 0.009553557, 34: 0.005492557,
             44: 0.0051619327, 59: 0.004937402, 47: 0.0036791929, 13: 0.0036268476, 78: 0.0026237431,
             .
             .
             .
             15: 4.2959977e-07, 2: 1.8014913e-07, 55: 8.2301405e-08, 53: 3.7719122e-08, 84: 3.1454117e-09,
             4: 8.135823e-10, 5: 7.925468e-10, 6: 7.350468e-10, 29: 3.6740272e-10}
        
        f looks like this:
            f = {0: 0, 11: 0.5235542058944702, 38: 0.752535417675972, 1: 0.8163514286279678,
                 30: 0.8558718636631966, 51: 0.8872960731387138, 48: 0.9070657510310411,
                 49: 0.9215030316263437, 8: 0.9317280808463693, 34: 0.9412816381081939,
                 44: 0.9467741949483752, 59: 0.9519361276179552, 47: 0.9568735295906663,
                 13: 0.9605527224484831, 78: 0.964179570088163,
                 .
                 .
                 .
                 15: 0.9999992546568706, 2: 0.9999996842566361, 55: 0.9999998644057655,
                 53: 0.9999999467071703, 84: 0.9999999844262923, 4: 0.999999987571704,
                 5: 0.9999999883852864, 6: 0.9999999891778332, 29: 0.99999998991288}
        '''
#        print(p)
#        print(f)
#        raise Exception
        
        # arithmetic coding is slower than vl_encode, so we display a "progress bar"
        # to let the user know that we are processing the file and haven't crashed...
        if k % 100 == 0:
            so.write('Arithmetic encoded %d%%    \r' % int(floor(k/len(x)*100)))
            so.flush()

        # 1) calculate the interval range to be the difference between hi and lo and 
        # add 1 to the difference. The added 1 is necessary to avoid rounding issues
        lohi_range =  hi - lo + 1

        # 2) narrow the interval end-points [lo,hi) to the new range [f,f+p]
        # within the old interval [lo,hi], being careful to round 'innwards' so
        # the code remains prefix-free (you want to use the functions ceil and
        # floor). This will require two instructions. Note that we start computing
        # the new 'lo', then compute the new 'hi' using the scaled probability as
        # the offset from the new 'lo' to the new 'hi'
        lo = lo + int(ceil(f[x[k]]*lohi_range))
        hi = lo + int(floor((p[x[k]])*lohi_range))

        if (lo == hi):
            raise NameError(f'Zero interval! lo = {lo}, hi = {hi}')

        # Now we need to re-scale the interval if its end-points have bits in common,
        # and output the corresponding bits where appropriate. We will do this with an
        # infinite loop, that will break when none of the conditions for output / straddle
        # are fulfilled
        while True:
            
            if hi < half: # if lo < hi < 1/2
                # stretch the interval by 2 and output a 0 followed by 'straddle' ones (if any)
                # and zero the straddle after that. In fact, HOLD OFF on doing the stretching:
                # we will do the stretching at the end of the if statement
                y.append(0) # append a zero to the output list y
                y.extend([1]*straddle) # extend by a sequence of 'straddle' ones
                straddle = 0 # zero the straddle counter
            elif lo >= half: # if hi > lo >= 1/2
                # stretch the interval by 2 and substract 1, and output a 1 followed by 'straddle'
                # zeros (if any) and zero straddle after that. Again, HOLD OFF on doing the stretching
                # as this will be done after the if statement, but note that 2*interval - 1 is equivalent
                # to 2*(interval - 1/2), so for now just substract 1/2 from the interval upper and lower
                # bound (and don't forget that when we say "1/2" we mean the integer "half" we defined
                # above: this is an integer arithmetic implementation!
                y.append(1) # append a 1 to the output list y
                y.extend([0]*straddle) # extend 'straddle' zeros
                straddle = 0 # reset the straddle counter
                lo -= half
                hi -= half # substract half from lo and hi
            elif lo >= quarter and hi < threequarters: # if 1/4 < lo < hi < 3/4
                # we can increment the straddle counter and stretch the interval around
                # the half way point. This can be impemented again as 2*(interval - 1/4),
                # and as we will stretch by 2 after the if statement all that needs doing
                # for now is to subtract 1/4 from the upper and lower bound
                straddle += 1 # increment straddle
                lo -= quarter
                hi -= quarter # subtract 'quarter' from lo and hi
            else:
                break # we break the infinite loop if the interval has reached an un-stretchable state
            # now we can stretch the interval (for all 3 conditions above) by multiplying by 2
            lo *= 2 # multiply lo by 2
            hi = 2*hi + 1 # multiply hi by 2 and add 1 (I DON'T KNOW WHY +1 IS NECESSARY BUT IT IS. THIS IS MAGIC.
                # A BOX OF CHOCOLATES FOR ANYONE WHO GIVES ME A WELL ARGUED REASON FOR THIS... It seems
                # to solve a minor precision problem.)


    # termination bits
    # after processing all input symbols, flush any bits still in the 'straddle' pipeline
    straddle += 1 # adding 1 to straddle for "good measure" (ensures prefix-freeness)
    if lo < quarter: # the position of lo determines the dyadic interval that fits
        y.append(0) # output a zero followed by "straddle" ones
        y.extend([1]*straddle)
    else:
        y.append(1) # output a 1 followed by "straddle" zeros
        y.extend([0]*straddle)
        
    print("The time required for encoding is:",time.time()-start)
    
    return(y)

def decode(y,p,n):
    
    precision = 48
    one = int(2**precision - 1)
    quarter = int(ceil(one/4))
    half = 2*quarter
    threequarters = 3*quarter
    
    length_trained_LSTM = 30
    
    # Number of characters that the LSTM was trained with
    n_vocab = 86    

    #load the neural network weights saved after training
    model = load_LSTM(n=length_trained_LSTM)    

    p = dict([(a,p[a]) for a in p if p[a]>0])
    
    alphabet = list(p)
    f = [0]
    for a in p:
        f.append(f[-1]+p[a])
    f.pop()
    
    p = list(p.values())
    
#    print(p)
#    print(f)
    
    y.extend(precision*[0]) # dummy zeros to prevent index out of bound errors
    x = n*[0] # initialise all zeros 
    
    # initialise by taking first 'precision' bits from y and converting to a number
    value = int(''.join(str(a) for a in y[0:precision]), 2) 
    position = precision # position where currently reading y
    lo,hi = 0,one

    for k in range(length_trained_LSTM):
    
        if k % 100 == 0:
            so.write('Arithmetic decoded %d%%    \r' % int(floor(k/n*100)))
            so.flush()


        lohi_range = hi - lo + 1
        # This is an essential subtelty: the slowest part of the decoder is figuring out
        # which symbol lands us in an interval that contains the encoded binary string.
        # This can be extremely wasteful (o(n) where n is the alphabet size) if you proceed
        # by simple looping and comparing. Here we use Python's "bisect" function that
        # implements a binary search and is 100 times more efficient. Try
        # for a = [a for a in f if f[a]<(value-lo)/lohi_range)][-1] for a MUCH slower solution.
        a = bisect(f, (value-lo)/lohi_range) - 1
        x[k] = alphabet[a] # output alphabet[a]
        
        lo = lo + int(ceil(f[a]*lohi_range))
        hi = lo + int(floor(p[a]*lohi_range))
        if (lo == hi):
            raise NameError('Zero interval!')

        while True:
            if hi < half:
                # do nothing
                pass
            elif lo >= half:
                lo = lo - half
                hi = hi - half
                value = value - half
            elif lo >= quarter and hi < threequarters:
                lo = lo - quarter
                hi = hi - quarter
                value = value - quarter
            else:
                break
            lo = 2*lo
            hi = 2*hi + 1
            value = 2*value + y[position]
            position += 1
            if position == len(y):
                raise NameError('Unable to decompress')
                
    
    for k in range(length_trained_LSTM,n):        
        
        if k % 100 == 0:
            so.write('Arithmetic decoded %d%%    \r' % int(floor(k/n*100)))
            so.flush()

        A = bytes(x[k-length_trained_LSTM:k])
        
        p,f,alphabet = get_probabilities(prior_sequence=A, 
                                model=model, 
                                n_vocab=n_vocab,
                                is_encoding=False)
        
        lohi_range = hi - lo + 1
        # This is an essential subtelty: the slowest part of the decoder is figuring out
        # which symbol lands us in an interval that contains the encoded binary string.
        # This can be extremely wasteful (o(n) where n is the alphabet size) if you proceed
        # by simple looping and comparing. Here we use Python's "bisect" function that
        # implements a binary search and is 100 times more efficient. Try
        # for a = [a for a in f if f[a]<(value-lo)/lohi_range)][-1] for a MUCH slower solution.
        a = bisect(f, (value-lo)/lohi_range) - 1
        
#        print(p)
#        print(f)
#        print(a)
#        print(alphabet)
#        print(k)
#        print(len(x))
#        raise Exception
        
        x[k] = alphabet[a] # output alphabet[a]


        lo = lo + int(ceil(f[a]*lohi_range))
        hi = lo + int(floor(p[a]*lohi_range))
        
        if (lo == hi):
            raise NameError('Zero interval!')

        while True:
            if hi < half:
                # do nothing
                pass
            elif lo >= half:
                lo = lo - half
                hi = hi - half
                value = value - half
            elif lo >= quarter and hi < threequarters:
                lo = lo - quarter
                hi = hi - quarter
                value = value - quarter
            else:
                break
            lo = 2*lo
            hi = 2*hi + 1
            value = 2*value + y[position]
            position += 1
            if position == len(y):
                raise NameError('Unable to decompress')

    return(x)
    


