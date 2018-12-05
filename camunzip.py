from trees import *
from vl_codes import *
import arithmetic
import conditional_arithmetic as con_ari
from json import load
from sys import argv


def camunzip(filename,length_of_LSTM_context=30):
    if (filename[-1] == 'h'):
        method = 'huffman'
    elif (filename[-1] == 's'):
        method = 'shannon_fano'
    elif (filename[-1] == 'a'):
        method = 'arithmetic'
    elif (filename[-1] == "c"):
        method = "conditional-arithmetic"
    else:
        raise NameError('Unknown compression method')
    
    with open(filename, 'rb') as fin:
        y = fin.read()
    y = bytes2bits(y)

    pfile = filename[:-1] + 'p'
    with open(pfile, 'r') as fp:
        frequencies = load(fp)
    n = sum([frequencies[a] for a in frequencies])
    p = dict([(int(a),frequencies[a]/n) for a in frequencies])

    if method == 'huffman' or method == 'shannon_fano':
        if (method == 'huffman'):
            xt = huffman(p)
            c = xtree2code(xt)
        else:
            c = shannon_fano(p)
            xt = code2xtree(c)

        x = vl_decode(y, xt)

    elif method == 'arithmetic':
        x = arithmetic.decode(y,p,n)
    elif method == "conditional-arithmetic":
        x = con_ari.decode(y,p,n,length_of_LSTM_context)
    else:
        raise NameError('This will never happen (famous last words)')
    
    # '.cuz' for Cam UnZipped (don't want to overwrite the original file...)
    outfile = filename[:-4] + '.cuz' 

    with open(outfile, 'wb') as fout:
        fout.write(bytes(x))

    return x


if __name__ == "__main__":
    if (len(argv) != 2):
        print('Usage: python %s filename\n' % sys.argv[0])
        print('Example: python %s hamlet.txt.czh' % sys.argv[0])
        print('or:      python %s hamlet.txt.czs' % sys.argv[0])
        print('or:      python %s hamlet.txt.cza' % sys.argv[0])
        exit()

    camunzip(argv[1])
