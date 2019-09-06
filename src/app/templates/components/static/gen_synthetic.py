import getopt
import pickle
from priv_publish import PSynDB
import sys 

def print_help():
    print('gen_synthetic.py -i <inputfile> -o <outputfile>')

def main(argv):
    inputfile = ''
    outputfile = ''

    try:
        opts, args = getopt.getopt(argv, 'hi:o:', ['infile=', 'outfile='])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)

    infile = ''
    outfile = ''
    for opt, arg in opts:
        if opt == '-h':
            print('gen_synthetic.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ('-i', '--infile'):
            infile = arg
        elif opt in ('-o', '--outfile'):
            outfile = arg

    if infile == '':
        print('error: must specify infile')
        print_help()
        sys.exit(1)
    if outfile == '':
        print('error: must specify outfile')
        print_help()
        sys.exit(1)

    eps = 0.1
    seed = 0
    fd = open('meta_objs.pickle', 'rb')
    W, A, config = pickle.load(fd)
    fd.close()
    p_syn_db = PSynDB(config, A)
    x_synth = p_syn_db.synthesize(infile, eps, seed)
    x_synth.to_csv(outfile, index=False)

if __name__ == '__main__':
    main(sys.argv[1:])
