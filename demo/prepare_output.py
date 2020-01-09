import os
import glob
import itertools

def init(name='default', wdir='./output', fext=['xdmf','h5']):

    # init output folder
    try:
        os.mkdir(wdir)
    except:
        pass

    # change to output folder
    try:
        os.chdir(wdir)
    except:
        raise Exception("Cannot change to directory '%s'." % wdir)

    # detetct files to erase
    files = itertools.chain.from_iterable( glob.glob(name + '*.' + e) for e in fext )

    # remove files
    try:
        for f in files: os.remove(f)
    except:
        raise Exception("Cannot remove files '%s'." % files)
