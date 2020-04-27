import os
import glob
import itertools


def init(name='default', wdir='./output', fext=['xdmf', 'h5']):
    # Init output folder
    try:
        os.mkdir(wdir)
    except OSError:
        pass

    # Change to output folder
    try:
        os.chdir(wdir)
    except OSError:
        raise Exception("Cannot change to directory '%s'." % wdir)

    # Detect files to erase
    files = itertools.chain.from_iterable(glob.glob(name + '*.' + e) for e in fext)

    # Remove files
    try:
        for f in files:
            os.remove(f)
    except OSError:
        raise Exception("Cannot remove files '%s'." % files)
