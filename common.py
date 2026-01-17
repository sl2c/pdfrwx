#!/usr/bin/env python3

import inspect,sys,os,re
from itertools import groupby

from pdfrw import PdfArray, PdfDict, IndirectPdfDict, PdfObject, PdfName
from .pdfgeometry import BOX

# ========================================================= MESSAGES

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def err(msg):
    '''Prints an error message in the form: 'Error in class.func(), line: msg',
    where class.func() are the class and the function that called err().
    Exits by sys.exit(1) afterwords'''
    stack = inspect.stack()    
    the_class = stack[1][0].f_locals["self"].__class__.__name__ if "self" in stack[1][0].f_locals else 'global'
    the_func = stack[1][0].f_code.co_name
    # the_func = inspect.getouterframes(inspect.currentframe(), 2)[1][3]
    lineno = inspect.getouterframes(inspect.currentframe(), 2)[1][2]
    eprint(f'{the_class}.{the_func}(): error in line {lineno}: {msg}')
    sys.exit(1)

def msg(msg):
    '''Prints a warning message in the form: 'func(): warning: msg', where func() is the function that called warn().'''
    stack = inspect.stack()
    className = stack[1][0].f_locals["self"].__class__.__name__ if "self" in stack[1][0].f_locals \
                    else stack[1][0].f_locals["cls"].__class__.__name__ if "cls" in stack[1][0].f_locals \
                    else 'global'
    callerName = inspect.getouterframes(inspect.currentframe(), 2)[1][3]

    # Need Python 3.11 for this to work
    # qualName = inspect.currentframe().f_back.f_back.f_code.co_qualname
    # eprint(f'{qualName}(): {msg}')
    eprint(f'{className}.{callerName}(): {msg}')

def warn(msg):
    '''Prints a warning message in the form: 'func(): warning: msg', where func() is the function that called warn().'''
    stack = inspect.stack()
    the_class = stack[1][0].f_locals["self"].__class__.__name__ if "self" in stack[1][0].f_locals else 'global'
    callerName = inspect.getouterframes(inspect.currentframe(), 2)[1][3]
    eprint(f'{the_class}.{callerName}(): warning: {msg}')

def er(msg):
    '''Prints a message in the form 'Error: msg' to stderr, then exits.'''
    eprint(f'Error: {msg}')
    sys.exit(1)

# ========================================================================== Dictionaries access

def encapsulate(obj):
    '''
    Returns PdfArray([obj]) if obj is not a PdfArray, or obj itself otherwise.
    '''
    return obj if isinstance(obj,PdfArray) else PdfArray([obj]) if obj != None else PdfArray()

def decapsulate(array:PdfArray):
    '''
    Returns None if len(array) == 0, or array[0] if len(array) == 1, or array otherwise. 
    '''
    return None if len(array) == 0 else array[0] if len(array) == 1 else array

def get_box(xobj:PdfDict):
    '''
    Returns xobj's box, which defaults to BOX(xobj.CropBox/MediaBox) for pdf pages,
    to BOX(xobj.BBox) for PDF Form xobjects, and to BOX([0,0,1,1]) for Image xobjects.
    For all other objects returns None
    '''
    # f = lambda array: [float(a) for a in array]
    cropBox = xobj.inheritable.CropBox
    if cropBox == None: cropBox = xobj.inheritable.MediaBox
    return BOX(cropBox) if xobj.Contents != None \
        else BOX(xobj.BBox) if xobj.Subtype == PdfName.Form \
        else BOX([0,0,1,1]) if xobj.Subtype == PdfName.Image \
        else None

# ========================================================================== Dictionaries access

def get_key(dic:dict, key:str, defaultValue = None):
    '''
    An auxiliary function to safely poll PDF dictionaries, where 'safely means' that both dic and key can be None:
    if dic == None or key not in dic, defaultValue is returned
    '''
    return dic[key] if dic != None and key in dic else defaultValue

def get_any_key(dic:dict, *keys):
    '''
    Return the value corresponding to the first key from keys that is found in dic,
    or None if none of the keys are in dic.
    '''
    if dic == None: return None
    for key in keys:
        if key in dic: return dic[key]
    return None

def chain(a:dict, b:dict):
    '''
    Returns dictionary c such that for k in a: c[k] = b[a[k]] or a[k] if a[k] is not in b
    '''
    return {k:b.get(v,v) for k,v in a.items()}

# ========================================================================== System utilities

def getExecPath():
    '''
    Returns the path to the executable
    '''
    return os.path.dirname(__file__)

# ========================================================================== Input/output

def formatSize(size:float, unit:str = '', kilo:int = 1024):
    '''
    Size formatting: `formatSize(1024, 'K')` returns '1M'.
    '''
    nextUnit = {'':'K','K':'M','M':'G','G':'T'}
    factor = lambda size: 10 if size < 10 else 1
    truncate = lambda size: f'{round(size*factor(size))/factor(size):.2f}'.rstrip('0').rstrip('.')
    return f'{truncate(size)}{unit}' if size < kilo or unit not in nextUnit else formatSize(size/kilo, nextUnit[unit])

def formatWithCommas(size:int):
    '''
    Call `formatWithCommas(12345) to get '12,345'`
    '''
    assert isinstance(size, int)
    s = f'{size}'
    s = ' '*(-len(s) % 3) + s
    s = ','.join(s[i:i+3] for i in range(0,len(s),3))
    return s.lstrip()

def listToRangesString(lst:list[int]):
    '''
    Converts a list of ints such as [1,2,3,6,7,9,13,14] to a ranges string: '1-3,6-7,9,13-14'.
    The list doesn't have to be sorted, and the integers don't have to be all unique.
    '''
    return ','.join(f'{x[0]}-{x[-1]}' if len(x)>1 else f'{x[0]}' 
                        for x in [[e[1] for e in g]
                        for _,g in groupby(enumerate(sorted(list(set(lst)))), key=lambda x: x[1]-x[0])])

def rangesStringToList(ranges:str):
    '''
    Converts a ranges string such as '1-3,6-7,9,13-14' to a sorted list of ints: [1,2,3,6,7,9,13,14].
    The ranges can overlap, yet each int in the resulting list will appear only once.
    '''
    limitsToRange = lambda a: range(int(a[0]), int(a[-1])+1)
    try:
        return sorted(list(set(n for r in re.split(',', ranges) for n in limitsToRange(re.split('-',r)))))
    except:
        raise ValueError(f'bad ranges string: {ranges}')