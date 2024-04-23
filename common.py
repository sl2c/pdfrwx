#!/usr/bin/env python3

import inspect,sys

from pdfrw import PdfArray, PdfDict, IndirectPdfDict, PdfObject, PdfName
from pdfrwx.pdfgeometry import BOX

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
    the_class = stack[1][0].f_locals["self"].__class__.__name__ if "self" in stack[1][0].f_locals else 'global'
    callerName = inspect.getouterframes(inspect.currentframe(), 2)[1][3]
    eprint(f'{the_class}.{callerName}(): {msg}')

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


# ========================================================================== PdfObjSize()

def pdfObjSize(obj:PdfObject, cache:set = set()):
    '''
    Returns (size, overhead), where size is the size of the streams of obj and all other
    dictionaries that the obj references, and overhead is (the estimate of) the corresponding size
    of their headers (the dictionaries per se). If cache dict is supplied, it is used to
    eliminate double-counting of dictionaries and arrays.
    '''

    # if cache != None and (isinstance(obj,PdfDict) or isinstance(obj,PdfArray)):
    # if cache != None:
    if id(obj) in cache: return 0, 0
    cache.add(id(obj))

    REF_OVERHEAD = 8 # a typical '123 0 R ' reference length
    DICT_OVERHEAD = 5 # '<<>> '
    ARRAY_OVERHEAD = 3 # '[] '

    if isinstance(obj, PdfDict):
        size = int(obj.Length) if obj.Length != None else 0
        overhead = DICT_OVERHEAD

        # Inherit page items
        INHERITABLE = [PdfName.Resources, PdfName.Rotate, PdfName.MediaBox, PdfName.CropBox]
        items = {k:v for k,v in obj.items()}
        if obj.Type == PdfName.Page:
            for name in INHERITABLE:
                if name not in items:
                    inherited = obj.inheritable[name]
                    if inherited != None:
                        items[name] = inherited

        for k,v in items.items():
            # Do not traverse up the page tree or to other pages
            if k == PdfName.Parent or isinstance(v,PdfDict) and v.Type == PdfName.Page: s,o = 0,0
            else: s,o = pdfObjSize(v, cache)
            if isinstance(v,IndirectPdfDict): o += REF_OVERHEAD
            size += s; overhead += len(k) + o + 2 # 2 separators
    elif isinstance(obj,PdfArray):
        size,overhead = 0,ARRAY_OVERHEAD
        for v in obj:
            # Do not traverse to other pages
            if isinstance(v,PdfDict) and v.Type == PdfName.Page: s,o = 0,0
            else: s,o = pdfObjSize(v, cache)
            if isinstance(v,IndirectPdfDict): o += REF_OVERHEAD
            size += s; overhead += o + 1 # 1 separator
    else:
        size,overhead = 0, len(str(obj))

    return size, overhead

