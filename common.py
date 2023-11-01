#!/usr/bin/env python3

import inspect,sys

from pdfrw import PdfArray

# ========================================================= MESSAGES

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def err(msg):
    '''Prints an error message in the form: 'Error in class.func(), line: msg',
    where class.func() are the class and the function that called err().
    Exits by sys.exit(1) aftewords'''
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
    '''Prints a message in the form 'Error: msg' to stderr, the exits.'''
    eprint(f'Error: {msg}')
    sys.exit(1)

# ========================================================================== Dictionaries access

def encapsulate(obj):
    '''
    Make a PdfArray([obj]) out of obj unless obj is already a PdfArray
    '''
    return obj if isinstance(obj,PdfArray) else PdfArray([obj]) if obj != None else PdfArray()

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
