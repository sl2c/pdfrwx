#!/usr/bin/env python3

from pdfrw import PdfName, PdfDict, IndirectPdfDict, PdfArray, PdfObject
from typing import Callable

# ============================================================================= getDictItems()

def getPdfDictItems(object:PdfDict):
    '''
    For `PdfDict` objects, the `object.items()` function doesn't return inherited items.
    The `getDictItems` returns the full set of dict's items (a `dict_items` object) including the inherited ones.
    '''
    assert isinstance(object, PdfDict)

    # Inherit the inheritables
    INHERITABLE = [PdfName.Resources, PdfName.Rotate, PdfName.MediaBox, PdfName.CropBox]

    items = {k:v for k,v in object.items()}

    if object.Type == PdfName.Page:
        for inheritable in INHERITABLE:
            if inheritable in items: continue
            inherited = object.inheritable[inheritable]
            if inherited != None:
                items[inheritable] = inherited

    return items.items()

# ============================================================================= PdfObjects [generator]

def PdfObjects(object,
               name:str = '',
               test:Callable[[PdfName, PdfObject],bool] = None,
               cache:set = None,
               debug = False):
    '''
    The depth-first iterator over (sub-name, sub-object) tuples for all sub-objects referenced by
    object argument recursively (via PdfDict entries & PdfArray elements).
    
    For sub-objects in PdfDicts, the 'sub-name' element of the tuple is the PdfName key in the PdfDict.
    
    For sub-objects in PdfArrays, the 'sub-name' is a string of the format
    '/arrayName[i]', where 'i' is the index of the object in the PdfArray and 'arrayName'
    if the corresponding 'name' of the array itself (thus, nested arrays' elements will have names like '/a[0][1]').

    The following tuples are excluded from the iterator:

    * those where the 'sub-name' is PdfName.Parent: we do not recurse upward;
    * those where 'sub-object' is a PdfDict and sub-object.Type == PdfName.Page: we do not recurse to other pages.

    The latter of these exclusions implies, in particular, that calling PdfObjects(object = pdfRoot)
    where pdfRoot is the root dictionary of the PDF document will yield all objects in the root except
    the pages. In other words, to iterate over sub-objects inside pages one needs to call PdfObjects(object = pdfPage)
    on each page separately.

    After yield (sub-name, sub-object) tuples, the generator always yields (name, object) tuple of the
    object argument itself. In other words, the generator always yields at least one tuple for any object argument.

    Every sub-object referenced by the object argument is yielded just once. If cache argument is a set of objects id()'s,
    the objects with these ids are excluded from the yielded, and the ids of the yielded objects are added to cache.
    This allows to effectively use cache in successive calls to PdfObjects(object) for objects that may share
    referenced sub-objects.
    
    The 'depth-first' approach in particular results in the iterator yielding tuples for sub-objects
    inside pdf page/xobject resources (fonts, images, ..) BEFORE yielding the tuple for page/xobject itself.
    This is useful when using the iterator to recursively parse all streams in a pdf page/xobject:
    by the time the stream parser is called on the page/xobject stream it will have been
    called on all the streams in the resources. If these resources are referenced in the page/xobject
    stream, the results of the resources streams parsing will be available.
    '''

    # Check the cache
    if cache is None: cache = set()
    if id(object) in cache: return
    cache.add(id(object))

    if debug: print("debug:", name)

    isPdfPage = lambda o: isinstance(o, PdfDict) and o.Type == PdfName.Page

    # Process PdfArrays
    if isinstance(object, PdfArray):

        for i in range(len(object)):
            element = object[i]
            # Do not recurse to (other) pages
            if not isPdfPage(element):
                yield from PdfObjects(element, name=f'{name}[{i}]', test=test, cache=cache, debug=debug)

    # Process PdfDicts
    if isinstance(object, PdfDict):

        # Iterate over items
        for itemName, item in getPdfDictItems(object):
            # Do not recurse to (other) pages or upward
            if itemName != PdfName.Parent and not isPdfPage(item):
                yield from PdfObjects(item, name=itemName, test=test, cache=cache, debug=debug)

    if test is None or test(name, object):
        yield (name, object)

# ========================================================================== PdfObjSize()

def pdfObjSize(obj:PdfObject, cache:set = None):
    '''
    Returns (size, overhead), where size is the size of the streams of obj and all other
    dictionaries that the obj references, and overhead is (the estimate of) the corresponding size
    of their headers (the dictionaries per se). If cache dict is supplied, it is used to
    eliminate double-counting of dictionaries and arrays.
    '''

    if obj is None:
        return (0,0)

    if cache is None:
        cache = set()

    # Check dicts and arrays against cache to count them only once
    if isinstance(obj, PdfDict) or isinstance(obj, PdfArray):
        if id(obj) in cache:
            return 0, 0
        cache.add(id(obj))

    # debug =  isinstance(obj, PdfDict) and obj.Type == PdfName.Annot

    REF_OVERHEAD = 7 # a typical '123 0 R' reference length
    DICT_OVERHEAD = 5 # '<<>> '
    INDIRECT_DICT_OVERHEAD = 43 # '1234 0 obj <<>> endobj ' + 20 bytes in the XRef table
    STREAM_OVERHEAD = 17 # 'stream endstream '
    ARRAY_OVERHEAD = 3 # '[] '

    if isinstance(obj, PdfDict):
        size = overhead = 0
        if obj.Length is not None:
            size = int(obj.Length)
            overhead = STREAM_OVERHEAD
        overhead += INDIRECT_DICT_OVERHEAD if isinstance(obj, IndirectPdfDict) else DICT_OVERHEAD

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
            # Do not traverse up the page tree, to object's resources, transparency groups
            # or to other pages & xobjects
            # Here, we exclude groups in order not to double-count
            if k in [PdfName.Parent, PdfName.Resources, PdfName.Group] \
                or isinstance(v,PdfDict) and (v.Type == PdfName.Page or v.Subtype == PdfName.Form):
                s,o = 0,0
            else:
                s,o = pdfObjSize(v, cache)
            if isinstance(v,IndirectPdfDict):
                o += REF_OVERHEAD
            size += s
            overhead += len(k) + o + 2 # 2 separators
    elif isinstance(obj,PdfArray):
        size,overhead = 0,ARRAY_OVERHEAD
        for v in obj:
            # Do not traverse to other pages & forms
            if isinstance(v,PdfDict) and (v.Type == PdfName.Page or v.Subtype == PdfName.Form):
                s,o = 0,0
            else:
                s,o = pdfObjSize(v, cache)
            if isinstance(v,IndirectPdfDict):
                o += REF_OVERHEAD
            size += s; overhead += o + 1 # 1 separator
    else:
        size,overhead = 0, len(str(obj))

    return size, overhead

# ============================================================================= removePageRefs()

def removeInvalidRefs(obj,
                invalidRefIds:set,
                cache:set = None,
                debug:bool = False) -> int:
    '''
    Removes all invalid references from `obj` by substituting them with `null`s.
    Invalid references are references to objects whose ids are in `invalidRefIds`.
    Returns the number of removed references.
    '''
    # Check the cache
    if cache is None: cache = set()
    if id(obj) in cache: return 0
    cache.add(id(obj))

    isPdfPage = lambda o: isinstance(o, PdfDict) and o.Type == PdfName.Page

    count = 0

    # Process PdfArrays
    if isinstance(obj, PdfArray):

        for i in range(len(obj)):
            element = obj[i]
            if id(element) in invalidRefIds:
                obj[i] = PdfObject('null')
                count += 1
                if debug:
                    print(f'removing element {i} from an array')
            # Do not recurse to (other) pages
            if not isPdfPage(element):
                count += removeInvalidRefs(element, invalidRefIds, cache, debug)

    # Process PdfDicts
    if isinstance(obj, PdfDict):

        # Iterate over items
        for itemName, item in getPdfDictItems(obj):
            # Remove references
            if id(item) in invalidRefIds:
                obj[itemName] = None
                count += 1
                if debug:
                    print(f'removing dict.{itemName}')
            # Do not recurse to (other) pages or upward
            if not isPdfPage(item) and itemName != PdfName.Parent:
                count += removeInvalidRefs(item, invalidRefIds, cache, debug)

    return count

