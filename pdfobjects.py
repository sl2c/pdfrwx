#!/usr/bin/env python3

from pdfrw import PdfName, PdfDict, PdfArray

# ============================================================================= PdfObjects [generator]

def PdfObjects(object, name:str = '', cache:set = None, debug = False):
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
            if isPdfPage(element): continue
            yield from PdfObjects(element, name=f'{name}[{i}]', cache=cache, debug=debug)

    # Process PdfDicts
    if isinstance(object, PdfDict):

        # Inherit the inheritables
        INHERITABLE = [PdfName.Resources, PdfName.Rotate, PdfName.MediaBox, PdfName.CropBox]
        items = {k:v for k,v in object.items()}
        if object.Type == PdfName.Page:
            for inheritable in INHERITABLE:
                if inheritable in items: continue
                inherited = object.inheritable[inheritable]
                if inherited != None:
                    items[inheritable] = inherited

        # Iterate over items
        for itemName, item in items.items():
            # Do not recurse to (other) pages or upward
            if itemName == PdfName.Parent or isPdfPage(item): continue
            yield from PdfObjects(item, name=itemName, cache=cache, debug=debug)

    yield (name, object)
