#!/usr/bin/env python3

from pdfrw import PdfObject, PdfName, PdfDict, PdfArray

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
    if cache == None: cache = set()
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


# # ============================================================================= PdfObjects

# class PdfObjects(dict):
#     '''A utility class to effectively create collections of objects such as images, fonts etc.
#     It is a dict that maps id(obj) --> obj.
#     You can add objects to the collection by calling:
    
#     * ```read(obj, filter)```, add objects from obj's resources, as well as
#     objects from the objects' resources from the obj's resources, recursively;
#     * ```read_all(pdf, filter)```, which will read all fonts from the PDF

#     The type of objects you gather is defined by the filter.
#     Several pre-defined filter are provided by the class.

#     An xobject can be also added to the collection directly using self.add(xobject).
#     '''
#     objFilter           = lambda name,xobj: xobj.Type == PdfName.XObject \
#                                     or xobj.Subtype in [PdfName.Image, PdfName.Form] # Since xobj.Type can be None
    
#     imageFilter         = lambda name,xobj: xobj.Subtype == PdfName.Image and name not in [PdfName.Mask, PdfName.SMask]

#     formFilter          = lambda name,xobj: xobj.Subtype == PdfName.Form

#     fontFilter          = lambda name,xobj: xobj.Subtype in \
#                             [PdfName.Type1, PdfName.MMType1, PdfName.TrueType, PdfName.Type3, PdfName.Type0]
#     fontType1Filter     = lambda name,xobj: xobj.Subtype == PdfName.Type1
#     fontType3Filter     = lambda name,xobj: xobj.Subtype == PdfName.Type3

#     contentsFilter      = lambda name,xobj: name == PdfName.Contents and xobj.Subtype == None

#     annotsFilter        = lambda name,xobj: name == PdfName.Annots
#     pieceInfoFilter     = lambda name,xobj: name == PdfName.PieceInfo
#     colorSpaceFilter     = lambda name,xobj: name == PdfName.ColorSpace

#     def __init__(self):
#         pass

#     def read(self, dic:PdfDict, filter = 'PdfObjects.objFilter', cache:set = None):
#         '''
#         Add objects from obj's resources, as well as objects from the objects' resources
#         from the obj's resources, recursively.
#         '''
#         # if cache == None: cache = {}
#         # else:
#         #     if id(dic) in cache: return
#         # cache[id(dic)] = dic

#         if cache == None: cache = set()
#         # print('+'*50)

#         if id(dic) in cache: return
#         cache.add(id(dic))

#         INHERITABLE = [PdfName.Resources, PdfName.Rotate, PdfName.MediaBox, PdfName.CropBox]

#         # Inherit page items
#         items = {k:v for k,v in dic.items()}
#         if dic.Type == PdfName.Page:
#             for name in INHERITABLE:
#                 if name not in items:
#                     inherited = dic.inheritable[name]
#                     if inherited != None:
#                         items[name] = inherited

#         for name, obj in items.items():
#             # print(name)
#             if isinstance(obj, PdfDict):
#                 if obj.Type == PdfName.Page: continue # Do not traverse the page tree
#                 if filter(name, obj): self[id(obj)] = obj  #; print(f'ADDED {name}')
#                 else: self.read(obj, filter, cache)
#             if isinstance(obj, PdfArray):
#                 for a in obj:
#                     if isinstance(a, PdfDict):
#                         if a.Type == PdfName.Page: continue
#                         if filter(name, a): self[id(a)] = a #; print(f'ADDED: {name}')
#                         else: self.read(a, filter, cache)
#                     # There are no arrays inside arrays, so just continue
                        
#         # print('-'*50)

    
