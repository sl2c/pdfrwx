#!/usr/bin/env python3

# pdf-fix-png-predictors, version 1.0

try:
    from pdfrw import PdfReader, PdfDict, crypt, PdfName
    from pdfrw.uncompress import uncompress
    from pdfrw.tokens import PdfTokens
    from pdfrw.py23_diffs import convert_load
    from pdfrw.errors import PdfParseError, log
except:
    print(f'pdfrw not found; run: pip3 install pdfrw')
    exit()

from pdfrw import PdfWriter, PdfObject, IndirectPdfDict, PdfArray

import gc

import argparse, os
from typing import Callable

# ============================================================================= Aux functions

def get_key(dic:dict, key:str, defaultValue = None):
    '''
    An auxiliary function to safely poll PDF dictionaries, where 'safely means' that both dic and key can be None:
    if dic == None or key not in dic, defaultValue is returned
    '''
    return dic[key] if dic != None and key in dic else defaultValue

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

# ============================================================================= PdfObjects()

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

# ============================================================================= class PdfReaderX

class PdfReaderX(PdfReader):

    '''
    A class is derived from PdfReader and overrides:
    
    * __init__() — to correctly open certain "hybrid reference files", see PDF Ref. 1.7 p.109;
    * load_stream_objects() — to correctly process object streams containing PDF comments.

    Thus, the PdfReaderX class is a drop-in replacement of the pdfrw's original PdfReader,
    with better stability.
    '''

    # ============================================================================
    # Functions below are overridden to fix bugs
    # ============================================================================

    def __init__(self, fname=None, fdata=None, decompress=False,
                 decrypt=False, password='', disable_gc=True, verbose=True):
        self.private.verbose = verbose

        # Runs a lot faster with GC off.
        disable_gc = disable_gc and gc.isenabled()
        if disable_gc:
            gc.disable()

        try:
            if fname is not None:
                assert fdata is None
                # Allow reading preexisting streams like pyPdf
                if hasattr(fname, 'read'):
                    fdata = fname.read()
                else:
                    try:
                        f = open(fname, 'rb')
                        fdata = f.read()
                        f.close()
                    except IOError:
                        raise PdfParseError('Could not read PDF file %s' %
                                            fname)

            assert fdata is not None
            fdata = convert_load(fdata)

            if not fdata.startswith('%PDF-'):
                startloc = fdata.find('%PDF-')
                if startloc >= 0:
                    log.warning('PDF header not at beginning of file')
                else:
                    lines = fdata.lstrip().splitlines()
                    if not lines:
                        raise PdfParseError('Empty PDF file!')
                    raise PdfParseError('Invalid PDF header: %s' %
                                        repr(lines[0]))

            self.private.version = fdata[5:8]

            endloc = fdata.rfind('%EOF')
            if endloc < 0:
                raise PdfParseError('EOF mark not found: %s' %
                                    repr(fdata[-20:]))
            endloc += 6
            junk = fdata[endloc:]
            fdata = fdata[:endloc]
            if junk.rstrip('\00').strip():
                log.warning('Extra data at end of file')

            private = self.private
            private.indirect_objects = {}
            private.deferred_objects = set()
            private.special = {'<<': self.readdict,
                               '[': self.readarray,
                               'endobj': self.empty_obj,
                               }
            for tok in r'\ ( ) < > { } ] >> %'.split():
                self.special[tok] = self.badtoken

            startloc, source = self.findxref(fdata)
            private.source = source

            # Find all the xref tables/streams, and
            # then deal with them backwards.
            xref_list = []
            while 1:
                source.obj_offsets = {}
                trailer, is_stream = self.parsexref(source)

                prev = trailer.Prev
                if prev is None:
                    token = source.next()
                    if token != 'startxref' and not xref_list:
                        source.warning('Expected "startxref" '
                                       'at end of xref table')
                    break

                xref_list.append((source.obj_offsets, trailer, is_stream))

                # If trailer references a cross-ref stream we have a "hybrid reference" file,
                # see "Compatibility with Applications That Do Not Support PDF 1.5", PDF Ref. 1.7 p.109
                if trailer.XRefStm:
                    source.floc = int(trailer.XRefStm)
                    source.obj_offsets = {}
                    source.next()
                    tr = self.parse_xref_stream(source)
                    xref_list.append((source.obj_offsets, tr, True))

                source.floc = int(prev)

            # Handle document encryption
            private.crypt_filters = None
            if decrypt and PdfName.Encrypt in trailer:
                identity_filter = crypt.IdentityCryptFilter()
                crypt_filters = {
                    PdfName.Identity: identity_filter
                }
                private.crypt_filters = crypt_filters
                private.stream_crypt_filter = identity_filter
                private.string_crypt_filter = identity_filter

                if not crypt.HAS_CRYPTO:
                    raise PdfParseError(
                        'Install PyCrypto to enable encryption support')

                self._parse_encrypt_info(source, password, trailer)

            if is_stream:
                self.load_stream_objects(trailer.object_streams)

            while xref_list:
                later_offsets, later_trailer, is_stream = xref_list.pop()
                source.obj_offsets.update(later_offsets)
                if is_stream:
                    trailer.update(later_trailer)
                    self.load_stream_objects(later_trailer.object_streams)
                else:
                    trailer = later_trailer

            trailer.Prev = None

            if (trailer.Version and
                    float(trailer.Version) > float(self.version)):
                self.private.version = trailer.Version

            if decrypt:
                self.decrypt_all()
                trailer.Encrypt = None

            if is_stream:
                self.Root = trailer.Root
                self.Info = trailer.Info
                self.ID = trailer.ID
                self.Size = trailer.Size
                self.Encrypt = trailer.Encrypt
            else:
                self.update(trailer)

            # self.read_all_indirect(source)
            private.pages = self.readpages(self.Root)
            if decompress:
                self.uncompress()

            # For compatibility with pyPdf
            private.numPages = len(self.pages)
        finally:
            if disable_gc:
                gc.enable()

    def load_stream_objects(self, object_streams):
        # read object streams
        objs = []
        for num in object_streams:
            obj = self.findindirect(num, 0).real_value()
            assert obj.Type == '/ObjStm'
            objs.append(obj)

        # read objects from stream
        if objs:
            # Decrypt
            if self.crypt_filters is not None:
                crypt.decrypt_objects(
                    objs, self.stream_crypt_filter, self.crypt_filters)

            # Decompress
            uncompress(objs)

            for obj in objs:
                objsource = PdfTokens(obj.stream, 0, False)
                next = objsource.next
                offsets = []
                firstoffset = int(obj.First)
                while objsource.floc < firstoffset:

                    toc1 = next()

                    # Skip PDF comments in object streams
                    if toc1[0] == '%':
                        continue
                    toc2 = next()

                    offsets.append((int(toc1), firstoffset + int(toc2)))

                for num, offset in offsets:
                    # Read the object, and call special code if it starts
                    # an array or dictionary
                    objsource.floc = offset
                    sobj = next()

                    # Skip PDF comments in object streams
                    while sobj[0] == '%':
                        sobj = next()

                    func = self.special.get(sobj)
                    if func is not None:
                        sobj = func(objsource)

                    key = (num, 0)
                    self.indirect_objects[key] = sobj
                    if key in self.deferred_objects:
                        self.deferred_objects.remove(key)

                    # Mark the object as indirect, and
                    # add it to the list of streams if it starts a stream
                    sobj.indirect = key

# ============================================================================= fix_png_predictors()

def fix_png_predictors(obj:IndirectPdfDict):
    '''
    For all filters that use PNG predictors (PDF Ref. p. 76),
    the function will set the value of the `/Predictor` parameter to 15:
    the actual value of the predictor parameter doesn't matter, according the PDF Ref.,
    as long as it is in the PNG predictors range (10..15).
    Setting it to 15 prevents errors in some PDF viewers/processors that erroneously
    base assumptions about the actual predictor values used in the stream on the value of
    the `/Predictor` entry. Returns True if the object's predictors have been modified.
    '''
    assert isinstance(obj, PdfDict)
    assert obj.Subtype == PdfName.Image

    filters = encapsulate(obj.Filter)
    parms = encapsulate(obj.DecodeParms)

    MODIFIED = False
    for f in range(len(filters)):

        filter = filters[f]
        parm = parms[f] if f < len(parms) else PdfDict()
        if parm == 'null': parm = PdfDict()

        if filter in ['/FlateDecode', '/Fl', '/LZWDecode', '/LZW']:
            predictor = int(get_key(parm, '/Predictor', '1'))
            if 10 <= predictor <= 14:
                parms[f].Predictor = 15
                obj.DecodeParms = decapsulate(parms)
                MODIFIED = True

    if obj.SMask:
        MODIFIED |= fix_png_predictors(obj.SMask)

    return MODIFIED


# ============================================================================= main()

if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument('inputPaths', nargs='+', metavar='FILE', help='input files: images or PDF')

    options = ap.parse_args()

    LINE_SINGLE = '-'*64
    LINE_DOUBLE = '='*64


    for inputPath in options.inputPaths:

        pdf = PdfReaderX(inputPath)

        # Iterate over pages
        isImage = lambda name, obj: isinstance(obj, PdfDict) and obj.Subtype == PdfName.Image

        cache = set()
        MODIFIED = False

        print(LINE_SINGLE)
        print('Processing:', inputPath)
        print('Pages:', len(pdf.pages))

        for pageNo, page in enumerate(pdf.pages):

            imageTuples = [t for t in PdfObjects(page, test = isImage, cache = cache)]

            if len(imageTuples) > 0:
                print('Page:', pageNo+1, ', # images:', len(imageTuples))

                for n, t in enumerate(imageTuples):

                    name, image = t

                    if fix_png_predictors(image):
                        print(f'*** Fixed image # {n+1}')
                        MODIFIED = True

        # ---------- Write processed results to PDF ----------
        if MODIFIED:
            outputPath = '-fixed'.join(os.path.splitext(inputPath))
            print(f'+++ writing output to: {outputPath}')
            PdfWriter(outputPath, trailer=pdf, compress=True).write()
        else:
            print(f'--- file not modified, no output is produced')

