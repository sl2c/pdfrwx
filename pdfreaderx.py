#!/usr/bin/env python3


from pdfrw import PdfReader, PdfDict, crypt, PdfName
from pdfrw.uncompress import uncompress
from pdfrw.tokens import PdfTokens
from pdfrw.py23_diffs import convert_load
from pdfrw.errors import PdfParseError, log

import gc

# ========================================================================== class PdfReaderX

class PdfReaderX(PdfReader):

    '''
    A class is derived from PdfReader and extends it with the following page manipulation functions:

    * removePage(page) — removes a page from the page tree;
    * insertPage(page, N) — inserts a page into a page tree.

    The class also overrides:
    
    * __init__() — to correctly open certain "hybrid reference files", see PDF Ref. 1.7 p.109;
    * load_stream_objects() — to correctly process object streams containing PDF comments.

    Thus, the PdfReaderX class is a drop-in replacement of the pdfrw's original PdfReader,
    with more functionality and better stability.
    '''

    def __shift_kids_count(leaf:PdfDict, shift:int):
        '''
        An auxiliary private function used by removePage() and insertPage()
        '''
        if leaf == None: return
        leaf.Count = int(leaf.Count) + shift
        PdfReaderX.__shift_kids_count(leaf.Parent, shift)

    def removePage(self, page:PdfDict):
        '''
        Remove the page from the page tree (the page should be part of the page tree).
        To remove page by pageNo call:

        self.removePage(self.pages[pageNo])
        '''
        page.Parent.Kids.remove(page)
        PdfReaderX.__shift_kids_count(page.Parent, -1)
        if page.Parent.Count == 0 and page.Parent.Parent != None:
            page.Parent.Parent.Kids.remove(page.Parent) # delete empty parent from the grandparent
        page.Parent = None
        self.private.pages = self.readpages(self.Root)
        self.Root.PageLabels = None

    def insertPage(self, page:PdfDict, pageNo:int):
        '''
        Insert page into the page tree before pageNo.
        Values n >= len(self.pages) append the page to the tree.
        Values n <= 0 prepend the page to the tree.
        '''
        if pageNo >= len(self.pages):
            page.Parent = self.Root.Pages
            page.Parent.Kids.append(page)
        else:
            p = self.pages[max(pageNo,0)]
            page.Parent = p.Parent
            page.Parent.Kids.insert(page.Parent.Kids.index(p),page)
        PdfReaderX.__shift_kids_count(page.Parent, +1)
        self.private.pages = self.readpages(self.Root)
        self.Root.PageLabels = None

    def copyPage(pageFrom:PdfDict, pageTo:PdfDict):
        '''
        Copies pageFrom to pageTo while keeping pageTo itself in the page tree.
        This is done by copying all attributes from pageFrom to pageTo,
        except the attributes related to pageTo's position in the page tree.
        This preserves all links to pageTo that may exist in the PDF while
        effectively replacing the page.

        The pageFrom object should not be accessed directly after the call to this function.
        '''

        p, q = pageTo, pageFrom

        p.indirect = q.indirect
        p._stream = q.stream

        p.Contents = q.Contents
        p.Resources = q.inheritable.Resources

        p.MediaBox = q.inheritable.MediaBox
        p.CropBox = q.inheritable.CropBox
        p.ArtBox = q.inheritable.ArtBox
        p.BleedBox = q.inheritable.BleedBox
        p.TrimBox = q.inheritable.TrimBox
        
        p.Rotate = q.inheritable.Rotate

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
