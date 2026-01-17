#!/usr/bin/env python3

import re
from sly import Lexer, Parser

import sys, os, re
from math import sqrt
from typing import Callable

# Try using: github.com/sarnold/pdfrw as it contains many fixes compared to pmaupin's version
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfDict, IndirectPdfDict, PdfName

from .common import err,msg,warn,eprint, encapsulate, get_box
from .pdffont import PdfFont, PdfTextString, PdfFontGlyphMap
from .pdfstreamparser import PdfStream
from .djvusedparser import DjVuSedLexer, DjVuSedParser
from .pdfstate import PdfState
from .pdffilter import PdfFilter
from .pdfimage import PdfImage
from .pdfgeometry import VEC, MAT, BOX

import xml.etree.ElementTree as ET # for parsing hOCR files

from PIL import Image, ImageOps
from math import floor, ceil

import numpy as np

import hashlib


# ========================================================================== class PdfStreamEditor

class PdfStreamEditor:

    def __init__(self,
                    xobj:PdfDict,
                    glyphMap:PdfFontGlyphMap = PdfFontGlyphMap(),
                    filterText:bool=False,
                    filterImages:bool=False,
                    filterVector:bool=False,
                    normalize=False,
                    debug:bool=False,
                    extractFontProgram:bool = False,
                    makeSyntheticCmap:bool = False):
        '''
        Parse self.stream and store the resulting parsed PDF stream tree in self.tree.

        The filterText, filterImages & filterVector cause the text, image & vector graphics
        operators to be skipped when parsing self.stream.

        If normalize == True, text operators are normalized: all text operators are converted
        to either Tj or TJ, and Tw operators are removed altogether by introducing explicit
        word spacings in text strings, if needed.
        '''
        self.xobj = xobj
        self.glyphMap = glyphMap

        self.filterText = filterText
        self.filterImages = filterImages
        self.filterVector = filterVector

        self.normalize = normalize
        self.debug = debug

        self.extractFontProgram = extractFontProgram
        self.makeSyntheticCmap = makeSyntheticCmap

        self.isModified = False

        # The actual state of the stream editor should be initialized just once!
        self.state = PdfState(resources = self.xobj.inheritable.Resources,
                              glyphMap = self.glyphMap,
                              extractFontProgram = self.extractFontProgram,
                              makeSyntheticCmap = self.makeSyntheticCmap)

        # Parse the stream tree
        stream = self.get_stream()
        if stream == None: stream = ''
        self.tree = PdfStream.stream_to_tree(stream=stream,
                                             filterText = filterText,
                                             filterImages = filterImages,
                                             filterVector = filterVector,
                                             XObject = xobj)
        if normalize:
            self.tree = self.normalize_text_operators(self.tree)

    def normalize_text_operators(self, tree:list, state:PdfState = None) -> list:
        '''
        Normalizes text operators by replacing:

        '," --> Tc, Tw, Tj
        Tw  --> deleted, its effect absorbed in the explicit character displacement values in TJ
        TD  --> TL, Td
        '''
        if state == None: state = PdfState(self.xobj.inheritable.Resources,
                                           self.glyphMap,
                                           extractFontProgram=False,
                                           makeSyntheticCmap=False)
        result = []
        for leaf in tree:
            cmd, args = leaf[0], leaf[1]
            cs = state.current_state

            if cmd == 'BT':
                state.update(cmd,args)
                leaf[2] = self.normalize_text_operators(leaf[2], state)
                result.append(leaf)
                continue

            if cmd == '"':
                state.update('Tw',[args[0]])
                state.update('Tc',[args[1]])
                result.append(['Tc',[args[1]]])
                cmd, args = "'", [args[2]]

            if cmd == "'":
                state.update('T*',[])
                result.append(['T*',[]])
                cmd = 'Tj'

            if cmd == 'TD':
                state.update('TL',[-float(args[1])])
                result.append(['TL',[-float(args[1])]])
                cmd = 'Td'

            if cmd in ['Tj', 'TJ'] and cs.Tw != 0 and not cs.font.is_cid():
                args = [self.apply_Tw(args[0], cs)]
                cmd = 'Tj' if isinstance(args[0],str) else 'TJ'

            state.update(cmd, args)

            if cmd != 'Tw':
                result.append([cmd,args])

        return result

    def apply_Tw(self, s, cs:dict):
        '''
        Introduces explicit word spacing (cs.Tw) in text operator strings. After applying this function
        to all such strings, the Tw operators can be dropped from the stream entirely.

        PDF Ref. 1.7, sec. 5.2.2: "Word spacing (Tw) is applied to every occurrence of the single-byte
        character code 32 in a string when using a simple font or a composite font that defines code 32
        as a single-byte code. It does not apply to occurrences of the byte value 32 in multiple-byte codes."
        '''
        Tw = cs.Tw
        # font = cs.font

        # Note: the numbers in the TJ operator arguments array are displacements that are expressed
        # in neither the text space units, nor the glyph space units:
        # "The number is expressed in thousandths of a unit of text space" (PDF Ref sec 5.3.2)
        # For explicit math refer to PdfState.update()
        scale_x = 1000 / cs.fontSize
        if isinstance(s,str): s = [s]
        sMod = []
        for tok in s:
            t = PdfTextString(tok)
            format = t.format()
            if format == None: sMod.append(tok); continue # pass the displacements
            codes = t.to_codes(isCID=False)
            if ' ' in codes:
                l = [c + ' ' for c in re.split(r' ', codes)]
                l[-1] = l[-1][:-1]
                l = [PdfTextString.from_codes(c, format=format) for c in l]
                l = [l[0]] + [a for c in l[1:] for a in (f'{-Tw * scale_x:f}', c)]
                sMod += l
            else:
                sMod.append(tok)

        return sMod if len(sMod) > 1 else sMod[0]


    def get_stream(self):
        '''
        Gets self.xobj's stream in cases where self.xobj.stream exists as well as in cases
        where self.xobj is a PDF page, and so its stream(s) are in self.xobj.Contents.
        '''
        return ''.join(PdfFilter.uncompress(c).stream for c in encapsulate(self.xobj.Contents)) \
            if self.xobj.Contents != None else PdfFilter.uncompress(self.xobj).stream
        
    def set_stream(self, stream:str):
        '''
        Sets self.xobj's stream in cases where self.xobj.stream exists as well as in cases
        where self.xobj is a PDF page, and so its stream(s) are in self.xobj.Contents.
        '''
        if self.xobj.Contents != None:
            self.xobj.Contents = IndirectPdfDict(stream = stream)
        else:
            self.xobj.stream = stream
            self.xobj.Filter = None

    def update_stream(self):
        '''
        This function just calls: self.set_stream(PdfStream.tree_to_stream(self.tree))
        '''
        self.set_stream(PdfStream.tree_to_stream(self.tree))

    # --------------------------------------------------------------- recurse()

    def recurse(self, recursedFunction:Callable, xobjCache, *args, **kwarg):
        '''
        When trying to perform a task on any xobject's stream one should keep in mind that
        the xobject's stream at hand may reference other xobjects (via Do operators), and those xobjects
        may reference yet other xobjects, ad infinitum. The directed graph of references looks like
        a tree at best, and may actually have loops at worst.
        
        So what you really need is a kind of automated way to recurse into this mess, making sure you visit
        each node (xobject) just once. You also want to make the results of the operation of your function
        on each xobject somehow available to the operating (recursed) function since its very operation on
        the current xobject's stream may depend on the results of its operation on other xobjects; and so
        the order of the recursion is important.

        The recurse() function tries to solve all of these problems at once: it recurses, as its name
        implies, through the graph of xobjects, parses each xobject's stream, calls the recursedFunction
        on the parsed stream tree of each item in self.XObject, and stores the result of this call in the xobjCache.
        By checking the stored results, the recurse() function makes sure it visits each xobject just once.
        At the very last, it calls the recursedFunction() on self and returns the result.

        For an example use of this recurse() function see .processText()
        '''
        try: xobjects = self.xobj.inheritable.Resources.XObject.values()
        except: xobjects = []
        for x in xobjects:
            if id(x) not in xobjCache and x.Subtype == PdfName.Form and x.stream != None:
                # Creates an editor of the same type as that of any inheriting class
                editor = type(self)(xobj = x,
                                    glyphMap = self.glyphMap,
                                    filterText = self.filterText,
                                    filterImages = self.filterImages,
                                    filterVector = self.filterVector,
                                    normalize = self.normalize,
                                    debug = self.debug,
                                    extractFontProgram = self.extractFontProgram,
                                    makeSyntheticCmap = self.makeSyntheticCmap)

                # to avoid infinite recursion in the recurse() call below when there are xobj loops
                xobjCache[id(x)] = None

                xobjCache[id(x)] = editor.recurse(recursedFunction, xobjCache, *args, **kwarg)
                if editor.isModified: self.isModified = True

        return recursedFunction(self, xobjCache, *args, **kwarg)

    # --------------------------------------------------------------- flattenImages()

    def flattenImages(self, dpi:float = 300):
        '''
        Flatten images
        '''

        # Set up the canvas
        cropBox = get_box(self.xobj)
        if cropBox is None: return None
        canvasWidth, canvasHeight = cropBox.w() * dpi/72, cropBox.h() * dpi/72
        canvasBox = BOX([0, 0, canvasWidth, canvasHeight])
        width, height = ceil(canvasWidth), ceil(canvasHeight)
        if width == 0 or height == 0: err(f'bad CropBox: {cropBox}')
        outImage = Image.new('RGB',(width,height), color = 'white')

        # This state is local to this function
        state = PdfState(resources = self.xobj.inheritable.Resources,
                              glyphMap = self.glyphMap,
                              extractFontProgram = False,
                              makeSyntheticCmap = False)

        res = self.xobj.inheritable.Resources

        outTree = []
        for leaf in self.tree:

            cmd, args = leaf[0],leaf[1]
            _, _ = state.update(cmd, args)

            # Images
            image = None
            if cmd == 'Do':
                xobj = res.XObject[args[0]]
                if xobj.Subtype == PdfName.Image:
                    image = PdfImage(obj = xobj)

            if image is None:
                outTree.append(leaf)
                continue

            image.render()
            pil = image.get_pil()

            w, h = pil.size

            # A transform from image space ([0,0,w,h]) to canvas space
            t = MAT([1, 0, 0, -1, 0, canvasHeight]) \
                        * canvasBox.transformFrom(cropBox) \
                        * state.current_state.CTM \
                        * MAT([1/w, 0, 0, -1/h, 0, 1])

            # The inverse matrix; the order of elements is different in numpy

            box = BOX([0,0,w,h])

            tox = t * box

            if abs(t[1]*h) < canvasWidth and abs(t[2]*w) < canvasHeight \
                and abs(tox.w() - w) < 1 and abs(tox.h() - h) < 1:
                fx, fy = w/tox.w(), h/tox.h()
                t[1] = t[2] = 0
                t[0] *= fx
                t[3] *= fy
                tox = t * box
            elif pil.mode == '1':
                pil = pil.convert('L')

            # Transform to relative coords
            llx, lly = tox.ll()
            t1 = MAT([1,0,0,1,-llx,-lly]) * t 

            # Transform image
            a,d,b,e,c,f = t1.inv()
            pil = pil.transform((w,h),
                                        Image.AFFINE, (a,b,c,d,e,f),
                                        resample = Image.BICUBIC,
                                        fillcolor = 'white')

            # Make mask
            mask = ImageOps.invert(pil.convert('L'))

            llx = int(round(llx)); lly = int(round(lly))
            outImage.paste(pil, box = (llx, lly), mask = mask)


        # Delete original images
        if res.XObject:
            toDelete = []
            for xName, xObj in res.XObject.items():
                if xObj.Subtype == PdfName.Image:
                    toDelete.append(xName)
            for xName in toDelete:
                res.XObject[xName] = None

                # * MAT([1, 0, 0, -1, 0, canvasHeight]) \
        t = cropBox.transformFrom(canvasBox) \
                * MAT([canvasWidth, 0, 0, canvasHeight, 0, 0])

        # register background image
        xobj = PdfImage(pil = outImage).encode()
        xobj_name = self.register_xobj(xobj)

        # Update tree
        preamble = [
            ['q', []],
            ['cm', t],
            ['Do', [xobj_name]],
            ['Q', []]
        ]
        self.tree = preamble + outTree


    # # --------------------------------------------------------------- paint_images()

    # @staticmethod
    # def paint_images(xobj:PdfDict, chunks:list, dpi = 72):
    #     '''
    #     Paints the images contained in chunks
    #     '''

    #     # Set up the canvas
    #     cropBox = get_box(xobj)
    #     if cropBox is None: return None
    #     canvasWidth, canvasHeight = cropBox.w() * dpi/72, cropBox.h() * dpi/72
    #     canvasBox = BOX([0, 0, canvasWidth, canvasHeight])
    #     width, height = ceil(canvasWidth), ceil(canvasHeight)
    #     if width == 0 or height == 0: err(f'bad CropBox: {cropBox}')
    #     outImage = Image.new('RGB',(width,height), color = 'white')

    #     for image, matrix in chunks:

    #         if not isinstance(image, PdfImage): continue


    #         image.render()
    #         pil = image.get_pil()

    #         w, h = pil.size


    #         # A transform from image space ([0,0,w,h]) to canvas space
    #         t = MAT([1, 0, 0, -1, 0, canvasHeight]) \
    #                     * canvasBox.transformFrom(cropBox) \
    #                     * matrix \
    #                     * MAT([1/w, 0, 0, -1/h, 0, 1])

    #         # The inverse matrix; the order of elements is different in numpy

    #         box = BOX([0,0,w,h])


    #         tox = t * box

    #         if abs(t[1]*h) < canvasWidth and abs(t[2]*w) < canvasHeight \
    #             and abs(tox.w() - w) < 1 and abs(tox.h() - h) < 1:
    #             fx, fy = w/tox.w(), h/tox.h()
    #             t[1] = t[2] = 0
    #             t[0] *= fx
    #             t[3] *= fy
    #             tox = t * box
    #         elif pil.mode == '1':
    #             pil = pil.convert('L')

    #         # Transform to relative coords
    #         llx, lly = tox.ll()
    #         t1 = MAT([1,0,0,1,-llx,-lly]) * t 

    #         # Transform image
    #         a,d,b,e,c,f = t1.inv()
    #         pil = pil.transform((w,h),
    #                                     Image.AFFINE, (a,b,c,d,e,f),
    #                                     resample = Image.BICUBIC,
    #                                     fillcolor = 'white')

    #         # Make mask
    #         mask = ImageOps.invert(pil.convert('L'))

    #         llx = int(round(llx)); lly = int(round(lly))
    #         outImage.paste(pil, box = (llx, lly), mask = mask)

    #     return outImage

    # --------------------------------------------------------------- processText()

    def processText(self, xobjCache, options:dict = {}):
        '''
        Print text contained in the stream
        '''
        result = self.recurse(recursedFunction = PdfStreamEditor.processTextFunction,
                                xobjCache = xobjCache,
                                tree = None,
                                options = options)
        return result

    # --------------------------------------------------------------- processTextFunction()

    def processTextFunction(self, xobjCache:dict, tree:list=None, options:dict = {}):
        '''
        An auxiliary function used by .print_text(). Available options:
        {'regex':'', 'regexCheckPath':None, 'removeOCR':False, 'render':False, 'removeAlt':False}
        '''
        res = self.xobj.inheritable.Resources
        firstCall = tree == None
        if firstCall: tree = self.tree
        
        # Editing options

        regex = options.get('regex', '')
        regexCheckPath = options.get('regexCheckPath', None)
        removeOCR = options.get('removeOCR', False)
        render = options.get('render', False)
        removeAlt = options.get('removeAlt', False)

        # HIDE = options.get('HIDE', False)

        edit = regex != '' or removeOCR

        # This is edited tree if options call for editing
        outTree = []

        # A list of chunks - (data, matrix) tuples
        chunks = []

        removeOutlinedText = False

        # merge adjacent BT blocks into a single BT block, but insert '1 0 0 1 0 0 Tm' between their bodies
        buffer = []
        tree_new = []
        for leaf in tree:
            if leaf[0] == 'BT':
                if len(buffer) > 0:
                    buffer.append(['Tm', [1,0,0,1,0,0]])
                buffer += leaf[2]
            else:
                if len(buffer) > 0:
                    tree_new.append(['BT', [], buffer])
                    buffer = []
                tree_new.append(leaf)
        if len(buffer) > 0:
            tree_new.append(['BT', [], buffer])

        tree = tree_new

        # Main processing starts here
        for leaf in tree:

            cmd,args = leaf[0],leaf[1]

            chunk = self.state.update(cmd, args)

            if chunk != (None, None): chunks.append(chunk)

            cs = self.state.current_state

            # Debug
            if self.debug:

                eprint('-------------------------------------------------')
                eprint(f'Command: {cmd} {args}')
                eprint(f'State: {cs}')
                eprint(f'Chunk: {[chunk[0]]}, {chunk[1]}')
                if isinstance(chunk[0], str) and cs.font != None:
                    eprint(f'SpaceWidth: {cs.font.spaceWidth}')
                if cmd == 'Tf':
                    eprint(f'SetFont: {[cs.font.name, cs.fontSize]}')

            # Process calls to XObjects
            if cmd == 'Do':

                xobj = res.XObject[args[0]]

                # Forms
                if xobj.Subtype == PdfName.Form and xobj.stream != None:
                    chunks += [(data, cs.CTM * matrix) for data, matrix in xobjCache[id(xobj)]]

                # Images
                if xobj.Subtype == PdfName.Image and render:
                    image = PdfImage(obj = xobj)
                    w,h = image.w(), image.h()
                    assert w != 0 and h != 0
                    chunks.append((image, cs.CTM))

            # Process the nested BT/ET block a recursive call on the body
            modified = False

            # if [cmd,args] == ['k', ['0','0','0','0']]:
            #     print('HIDE = True')
            #     options['HIDE'] = True

            # if cmd == 'k' and args != ['0','0','0','0']:
            #     print('HIDE = False')
            #     options['HIDE'] = False

            if cmd == 'BT':

                btChunks = self.processTextFunction(xobjCache, leaf[2], options)

                chunks += btChunks

                if edit:

                    btText = PdfStreamEditor.chunks_to_text(btChunks)

                    discardText = (regex != '' and re.search(regex,btText) != None)
                    # discardText = options.get('HIDE')

                    containsOCR = any(len(kid[1])>0 and (kid[0],kid[1][0]) == ('Tr','3') for kid in leaf[2])

                    discardOCR = removeOCR and containsOCR

                    if discardText or discardOCR:

                        print(f'Removed text: {btText}')
                        self.isModified = True
                        modified = True

                        body = [l for l in leaf[2] if l[0] not in ['Tj', 'TJ', '"', "'", 'T*', 'Td', 'TD']]
                        outTree.append(['BT', [], body])

                        if regexCheckPath != None:
                            with open(regexCheckPath, 'a') as file:
                                file.write(btText + '\n')


                    # if removeOCR:

                    #     assert modified is False

                    #     body = []

                    #     for l in leaf[2]:

                    #         if l[0] == 'Tr':

                    #             if l[1][0] == '2':
                    #                 l[1][0] = 0
                    #                 modified = True

                    #             removeOutlinedText = l[1][0] == '1'
                    #             body.append(l)

                    #         else:
                    #             if removeOutlinedText and l[0] in ['Tj', 'TJ', '"', "'"]:
                    #                 modified = True
                    #             else:
                    #                 body.append(l)
    
                    #     if modified:
                    #         outTree.append(['BT', [], body])
                    #         self.isModified = True

            # Process marked document content (for removing alternate text)
            if cmd == 'BDC' and removeAlt:

                assert len(leaf) == 2
                assert len(args) == 2


                altKeys = [PdfName.E, PdfName.Alt, PdfName.ActualText]
                properties = args[1]

                if any(k in altKeys for k in properties):

                    text = '; '.join(properties[k] for k in properties if k in altKeys)
                    properties = {k:properties[k] for k in properties if k not in altKeys}
                    args[1] = properties
                    outTree.append([cmd, args])
                    self.isModified = True
                    modified = True
                    print('Removed Alt Text:', text)

            if not modified:
                outTree.append(leaf)

        if firstCall and self.isModified:
            self.tree = outTree
            self.set_stream(PdfStream.tree_to_stream(outTree))

        if firstCall and self.xobj.Matrix != None:
            xm = MAT(self.xobj.Matrix)
            chunks = [(data, xm * matrix) for data, matrix in chunks]
 
        return chunks

    # --------------------------------------------------------------- chunks_to_text()

    @staticmethod
    def chunks_to_text(chunks:list):
        '''
        Extract text from chunks. All text chunks consisting entirely of space runs are discarded and
        spaces are inferred/reconstructed from chunks' box matrices.
        '''

        # Analyze all text chunks to get text's left edge (xmin_global) and average symbol height
        bbox = None
        heights = []
        if len(chunks) == 0:
            return ''

        for chunk in chunks:
            chunkMatrix = chunk[1]
            box = chunkMatrix * BOX([0,0,1,1])
            heights.append(abs(box[3] - box[1]))
            bbox = box + bbox
        xmin_global = bbox[0]
        height_avg = sum(heights)/len(heights) if len(heights) > 0 else 1

        # Convert chunks to text
        chunks = [chunk for chunk in chunks if isinstance(chunk[0], str) and not re.search(r'^ *$', chunk[0])]
        chunks = sorted(chunks, key = lambda t: t[1])
        chunk_prev = None
        text = []
        for chunk in chunks:
            spacer = '' if chunk_prev is None else chunk[1].spacer(chunk_prev[1])
            chunk_prev = chunk
            if len(spacer) > 0 and spacer[0] == '\n': # Check if this is a new line
                box = chunk[1] * BOX([0,0,1,1])
                xmin = box[0]
                n_spaces = int((xmin - xmin_global) / (0.67 * height_avg))
                spacer2 = spacer + ' '*n_spaces
                text.append(spacer2 + chunk[0])
            else:
                text.append(spacer + chunk[0])
        return ''.join(text)


    # --------------------------------------------------------------- make_coords_relative()

    def make_coords_relative(self):
        path_construction_commands = ['m','l','c','v','y','re','h','W','W*']
        path_painting_commands = ['s','S','f','F','f*','B','B*','b','b*','n']
        # path_commands = path_construction_commands + path_painting_commands
        p = lambda x: f'{x:f}'.rstrip('0').rstrip('.')

        out = []
        tree_chunk_relative = []
        tree_chunk_original = []
        cm = None
        inside = False
        x,y = None, None

        for leaf in self.tree:
            cmd,args = leaf[0],leaf[1]

            if not inside:
                if cmd in ['m','re']: # a graphics block can actually start with a re command as well!
                    x,y = [float(a) for a in args[:2]]
                    inside = True
                    cm = ['cm',[1,0,0,1,p(x),p(y)]]
                    if cm[1][-2:] == ['0','0']: cm = None # Do not insert zero shifts
                    tree_chunk_relative.append([cmd,['0','0'] + args[2:]])
                    tree_chunk_original.append(leaf)
                else:
                    out.append(leaf)
            else:
                if cmd in path_construction_commands:
                    n = len(args) if cmd != 're' else 2
                    args_rel = args.copy()
                    for i in range(0,n,2): args_rel[i] = p(float(args[i]) - x)
                    for i in range(1,n,2): args_rel[i] = p(float(args[i]) - y)
                    tree_chunk_relative.append([cmd,args_rel])
                    tree_chunk_original.append(leaf)
                elif cmd in path_painting_commands:
                    inside = False
                    x,y = None,None
                    # Clipping paths cannot be put inside a q/Q pair: their effect outside the pair is lost
                    if any(x[0] in ['W','W*'] for x in tree_chunk_original):
                        out += tree_chunk_original
                        out.append(leaf)
                    else:
                        if cm != None: out.append(['q',[]]) ; out.append(cm)
                        out += tree_chunk_relative
                        out.append(leaf)
                        if cm != None: out.append(['Q',[]])
                    tree_chunk_original = []
                    tree_chunk_relative = []
                else:
                    err(f'unexpected command inside path: {leaf}')

        self.tree = out

    # --------------------------------------------------------------- create_xobjects()

    def create_xobjects(self,xobjects:dict):
        '''Turn duplicate blocks of PDF stream operators into XObjects in order to "define once, use many times".
        '''
        # Coordinate precision: value of 100 means keeping 2 digits after the point in coordinates, 1000 — 3 and so on
        # Lower precision reduces file size and can make the -x command more effective in matching glyphs;
        # To not limit the coordinate precision comment this out
        PRECISION=100 

        path_construction_commands = ['m','l','c','v','y','re','h','W','W*']
        path_painting_commands = ['s','S','f','F','f*','B','B*','b','b*','n']
        path_commands = path_construction_commands + path_painting_commands
        tree,xtree = [],[]
        counter = 0 # cumulative xobjects reference counter
        chunks = {} # reference counter for chunks that are candidates to become xobjects
        p = lambda x: f'{x:f}'.rstrip('0').rstrip('.')

        # --------------- First run: populate tree

        for i in range(len(self.tree)):
            leaf = self.tree[i]
            cmd,args = leaf[0],leaf[1]

            # Limit precision if PRECISION is defined
            if cmd in path_construction_commands:
                if PRECISION:
                    args = [f'{p(round(float(arg)*PRECISION)/PRECISION)}' for arg in args]
                else:
                    args = [f'{p(float(arg))}' for arg in args]
                leaf = [cmd,args]

            if cmd == 'cm' and PRECISION:
                args = args[0:4] + [f'{p(round(float(arg)*PRECISION)/PRECISION)}' for arg in args[4:]]
                leaf = ['cm',args] if args != ['1','0','0','1','0','0'] else None

            if cmd in path_commands and leaf != None: xtree.append(leaf); continue

            if len(xtree) > 0:

                if xtree[-1][0] not in path_painting_commands:
                    err(f'expected a path-painting command at the end of a graphics chunk:\n{xtree}')

                # Clipping paths should be included directly into the contents stream (no xobjects)
                if any(x[0] in ['W','W*'] for x in xtree):
                    tree += xtree
                else:
                    s = PdfStream.tree_to_stream(xtree)
                    md5 = hashlib.md5(s.encode("utf-8")).hexdigest()
                    try:
                        xobj = xobjects[md5]
                        xobj_name = self.register_xobj(xobj)
                        tree.append(['Do',[xobj_name]])
                        counter+=1
                    except: # if s's md5 is not in xobjects
                        if s in chunks: chunks[s] += 1
                        else: chunks[s] = 1
                        # store xtree & its string rep to avoid recomputing it in the 2nd run below
                        tree.append(['DoTemp',[s,xtree]])

                xtree = []

            if leaf != None: tree.append(leaf)

        # Flush xtree into tree
        if len(xtree)>0:
            if xtree[-1][0] not in path_painting_commands:
                err(f'expected a path-painting command at the end of a graphics chunk:\n{xtree}')
            tree += xtree

        # --------------- Second run: populate self.tree

        self.tree = []
        for i in range(len(tree)):
            leaf = tree[i]
            cmd,args = leaf[0],leaf[1]

            if cmd != 'DoTemp':
                self.tree.append(leaf)
            else:
                s,xtree = args
                md5 = hashlib.md5(s.encode("utf-8")).hexdigest()
                try:
                    xobj = xobjects[md5]
                    xobj_name = self.register_xobj(xobj)
                    self.tree.append(['Do',[xobj_name]])
                    counter+=1
                except: # if s's md5 is not in xobjects

                    if s not in chunks: err(f'chunk not in chunks: {s}')

                    # Compare gain from the use of xobjects to their overhead
                    # 0.4 is an empirical deflate compression factor for typical graphics streams
                    # 256 is a rough empirical estimate of the average xobjects overhead
                    if (chunks[s] - 1) * 0.4 * len(s) < 256:
                        self.tree += xtree
                    else:
                        xobj = IndirectPdfDict(
                            Name = PdfName(f'gl{len(xobjects)}'),
                            Type = PdfName.XObject,
                            Subtype = PdfName.Form,
                            FormType = 1,
                            # BBox = self.get_box(xtree) or PdfArray([-1000,-1000,1000,1000]),
                            BBox = PdfArray([-1000,-1000,1000,1000]), # b/c often parts of glyphs are put in xobj
                            Resources = PdfDict(ProcSet = [PdfName.PDF])
                        )
                        xobj.stream = s
                        xobjects[md5] = xobj

                        # debug
                        # if ref: tree.append(['rg',['1 0 0']])

                        xobj_name = self.register_xobj(xobj)
                        self.tree.append(['Do',[xobj_name]])
                        counter += 1

        return counter

    # --------------------------------------------------------------- register_xobj()

    def register_xobj(self, xobj:PdfDict):
        '''Registers xobj in the PDF page resources (required before you can call '/name Do')
        by safely setting resources.XObject[name] = xobj, where name == xobj.Name, unless xobj.Name
        is already in resources.XObject and id(xobj) != id(resources.XObject[xobj.Name]) -- a name collision.
        Such collisions are resolved by setting name == xobj.Name+'z'*N, where N is the minimal integer that
        resolves the collision. The variable name is then returned.
        
        Note 1: collision resolution may result in name != xobj.Name, which is ok
        since, according to the PDF specs, xobj.Name is itself entirely optional (we set it mostly for debugging).
        Note 2: if initially resources.XObjects does not exist it's created. 
        '''
        if self.xobj.Resources == None:
            self.xobj.Resources = PdfDict() if self.xobj.inheritable.Resources == None \
                                                else self.xobj.inheritable.Resources.copy()
        res = self.xobj.Resources
        if res.XObject == None: res.XObject = PdfDict()
        name = xobj.Name
        if name == None: name = PdfName('XYZ')
        while name in res.XObject:
            if id(xobj) == id(res.XObject[name]): return name
            name = PdfName(name.lstrip('/')+'z')
        res.XObject[name] = xobj
        return name