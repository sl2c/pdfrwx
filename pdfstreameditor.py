#!/usr/bin/env python3

import re
from sly import Lexer, Parser

import sys, os, re
from math import sqrt
from typing import Callable

# Try using: github.com/sarnold/pdfrw as it contains many fixes compared to pmaupin's version
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfDict, IndirectPdfDict, PdfName

from pdfrwx.common import err,msg,warn,eprint, encapsulate
from pdfrwx.pdffont import PdfFont, PdfFontUtils, PdfTextString
from pdfrwx.pdffontencoding import PdfFontEncoding
from pdfrwx.pdfstreamparser import PdfStream
from pdfrwx.djvusedparser import DjVuSedLexer, DjVuSedParser
from pdfrwx.pdfstate import PdfState
from pdfrwx.pdffontglyphmap import PdfFontGlyphMap
from pdfrwx.pdffilter import PdfFilter

import xml.etree.ElementTree as ET # for parsing hOCR files


from pdfrwx.common import err,warn,eprint

# ========================================================================== class PdfStreamEditor

class PdfStreamEditor:

    def __init__(self, xobj:PdfDict, glyphMap:PdfFontGlyphMap,
                    textOnly:bool=False, graphicsOnly:bool=False, normalize=False, debug:bool=False):
        '''
        Parse self.stream and store the resulting parsed PDF stream tree in self.tree.
        If textOnly==True, only the text and state operators are parsed, i.e. the bare minimum that allows
        to extract full text–related information from the stream. Correspondingly, if graphicsOnly==True,
        only the graphics operators are parsed.

        If normalize == True, text operators are normalized: all text operators are converted
        to either Tj or TJ, and Tw operators are removed altogether by introducing explicit
        word spacings in text strings, if needed.
        '''
        self.xobj = xobj
        self.glyphMap = glyphMap
        self.textOnly = textOnly
        self.graphicsOnly = graphicsOnly
        self.normalize = normalize
        self.debug = debug

        # Parse the stream tree
        stream = self.get_stream()
        if stream == None: stream = ''
        self.tree = PdfStream.stream_to_tree(stream, textOnly, graphicsOnly)
        if normalize:
            self.tree = self.normalize_text_operators(self.tree)

    def normalize_text_operators(self, tree:list, state:PdfState = None) -> list:
        '''
        Normalizes text operators by replacing:

        '," --> Tc, Tw, Tj
        Tw  --> deleted, its effect absorbed in the explicit character displacement values in TJ
        TD  --> TL, Td
        '''
        if state == None: state = PdfState(self.xobj.inheritable.Resources, self.glyphMap)
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
                state.update('TL',[-args[1]])
                result.append(['TL',[-args[1]]])
                cmd = 'Td'

            if cmd in ['Tj', 'TJ'] and cs.Tw != 0 and not cs.font.is_cid():
                args = [self.apply_Tw(args[0], cs.Tw)]
                cmd = 'Tj' if isinstance(args[0],str) else 'TJ'

            state.update(cmd, args)

            if cmd != 'Tw':
                result.append([cmd,args])

        return result

    def apply_Tw(self, s, Tw):
        '''
        Introduces explicit word spacing (Tw) in text operator strings. After applying this function
        to all such strings, the Tw operators can be dropped from the stream entirely.
        '''
        # Word spacing (Tw) is applied to every occurrence of the single-byte character code 32 in
        # a string when using a simple font or a composite font that defines code 32 as a single-byte code.
        # It does not apply to occurrences of the byte value 32 in multiple-byte codes (PDF Ref. 1.7, sec. 5.2.2)
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
                l = [l[0]] + [a for c in l[1:] for a in (f'{-Tw * 1000:f}', c)]
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
                editor = type(self)(x, self.glyphMap, textOnly = self.textOnly, graphicsOnly=self.graphicsOnly,
                                    normalize = self.normalize, debug = self.debug)
                xobjCache[id(x)] = editor.recurse(recursedFunction, xobjCache, *args, **kwarg)

        return recursedFunction(self, xobjCache, *args, **kwarg)

    # --------------------------------------------------------------- processText()

    def processText(self, xobjCache, options:dict = {}):
        '''
        Print text contained in the stream
        '''
        return self.recurse(PdfStreamEditor.processTextFunction, xobjCache=xobjCache, tree=None, state=None, options=options)

    def processTextFunction(self, xobjCache:dict, tree:list=None, state:PdfState = None, options:dict = {}):
        '''
        An auxiliary function used by .print_text()
        '''
        superBox = lambda a,b: [min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3])]

        multiply = lambda a,b: [a[0]*b[0] + a[2]*b[1], a[1]*b[0] + a[3]*b[1], a[0]*b[2] + a[2]*b[3],
                    a[1]*b[2] + a[3]*b[3], a[0]*b[4] + a[2]*b[5] + a[4], a[1]*b[4] + a[3]*b[5] + a[5]]
        
        transform_box = lambda a,b: [a[0]*b[0] + a[2]*b[1] + a[4], a[1]*b[0] + a[3]*b[1] + a[5],
                                    a[0]*b[2] + a[2]*b[3] + a[4], a[1]*b[2] + a[3]*b[3] + a[5]]
        
        inside = lambda x,y,c: True if c == None else (c[0]<x<c[2] and c[1]<y<c[3])

        res = self.xobj.inheritable.Resources
        firstCall = tree == None
        if firstCall: tree = self.tree
        if state == None: state = PdfState(self.xobj.inheritable.Resources, self.glyphMap)

        # Editing options
        regex = options.get('regex', '')
        removeOCR = options.get('removeOCR', False)
        cropBox = options.get('cropBox', None)
        edit = regex != '' or removeOCR
        
        outText = ''
        outTree = []

        outBBoxDefault = [1000000,1000000,-1000000,-1000000]
        outBBox = outBBoxDefault
        allBoxes = []

        isModified = False

        for leaf in tree:
            cmd,args = leaf[0],leaf[1]

            # Old current_state
            cs = state.current_state

            # Cursor coords before the command
            m = multiply(cs.CTM, cs.Tm)
            x,y = m[4], m[5]

            cmdText, cmdWidth, cmdRect = state.update(cmd,args)

            # Updated current_state
            cs = state.current_state

            if cmdText != None:
                llx,lly,ulx,uly,lrx,lry,urx,ury = cmdRect
                x1 = min(min(llx,ulx),min(lrx,urx))
                x2 = max(max(llx,ulx),max(lrx,urx))
                y1 = min(min(lly,uly),min(lry,ury))
                y2 = max(max(lly,uly),max(lry,ury))
                cmdBBox = [x1,y1,x2,y2]
                if inside(x,y,cropBox):
                    outBBox = superBox(outBBox, cmdBBox)
                    allBoxes.append(cmdBBox)

            if self.debug:
                outText += '-------------------------------------------------\n'
                outText += f"Command: {cmd} {args}\n"
                outText += f'State: {cs}\n'
                if cmdText != None:
                    # textString = re.sub(r'\n','[newline]',textString)
                    outText += f"Text: {[cmdText]}\n"
                    outText += f"BBox: {cmdBBox}\n"
                    if cs.font != None:
                        outText += f"SpaceWidth: {cs.font.spaceWidth}\n"
                        if cs.font.encoding != None:
                            outText += f"Encoding: {cs.font.encoding.cc2glyphname}\n"

                if cmd == 'Tf':
                    outText += f'SetFont: {[cs.font.name, cs.fontSize]}\n'
            else:
                outText += cmdText if cmdText != None and inside(x,y,cropBox) else ''

            # Process calls to XObjects
            if cmd == 'Do':
                xobj = res.XObject[args[0]]
                if xobj.Subtype == PdfName.Form and xobj.stream != None:
                    doText, doBBox, _, _, doBoxes = xobjCache[id(xobj)]

                    # Transform to current coordinate frame
                    if self.debug:
                        outText += f'xobj.Matrix: {xobj.Matrix}\n'
                        outText += f'doBBox: {doBBox}\n'
                        outText += f'doBoxes: {doBoxes}\n'
                    doBBox = transform_box(cs.CTM, doBBox)
                    doBoxes = [transform_box(cs.CTM, b) for b in doBoxes]

                    outText += doText
                    outBBox = superBox(outBBox, doBBox)
                    allBoxes += doBoxes

            # Process the nested BT/ET block a recursive call on the body
            if cmd == 'BT':
                blockText, blockBBox, _, _, blockBoxes = self.processTextFunction(xobjCache, leaf[2], state, options)
                outText += blockText
                outBBox = superBox(outBBox, blockBBox)
                allBoxes += blockBoxes

                if edit:
                    discardText = (regex != '' and re.search(regex,blockText) != None)
                    # discardOCR = False if not removeOCR else \
                        # any(len(kid[1])>1 and (kid[0],kid[1][0]) == ('Tf','/OCR') for kid in leaf[2])
                    discardOCR = False if not removeOCR else \
                        any(len(kid[1])>1 and (kid[0],kid[1][0]) == ('Tr','3') for kid in leaf[2])
                    if discardText or discardOCR:
                        print(f'Removed text: {blockText}'); isModified = True
                        body = [l for l in leaf[2] if l[0] not in ['Tj', 'TJ', '"', "'"]]
                        outTree.append(['BT',[],body])
                    else:
                        outTree.append(leaf)
            else:
                if edit: outTree.append(leaf)

        if isModified and firstCall:
            self.tree = outTree
            stream = PdfStream.tree_to_stream(outTree)
            self.set_stream(stream)

        # Adjust boxes if self.xobj.Matrix is present
        if self.xobj.Matrix != None and firstCall:
            m = [float(x) for x in self.xobj.Matrix]
            outBBox = transform_box(m,outBBox)
            allBoxes = [transform_box(m,b) for b in allBoxes]

        return outText, outBBox, outTree, isModified, allBoxes

    # --------------------------------------------------------------- make_coords_relative()

    def make_coords_relative(self, tree:list):
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

        for leaf in tree:
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

        return out

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
                    try:
                        xobj = xobjects[s]
                        xobj_name = self.register_xobj(xobj,self.resources)
                        tree.append(['Do',[xobj_name]])
                        counter+=1
                    except: # if s is not in xobjects
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
                try:
                    xobj = xobjects[s]
                    xobj_name = self.register_xobj(xobj,self.resources)
                    self.tree.append(['Do',[xobj_name]])
                    counter+=1
                except: # if s is not in xobjects

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
                            BBox = self.get_bbox(xtree),
                            Resources = PdfDict(ProcSet = [PdfName.PDF])
                        )
                        xobj.stream = s
                        xobjects[s] = xobj

                        # debug
                        # if ref: tree.append(['rg',['1 0 0']])

                        xobj_name = self.register_xobj(xobj,self.resources)
                        self.tree.append(['Do',[xobj_name]])
                        counter += 1

        return counter

    # --------------------------------------------------------------- register_xobj()

    def register_xobj(self,xobj:PdfDict,resources:PdfDict):
        '''Registers xobj in the PDF page resources (required before you can call: 'name Do')
        by safely setting resources.XObject[name] = xobj, where name == xobj.Name, unless xobj.Name
        is already in resources.XObject and id(xobj) != id(resources.XObject[xobj.Name]) -- a name collision.
        Such collisions are resolved by setting name == xobj.Name+'z'*N, where N is the minimal integer that
        resolves the collision. The variable name is then returned.
        
        Note 1: collision resolution may result in name != xobj.Name, which is ok
        since, according to the PDF specs, xobj.Name is itself entirely optional (we set it mostly for debugging).
        Note 2: if initially resources.XObjects does not exist it's created. 
        '''
        if resources.XObject == None: resources.XObject = PdfDict()
        name = xobj.Name
        while name in resources.XObject:
            if id(xobj) == id(resources.XObject[name]): return name
            name = PdfName(name.lstrip('/')+'z')
        resources.XObject[name] = xobj
        return name

    # --------------------------------------------------------------- get_bbox()

    def get_bbox(self,tree:list):
        '''Get bounding box for everythin inside the parsed PDF stream tree;
        a placeholder for now, this needs to be coded'''
        return [-1000, -1000, 2000, 2000]


    # def fixCMaps(self, tree:list, bbox:PdfArray, cMapsTable:dict, xobjCache:list, streamCumulative='', level=0, debug=False):
    #     '''Fix ToUnicode CMaps by running OCR on the rendered chunks of text defined in the parsed PDF stream tree
    #     '''
    #     res = self.resources
    #     fontname,cmap = None,None
    #     count = 0

    #     for leaf in tree:
    #         cmd,args = leaf[0],leaf[1]

    #         # Process font commands
    #         if cmd == 'Tf': fontname = args[0]
    #         if cmd == 'q': self.state.append(fontname)
    #         if cmd == 'Q': fontname = self.state.pop()

    #         if fontname != None:
    #             if res == None or res.Font == None or fontname not in res.Font: err(f'font not found: {fontname}')
    #             font = res.Font[fontname]
    #             cmap = cMapsTable[id(font)] if id(font) in cMapsTable else {}
    #             width = 2 if font.Subtype == '/Type0' else 1
            
    #         cmdStr = ' '.join(f'{arg}' for arg in args)+' '+cmd+'\n' if len(args) != 0 else cmd+'\n'

    #         # Process text command
    #         if cmd in ['Tj','TJ',"'",'"']:
    #             count += 1
    #             streamSnapshot = streamCumulative + cmdStr + 'ET '*level + 'Q '*len(self.state)

    #             eprint(f'{cmd} {args} ---> {cmdStr}')

    #             # create a PDF document
    #             pdfOut = PdfWriter('test.pdf')

    #             page = PdfDict(
    #                 Type = PdfName.Page,
    #                 MediaBox = bbox,
    #                 Contents = IndirectPdfDict(stream=streamSnapshot),
    #                 Resources = res
    #             )

    #             if count == 5:
    #                 pdfOut.addPage(page)
    #                 pdfOut.write()
    #                 sys.exit()

    #             if cmd in ['Tj', "'"]: hexString = PdfUniversalString(args[0])
    #             elif cmd == '"': hexString = PdfUniversalString(args[2])
    #             elif cmd == 'TJ':
    #                 hexString = ''.join([PdfUniversalString(t).hex() for t in args[0] if t[0] in '<('])

    #             # unicode = self.cmap.hexToUnicode(hexString)

    #         # Process XObjects that are being called in the stream
    #         elif cmd == 'Do':
    #             xobjName = args[0]
    #             if res.XObject == None or xobjName not in res.XObject: err(f'XObject not found: {xobjName}')
    #             xobj = res.XObject[xobjName]
    #             if id(xobj) not in xobjCache and xobj.Subtype == PdfName.Form and xobj.stream != None:
    #                 xobjEditor = PdfStreamEditor(xobj.stream, xobj.inheritable.Resources)
    #                 xobjEditor.tree = xobjEditor.parse_stream(textOnly=True)
    #                 xobjEditor.fixCMaps(xobjEditor.tree, xobj.bbox, cMapsTable, xobjCache, '', 0, debug)
    #                 xobjCache.append(id(xobj))

    #         # Just update streamCumulative if cmd is non-text and non-xobj
    #         else:
    #             streamCumulative += cmdStr

    #         # Process nested operators by calling self.treeToText() recursively on kids
    #         if len(leaf) == 3: self.fixCMaps(leaf[2], bbox, cMapsTable, xobjCache, streamCumulative, level+1, debug)
