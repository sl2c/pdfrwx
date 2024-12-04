#!/usr/bin/env python3

import re

from pdfrw import PdfDict, PdfName

from .common import err, warn, msg
from .pdffont import PdfFont, PdfTextString
from .pdffontglyphmap import PdfFontGlyphMap
from .pdfgeometry import VEC, MAT, BOX

class PdfState:

    def __init__(self,
                 resources:PdfDict,
                 glyphMap = PdfFontGlyphMap(),
                 extractFontProgram:bool = False,
                 makeSyntheticCmap:bool = False):
        '''
        Creates an instance of PdfState -- the class that keeps track of the graphics state.

        Call PdfState.update(cmd,args) for each command in a PDF stream tree and read
        the current state from PdfState.current_state, which has the following attributes:

        ```
        CTM = [1,0,0,1,0,0], # the CTM matrix

        Tm = [1,0,0,1,0,0], # the text matrix
        Tlm = [1,0,0,1,0,0], # the text line matrix

        Tc = 0, # character spacing: charSpace Tc
        Tw = 0, # word spacing: wordSpace Tw
        Th = 100, # horizontal scaling: scale Tz
        Tl = 0, # leading: leading TL
        Trise = 0, # rise: rise Ts
        Tmode = 0, # text rendering mode: render Tr

        fontName = None, # current font name (the key in Resources.Font, set by the Tf command)
        fontSize = None, # current font size (set by the Tf command)
        font = None, # current font: PdfFont(Resources.Font[fontName])
        ```
        '''
        self.__stack = []
        self.__fontCache = {}
        self.resources = resources
        self.glyphMap = glyphMap

        self.extractFontProgram = extractFontProgram
        self.makeSyntheticCmap = makeSyntheticCmap

        self.current_state = PdfDict(
            CTM = MAT(),
            Tm = MAT(), Tlm = MAT(), Tm_prev = MAT(),
            Tc = 0,  Tw = 0, Th = 100, Tl = 0, Trise = 0, Tmode = 0, 
            fontName = None, fontSize = None, font = None,
        )

    def update(self, cmd, args):
        '''
        Updates the graphics state.

        If cmd is a text command (`Tj`, `TJ` etc.) , returns a tuple `(text, matrix)`,
        where `text` is a Unicode string and `matrix` is a transformation matrix such
        that it transforms a unit rectangle at the origin (`[0,0,1,1]`) into the text's bounding box.

        If cmd is not a text command, returns a tuple `(None, None)`.
        '''
        f = lambda arg: [float(a) for a in arg] if len(args) > 1 else float(args[0])

        cs = self.current_state

        # State stack commands
        if cmd == 'q': self.__stack.append(self.current_state.copy())
        if cmd == 'Q':
            try: self.current_state = self.__stack.pop().copy()
            except: warn(f'extra Q ignored')

        # General geometry
        if cmd == 'cm': cs.CTM = cs.CTM * MAT(args)

        # Text state parameters        
        if cmd in ['Tc','Tw','Tz','TL','Ts']:
            arg = float(args[0])
            if cmd == 'Tc': cs.Tc = arg
            if cmd == 'Tw': cs.Tw = arg
            if cmd == 'Tz': cs.Th = arg
            if cmd == 'TL': cs.Tl = arg
            if cmd == 'Ts': cs.Trise = arg
        if cmd == 'Tr':
            cs.Tmode = int(args[0])
        if cmd == 'TD':
            cs.Tl = -float(args[1])
        if cmd == '"':
            cs.Tw, cs.Tc = f(args)
            
        # Geometry
        if cmd == 'BT':
            cs.Tlm = MAT()
        if cmd == 'Tm':
            cs.Tlm = MAT(args)            
        if cmd in ['Td','TD']:
            cs.Tlm = cs.Tlm * MAT([1,0,0,1] + args)
        if cmd in ['T*','"',"'"]:
            cs.Tlm = cs.Tlm * MAT([1,0,0,1,0,-cs.Tl]) # There is a Tl sign typo in PDF Ref. sec. 5.3.1
        if cmd in ['BT','Tm','Td','TD','T*','"',"'"]:
            cs.Tm = cs.Tlm.copy()

        # Fonts
        if cmd == 'Tf':
            cs.fontName, cs.fontSize = args[0], float(args[1])
            cs.fontName = re.sub(r'#20',' ', cs.fontName) # !!! DEAL WITH THE CODE IN LITERAL NAMES; Pdf. Ref. sec. 3.2.4
            res = self.resources
            cs.font = res.Font[cs.fontName]
            if cs.font == None: raise ValueError(f'font {cs.fontName} not in not in resources.Font: {res.Font}')
            fontId = id(cs.font)
            # warn(f'cs.font: {cs.font}, {cs.fontName} -> {res.Font}')
            if fontId in self.__fontCache:
                cs.font = self.__fontCache[fontId]
            else:
                cs.font = PdfFont(font = cs.font,
                                    glyphMap = self.glyphMap,
                                    extractFontProgram = self.extractFontProgram,
                                    makeSyntheticCmap = self.makeSyntheticCmap)
                self.__fontCache[fontId] = cs.font
 
        # Text commands
 
        if cmd in ['Tj', 'TJ', "'", '"']:

            if cs.font == None:
                warn(f'font undefined: {cmd}, {args}; state: {cs}')
                return (None, None)
            s = args[2] if cmd == '"' else args[0] if len(args) > 0 else ''
            if isinstance(s,str): s = [s] if s != '' else []

            # Decompose string arguments of text operators down to single chars
            z = []
            isCID = cs.font.is_cid() # 1- or 2-byte codes?
            for token in s:
                if token[0] in '<(':
                    codes = PdfTextString(token).to_codes(isCID=isCID)
                    # Word spacing (Tw) is applied to every occurrence of the single-byte character code 32 in
                    # a string when using a simple font or a composite font that defines code 32 as a single-byte code.
                    # It does not apply to occurrences of the byte value 32 in multiple-byte codes (PDF Ref. 1.7, sec. 5.2.2)
                    space = lambda e: cs.Tc + (cs.Tw if not isCID and e == ' ' else 0)
                    z += [x for e in codes for x in (e,space(e))]
                else:
                    # Note: the numbers in the TJ operator arguments array are displacements that are expressed
                    # in neither the text space units, nor the glyph space units:
                    # "The number is expressed in thousandths of a unit of text space" (PDF Ref sec 5.3.2)
                    if len(z) == 0: z.append(0)
                    z[-1] -= float(token)*cs.fontSize/1000

            # don't interpret the last displacement as space since it may be followed a negative displacement
            # in TD etc; instead, just ignore it, but include it in the width calculation (next line)
            # this way it will contribute to cs.Tm and will be interpreted as space, if necessary, later
            zTight = z if len(z) == 0 or isinstance(z[-1],str) else z[:-1]

            zTight = z
            shiftTight = 0
            while len(zTight) > 0:
                if isinstance(zTight[0], float): shiftTight += zTight[0]; zTight = zTight[1:] ; continue
                if isinstance(zTight[-1], float): zTight = zTight[:-1] ; continue
                if zTight[0] == '': zTight = zTight[1:] ; continue
                if zTight[-1] == '': zTight = zTight[:-1] ; continue
                break
            shiftTight *= (cs.Th / 100.0)

            textString = ''.join(cs.font.decodeCodeString(t) if isinstance(t,str)
                                    else f' ' if t > cs.font.spaceWidth * cs.fontSize * .5 else f''
                                    for t in zTight)

            # This is the displacement by which a point at which the next symbol is placed is moved
            textWidth = sum(cs.font.width(t, isEncoded = True) * cs.fontSize if isinstance(t,str)
                            else t for t in z) * (cs.Th / 100.0)
            
            # This is the actual visual width the text string; it is textWidth minus the spacing at the end
            textWidthTight = sum(cs.font.width(t, isEncoded = True) * cs.fontSize if isinstance(t,str)
                                else t for t in zTight) * (cs.Th / 100.0)

            # Get gap
            # gap = self._get_gap()

            # Calculate left coords of the tight text rectangle
            fs = cs.fontSize
            fontBox = (MAT([fs,0,0,fs,0,0]) * cs.font.fontMatrix) * cs.font.bbox
            textHeight = fontBox[3] - fontBox[1]
            baseLine = fontBox[1]
            if textHeight == 0: raise ValueError(f'textHeight == 0, font.bbox = {cs.font.bbox}, font.fontMatrix = {cs.font.fontMatrix}')
            if textWidthTight == 0: textWidthTight = 0.001
            textMatrix = cs.CTM * cs.Tm * MAT([textWidthTight, 0, 0, textHeight, shiftTight, baseLine])
            textMatrix = MAT([round(x*1000)/1000 for x in textMatrix])
 
            # Update Tm, but not Tlm
            cs.Tm = cs.Tm * MAT([1,0,0,1,textWidth,0])

            # Update cs.Tm_prev
            cs.Tm_prev = cs.Tm.copy()


            return (textString, textMatrix)

        return (None, None)

