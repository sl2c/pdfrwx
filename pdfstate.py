#!/usr/bin/env python3


from pdfrw import PdfDict

from pdfrwx.common import err, warn, msg
from pdfrwx.pdffont import PdfFont, PdfTextString
from pdfrwx.pdffontglyphmap import PdfFontGlyphMap

class PdfState:

    def __init__(self, resources:PdfDict, glyphMap = PdfFontGlyphMap()):
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

        self.current_state = PdfDict(
            CTM = [1,0,0,1,0,0],
            Tm = [1,0,0,1,0,0], Tlm = [1,0,0,1,0,0], Tm_prev = [1,0,0,1,0,0],
            Tc = 0,  Tw = 0, Th = 100, Tl = 0, Trise = 0, Tmode = 0, 
            fontName = None, fontSize = None, font = None,
        )

    def update(self, cmd, args):
        '''
        Updates the graphics state; for text commands, returns the Unicode string that is printed by
        the command and the list of 8 coordinates corresponding to the 4 corners of the text rectangle:
        ```python
        (UnicodeString, [llx,lly,ulx,uly,lrx,lry,urx,ury])
        ```
        '''
        f = lambda arg: [float(a) for a in arg] if len(args) > 1 else float(args[0])
        multiply = lambda a,b: [a[0]*b[0] + a[2]*b[1], a[1]*b[0] + a[3]*b[1], a[0]*b[2] + a[2]*b[3],
                    a[1]*b[2] + a[3]*b[3], a[0]*b[4] + a[2]*b[5] + a[4], a[1]*b[4] + a[3]*b[5] + a[5]]

        cs = self.current_state

        # State stack commands
        if cmd == 'q': self.__stack.append(self.current_state.copy())
        if cmd == 'Q': self.current_state = self.__stack.pop().copy()

        # General geometry
        if cmd == 'cm': cs.CTM = multiply(cs.CTM, f(args))

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
            cs.Tlm = [1,0,0,1,0,0]
        if cmd == 'Tm':
            cs.Tlm = f(args)            
        if cmd in ['Td','TD']:
            cs.Tlm = multiply(cs.Tlm, [1,0,0,1] + f(args))
        if cmd in ['T*','"',"'"]:
            cs.Tlm = multiply(cs.Tlm, [1,0,0,1,0,-cs.Tl]) # There is a Tl sign typo in PDF Ref. sec. 5.3.1
        if cmd in ['BT','Tm','Td','TD','T*','"',"'"]:
            cs.Tm = cs.Tlm.copy()

        # Fonts
        if cmd == 'Tf':
            cs.fontName, cs.fontSize = args[0], float(args[1])
            try:
                cs.font = self.resources.Font[cs.fontName]
            except:
                raise ValueError(f'font name not in self.resources.Font: {cs.fontName}')
            fontId = id(cs.font)
            if fontId in self.__fontCache:
                cs.font = self.__fontCache[fontId]
            else:
                cs.font = PdfFont(cs.font, self.glyphMap)
                self.__fontCache[fontId] = cs.font
 
        # Text commands
 
        if cmd in ['Tj', 'TJ', "'", '"']:

            if cs.font == None:
                warn(f'font undefined: {cmd}, {args}; state: {cs}')
                return None, None, None
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
            # NB: the 0.25 factor is totally ad hoc and needs to be tested! See also same issue in self._get_gap()
            zTight = z if len(z) == 0 or isinstance(z[-1],str) else z[:-1]
            textString = ''.join(cs.font.decodeCodeString(t) if isinstance(t,str)
                                    else f' ' if t > cs.font.spaceWidth * cs.fontSize * .667 *.25 else f''
                                    for t in zTight)

            # This is the displacement by which a point at which the next symbol is placed is moved
            textWidth = sum(cs.font.width(t, isEncoded = True) * cs.fontSize if isinstance(t,str)
                            else t for t in z) * (cs.Th / 100.0)
            
            # This is the actual visual width the text string; it is textWidth minus the spacing at the end
            textWidthTight = sum(cs.font.width(t, isEncoded = True) * cs.fontSize if isinstance(t,str)
                                else t for t in zTight) * (cs.Th / 100.0)

            # Get gap
            gap = self._get_gap()

            # Calculate left coords of the tight text rectangle
            m = multiply(cs.CTM, cs.Tm)
            m = multiply(m, cs.font.fontMatrix)
            llx,lly = m[4],m[5]
            ulx = m[2] * cs.fontSize * cs.font.bbox[3] + m[4]
            uly = m[3] * cs.fontSize * cs.font.bbox[3] + m[5]

            # Calculate the tight updated Tm matrix
            TmTight = multiply(cs.Tm, [1,0,0,1,textWidthTight,0])

            # Calculate right coords tight text rectangle
            m = multiply(cs.CTM, TmTight)
            m = multiply(m, cs.font.fontMatrix)
            lrx,lry = m[4],m[5]
            urx = m[2] * cs.fontSize * cs.font.bbox[3] + m[4]
            ury = m[3] * cs.fontSize * cs.font.bbox[3] + m[5]

            # Form the text rectangle
            textRectTight = [llx,lly,ulx,uly,lrx,lry,urx,ury]
            # textRectTight = [round(x*1000)/1000 for x in textRect]

            # Update Tm, but not Tlm
            cs.Tm = multiply(cs.Tm, [1,0,0,1,textWidth,0])

            # Insert the newlines in text, if needed
            if cmd in ["'", '"', 'T*'] and abs(cs.Tl) > cs.fontSize:
                textString = '\n' + textString

            # Update cs.Tm_old
            cs.Tm_prev = cs.Tm.copy()

            return gap+textString, textWidth, textRectTight

        return None, None, None

    def _get_gap(self):
        cs = self.current_state
        T1,T2 = cs.Tm_prev, cs.Tm
        if cs.fontSize == None: return ''
        gap_h, gap_v = T1[4] - T2[4], T1[5] - T2[5]
        if abs(gap_v) > cs.fontSize * min(abs(T1[3]), abs(T2[3])):
            return '\n' if T2 != [1,0,0,1,0,0] else ''
        else:
            return ' ' if gap_h > 0.25 * cs.font.spaceWidth *.667 * T1[0] * cs.fontSize * (cs.Th / 100.0) else ''

#  ==========================================================================================

# class PdfState:

#     def __init__(self, Resources:PdfDict, glyphMap:PdfFontGlyphMap):
#         '''
#         Creates an instance of PdfState -- the class that keeps track of the graphics state.

#         Call PdfState.update(cmd,args) for each command in a PDF stream tree and read
#         the current state from PdfState.current_state. The attributes of current_state:

#             CTM = [1,0,0,1,0,0], # CTM matrix: a b c d e f cm
  
#             Tc = 0, # character spacing: charSpace Tc
#             Tw = 0, # word spacing: wordSpace Tw
#             Th = 100, # horizontal scaling: scale Tz
#             Tl = 0, # leading: leading TL
#             Trise = 0, # rise: rise Ts
  
#             Tmode = 0, # text rendering mode: render Tr

#             Font = PdfDict(), # text font; attributes: .font:PdfFont, .fontSize:float, .spaceWidth:float

#             Tm = [1,0,0,1,0,0], # Text matrix
#             Tlm = [1,0,0,1,0,0], # Text line matrix
#         '''

#         defaultState = PdfDict(
#             CTM = [1,0,0,1,0,0], # CTM matrix: a b c d e f cm
  
#             Tc = 0, # character spacing: charSpace Tc
#             Tw = 0, # word spacing: wordSpace Tw
#             Th = 100, # horizontal scaling: scale Tz
#             Tl = 0, # leading: leading TL
#             Trise = 0, # rise: rise Ts
  
#             Tmode = 0, # text rendering mode: render Tr

#             Font = PdfDict(), # text font; attributes: .font:PdfFont, .fontSize:float, .spaceWidth:float

#             Tm = [1,0,0,1,0,0], # Text matrix
#             Tlm = [1,0,0,1,0,0], # Text line matrix
#         )

#         self.__stack = []
#         self.current_state = defaultState
#         self.Tm_old = [1,0,0,1,0,0] # Text matrix, old value -- needed to calculate line/word gaps
#         self.Resources = Resources
#         self.glyphMap = glyphMap

#     def update(self, cmd, args):
#         '''
#         Updates the graphics state
#         '''
#         f = lambda arg: [float(a) for a in arg] if len(args) > 1 else float(args[0])
#         multiply = lambda a,b: [a[0]*b[0] + a[2]*b[1], a[1]*b[0] + a[3]*b[1], a[0]*b[2] + a[2]*b[3],
#                     a[1]*b[2] + a[3]*b[3], a[0]*b[4] + a[2]*b[5] + a[4], a[1]*b[4] + a[3]*b[5] + a[5]]

#         cs = self.current_state

#         if cmd == 'q': self.__stack.append(self.current_state.copy())
#         if cmd == 'Q': self.current_state = self.__stack.pop().copy()

#         if cmd == 'cm': cs.CTM = multiply(cs.CTM, f(args))

#         if cmd in ['Tc','Tw','Tz','TL','Ts']:
#             arg = float(args[0])
#             if cmd == 'Tc': cs.Tc = arg
#             if cmd == 'Tw': cs.Tw = arg
#             if cmd == 'Tz': cs.Th = arg
#             if cmd == 'TL': cs.Tl = arg
#             if cmd == 'Ts': cs.Trise = arg

#         if cmd == 'Tr':
#             cs.Tmode = int(args[0])
            
#         # Geometry

#         if cmd == 'BT':
#             cs.Tm = [1,0,0,1,0,0]
#             cs.Tlm = [1,0,0,1,0,0]
#             return ''

#         if cmd in ['Tm','Td','TD']:
#             if cmd == 'TD': cs.Tl = -float(args[1])
#             # update Tlm first as Tm may contain displacement contribution from text type-setting
#             if cmd in ['Td','TD']:
#                 cs.Tlm = multiply(cs.Tlm, [1,0,0,1] + f(args))
#             else: # Tm command
#                 cs.Tlm = f(args)
#             cs.Tm  = cs.Tlm.copy()
#             return ''
            
#         if cmd == '"':
#             cs.Tw, cs.Tc = f(args)

#         if cmd in ['T*','"',"'"]:           
#             cs.Tm = multiply(cs.Tm, [1,0,0,1,0,-cs.Tl])
#             cs.Tlm = cs.Tm.copy()

#         # Process font commands
#         if cmd == 'Tf':

#             res = self.Resources
#             fontName, fontSize = args[0], float(args[1])
#             if res == None or res.Font == None or fontName not in res.Font: err(f'cannot find font: {fontName}')

#             # This PdfDict will never be part of PDF; it's used for convenience
#             cs.Font = PdfDict(
#                 fontName = fontName,
#                 fontSize = fontSize,
#                 font = PdfFont(fontDict = res.Font[fontName], glyphMap = self.glyphMap)
#             )
#             cs.Font.fontNameReal = cs.Font.font.name
#             cs.Font.spaceWidth = cs.Font.font.width(' ')
#             if cs.Font.spaceWidth == 0: cs.Font.spaceWidth = cs.Font.font.widthDefault * 0.667

#         # Process text output commands
#         if cmd in ['Tj', 'TJ', "'", '"']:

#             font = cs.Font.font
#             if font == None:
#                 warn(f'font undefined: {cmd}, {args}; state: {self.current_state}')
#                 return None
#             s = args[2] if cmd == '"' else args[0] if len(args) > 0 else ''
#             if isinstance(s,str): s = [s] if s != '' else []

#             # Decompose string arguments of text operators down to single chars
#             z = []
#             for token in s:
#                 if token[0] in '<(':
#                     isCID = cs.Font.font.is_cid() and not isinstance(cs.Font.font.font.Encoding, PdfDict) # 1- or 2-byte string?
#                     codes = PdfTextString(token).to_codes(isCID=isCID)
#                     # Word spacing (Tw) is applied to every occurrence of the single-byte character code 32 in
#                     # a string when using a simple font or a composite font that defines code 32 as a single-byte code.
#                     # It does not apply to occurrences of the byte value 32 in multiple-byte codes (PDF Ref. 1.7, sec. 5.2.2)
#                     space = lambda e: (cs.Tc + (cs.Tw if token[0] == '(' and e == ' ' else 0))/cs.Font.fontSize
#                     z += [x for e in codes for x in (e,space(e))]
#                 else:
#                     if len(z) == 0: z.append(0)
#                     z[-1] -= float(token)/1000

#             # don't interpret the last displacement as space since it may be followed a negative displacement
#             # in TD etc; instead, just ignore it, but include it in the width calculation (next line)
#             # this way it will contribute to cs.Tm and will be interpreted as space, if necessary, later
#             textString = ''.join([ font.decodeCodeString(t) if isinstance(t,str)
#                                     else f' ' if t > cs.Font.spaceWidth * 0.25 and i != len(z)-1
#                                     else f''
#                                     for i,t in enumerate(z) ])

#             textWidth = sum(font.width(t, isEncoded = True) if isinstance(t,str) else t for t in z) \
#                             * cs.Font.fontSize * (cs.Th / 100.0)

#             # Get gap
#             gap = self._get_gap()

#             # Update Tm, but not Tlm
#             cs.Tm = multiply(cs.Tm, [1,0,0,1,textWidth,0])

#             if cmd in ["'", '"', 'T*'] and abs(cs.Tl) > cs.Font.fontSize :
#                 textString = '\n' + textString

#             # Update cs.Tm_old
#             self.Tm_old = cs.Tm.copy()

#             return gap+textString

#         return None

#     def _get_gap(self):
#         cs = self.current_state
#         T1,T2 = self.Tm_old, cs.Tm
#         if cs.Font.fontSize == None: return ''
#         gap_h, gap_v = T1[4] - T2[4], T1[5] - T2[5]
#         if abs(gap_v) > cs.Font.fontSize * min(abs(T1[3]), abs(T2[3])):
#             return '\n' if T2 != [1,0,0,1,0,0] else ''
#         else:
#             return ' ' if gap_h > 0.5 * cs.Font.spaceWidth * T1[0] * cs.Font.fontSize * (cs.Th / 100.0) else ''


