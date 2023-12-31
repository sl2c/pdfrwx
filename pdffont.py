#!/usr/bin/env python3

from gettext import find
from multiprocessing.sharedctypes import Value
import re

from pdfrw import PdfDict, IndirectPdfDict, PdfString, PdfArray, PdfName, py23_diffs

from pdfrwx.common import err, warn, msg
from pdfrwx.pdffontencoding import PdfFontEncoding
from pdfrwx.pdffontglyphmap import PdfFontGlyphMap
from pdfrwx.pdffontcore14 import PdfFontCore14
from pdfrwx.pdffontcmap import PdfFontCMap
from pdfrwx.pdffilter import PdfFilter

# ================================================== class PdfTextString

class PdfTextString(str):

    '''
    This is an adaptation of the PdfString class — which is part of pdfrw, and represents
    strings that are used outside of content streams — to a class that represents strings
    that are used in content streams as arguments of the text typesetting operators.
    The new class is called PdfTextString and has no relation to the so-called PDF text string type,
    PDF ref. 1.7, sec. 3.8, whatever that means.

    The fact is that the PDF string objects can be unambiguously converted to/from byte strings. However,
    to use a character string in PDF one needs to somehow convert it to a bytes string first (encoding)
    and then convert the bytes string to a PDF string object (formatting).
    
    It turns out that the encoding part of the process is slightly different for strings inside/outside content
    streams:  the strings used in the text typesetting operators in contents streams
    do not actually encode Unicode strings, but rather strings of 1/2-byte character codes (we
    shall call them "code strings"). In particular, when encoding the code strings using
    the 2-byte per char PDF hex string format (using the UTF-16-BE encoding),
    no BOM (byte order mark) should be inserted at the beginning of the string.

    Unfortunately, encoding of the strings used in contents streams is not well documented in the
    PDF reference; encoding of the strings used outside of the content streams is described in
    secs. 3.2.3. and 3.8.1.

    Given this, the PdfTextString class adopts the to_bytes()/from_bytes() methods unchanged from the
    parent class, and introduces two new methods: to_codes()/from_codes() that convert to/from the
    code strings represented by the Python str class. In this representation, there's a one-to-one
    correspondence between the chars of the string and the character codes, code == ord(char),
    and so len(code_string) exactly equals the number of character codes in it.

    UPD: We now overload the .from_bytes() function as it turns out that the pdfrw version has a bug;
    see function's help for more info.
    UPD2: We no longer derive the class from pdfrw's PdfString class; we derive it from Python's str
    class instead.
    '''

    def format(self):
        '''
        Returns the format of self: 'literal', 'hex', or None if format is invalid.
        '''
        if self.startswith('(') and self.endswith(')'): return 'literal'
        elif self.startswith('<') and self.endswith('>'): return 'hex'
        else: None

    def to_codes(self, isCID = False):
        return self.to_bytes().decode('utf-16-be' if isCID else 'latin1')

    @classmethod
    def from_codes(cls, codeString:str, format:str = 'auto', forceCID = False):
        if format not in ['auto', 'hex', 'literal']:
            raise ValueError(f'format should be one of: auto, hex, literal')
        isCID = any(ord(c)>255 for c in codeString) or forceCID
        suitableFormat = 'hex' if isCID else 'literal'
        format = format if format != 'auto' else suitableFormat # Do not let from_bytes() decide on the format
        bytes = codeString.encode('utf-16-be' if isCID else 'latin1')
        return cls.from_bytes(bytes, bytes_encoding=format)
    
    def to_bytes(self):
        '''
        Converts a PDF string to bytes
        '''
        format = self.format()
        if format == None: raise ValueError(f'invalid PDF string: {self}')
        s = self[1:-1]

        if format == 'hex':
            try: return bytes.fromhex(s)
            except: raise ValueError(f'invalid PDF hex string: {self}')

        if format == 'literal':
            # Normalize a PDF literal string according to specs (see PDF Ref sec. 3.2.3)
            s = re.sub(r'\r\n','\n',s) # any end-of-line marker not preceded by backslash is equivalent to \n
            s = re.sub(r'\r','\n',s) # any end-of-line marker not preceded by backslash is equivalent to \n
            s = re.sub(r'\\\n','',s) # a combination of backslash followed by any end-of-line marker is ignored (=no new line)
            s = re.sub(r'(?<!\\)\\(?![nrtbf()\\0-7])','', s) # if \ is not foll-d by a special char & not prec-d by \ ignore it
            s = s.replace('\\(','(').replace('\\)',')')

            # First encode() creates bytes, then decode() interprets escapes contained in bytes
            # (latin1 encoding used in the first step translate the escapes literally)
            try: return s.encode('latin1').decode('unicode-escape').encode('latin1')
            except: raise ValueError(f'invalid PDF literal string: {self}')

    @classmethod
    def from_bytes(cls, raw:bytes, bytes_encoding='auto'):
        '''
        Overload the buggy PdfString.from_bytes() from pdfrw, which outputs \\r unescaped which makes it
        identical to \\n according to PDF Ref. (Acrobat interprets it as \\n too).
        We escape all chars outside the 32..126 code range (yes, we escape b'\\x7f').
        '''
        if bytes_encoding not in ('hex', 'literal', 'auto'):
            raise ValueError(f'Invalid bytes_encoding value: {bytes_encoding}')

        # Keep the 'minimum encoded string size' logic in the 'auto' mode for compatibility
        force_hex = bytes_encoding == 'hex'
        if bytes_encoding == 'auto' and len(re.split(br'(\(|\\|\))', raw)) // 2 >= len(raw):
            force_hex = True

        if force_hex:
            # Keep the pdfrw logic: "The spec does not mandate uppercase, but it seems to be the convention."
            result = '<' + raw.hex().upper() + '>'
        else:
            # Encode a PDF literal string according to specs (see PDF Ref sec. 3.2.3)
            specialChars = {'\n':'\\n', '\r':'\\r', '\t':'\\t', '\b':'\\b', '\f':'\\f',
                            '(':'\\(', ')':'\\)', '\\':'\\\\'}
            escapedString = [c if c not in specialChars else specialChars[c] for c in raw.decode('latin1')]
            result = '(' + ''.join(escapedString) + ')'

        return cls(result)
        
# class PdfTextString(str):
#     '''
#     The argument is a string that should obey the syntax for PDF string objects.
#     These PDF strings are series of character codes that are encoded using one of the following formats:

#     * a literal '(..)' string  (a string encoded using 1 byte per char);
#     * a hex '<..>' string  (a string encoded using 2 bytes per char).

#     The precise format of the encoding is described in PDF Reference 1.7, Sec. 3.2.3.

#     The function returns the character codes that are encoded in the pdfStringObject
#     as a Python string. The chars in the returned string are not Unicode values of the
#     encoded characters, rather each char in the returned string represents a character code
#     equal to ord(char), and so what is returned is essentially a series of integers.
    
#     In pdfrw, there's a PdfString class which is similar to this one, but not similar enough
#     to make this class unnecessary; below is the summary of the differences:
    
#     PdfString class:

#     * designed to represent PDF strings that appear strictly outside dictionary streams;
#     * these strings can use one of only two encodings; the actual encoding used is determined
#     by the first two bytes of the string itself;
#     * the class provides two methods to decode the string: to_bytes() represents encoded characters
#     by a series of one or two bytes per character, depending on the encoding used; to_unicode()
#     actually decodes the string using its fixed encoding scheme.

#     PdfTextString class:

#     * designed to represent strings that are used in the text typesetting operators inside
#     PDF content streams;
#     * these strings are actually just sequences of character codes; the actual conversion of
#     the character codes to glyphs/Unicode points is determined by the specific font used in
#     typesetting the string;
#     * the PdfTextString class therefore provides only one decode() function, which is closer
#     to the to_bytes() function of the PdfString class, with the difference being that
#     the string is decoded into an instance of Python string (str) class; this has the benefit
#     of having a one-to-one correspondence between the chars in the decoded string and character
#     codes irrespective of the actual encoding used (1 or 2-byte); in particular, the length of
#     the returned (decoded) string exactly equals the number of encoded characters.
    
#     See Pdf Reference 1.7, Sec. 3.8.1 for details on the encodings of text strings.        
#     '''

#     format: str
#     codes: str

#     def __new__(cls, codeString:str = None, pdfString:str = None, toFormat:str = 'auto', forceCID:bool = False):
#         if not isinstance(codeString, str) and not isinstance(pdfString, str):
#             raise ValueError(f'either codeString or pdfString should be given and should be an instance of str')
#         if toFormat not in ['hex', 'literal', 'auto']:
#             raise ValueError(f'foFormat argument should be one of: "hex", "literal", "auto"')

#         # Parse a pdfString
#         if codeString == None:
#             s = pdfString
#             format = 'literal' if s.startswith('(') and s.endswith(')') \
#                     else 'hex' if s.startswith('<') and s.endswith('>') \
#                     else None
#             s = s[1:-1]

#             if format == None:
#                 raise ValueError(f'textString argument should be either a <hex> or a (literal) PDF string: {s}')

#             if format == 'hex':
#                 try:
#                     codeString = bytes.fromhex(s).decode('utf-16-be' if forceCID else 'latin1')
#                 except:
#                     raise ValueError(f'invalid PDF hex string: {pdfString}')

#             if format == 'literal':
#                 # Normalize a PDF literal string according to specs (see PDF Ref sec. 3.2.3)
#                 s = re.sub(r'\r\n','\n',s) # any end-of-line marker not preceded by backslash is equivalent to \n
#                 s = re.sub(r'\r','\n',s) # any end-of-line marker not preceded by backslash is equivalent to \n
#                 s = re.sub(r'\\\n','',s) # a combination of backslash followed by any end-of-line marker is ignored (=no new line)
#                 # s = re.sub(r'([^nrtbf()\\0-7])\\([^nrtbf()\\0-7])','\1\2', self[1:-1]) # if backslash is not followed by a special char ignore it
#                 s = re.sub(r'([^\\])[\\]([^nrtbf()\\0-7])','\1\2', s) # if \ is not foll-d by a special char & not prec-d by \ ignore it
#                 s = s.replace('\\(','(').replace('\\)',')')

#                 # First encode() creates bytes, then decode() interprets escapes contained in bytes
#                 # (latin1 encoding used in the prev step translate the escapes literally)
#                 try:
#                     s = s.encode('latin1').decode('unicode-escape')
#                     if forceCID:
#                         s = s.encode('latin1').decode('utf-16-be')
#                 except:
#                     raise ValueError(f'invalid PDF literal string: {pdfString}')

#                 codeString = s

#         # Format codeString
#         if pdfString == None:

#             suitableFormat = 'hex' if any(ord(c)>255 for c in codeString) else 'literal'

#             format = toFormat if toFormat != "auto" else suitableFormat

#             # Format as a hex string
#             if format == 'hex':
#                 try:
#                     pdfString = '<' + codeString.encode('utf-16-be').hex() + '>'
#                 except:
#                     raise ValueError(f'cannot encode as a PDF hex string: {codeString}')

#             # Format as a literal string
#             if format == 'literal':
#                 if suitableFormat == 'hex':
#                     raise ValueError(f'cannot encode as a PDF literal string: {codeString}')

#                 needEsc = ['(',')','\\'] # Chars that need to be escaped
#                 r = [
#                         '\\' + c if c in needEsc
#                         else c if (32 <= ord(c) < 127)
#                         else f'\\{ord(c):03o}'
#                         for c in codeString
#                     ]

#                 pdfString = '(' + ''.join(r) + ')' 

#         instance = super().__new__(cls, pdfString)
#         instance.format = format
#         instance.codes = codeString
#         return instance


# =========================================================================== class PdfFont

class PdfFont:
    '''
    The PdfFont class provides functions to greately simplify the creation and utilization of the PDF fonts.

    Here's a 'Hello, World' example:

    ```
    from pdfrw import PdfWriter, PdfName, PdfDict, IndirectPdfDict
    from pdfrwx.pdffont import PdfFont, PdfFontUtils

    pdf = PdfWriter('hello.pdf')
    page = PdfDict(Type=PdfName.Page, MediaBox=[0,0,100,100], Contents=IndirectPdfDict())
    pdf.addPage(page)

    font = PdfFont(PdfFontUtils.make_core14_font('Helvetica'))
    font.install(page, 'F1')
    encodedString = font.encode('Hello, World')
    page.Contents.stream = f'/F1 12 Tf {encodedString} Tj'
    ```
    '''

    __fontNameCache = {} # a map from id(fontDict) to its consecutive number; needed to number unnamed Type3 fonts

    def __init__(self, fontDict:PdfDict, glyphMap:PdfFontGlyphMap = PdfFontGlyphMap()):
        '''
        Creates an instance of the PdfFont class from an existing fontDict.
        The argument fontDict is stored in self.font.
        '''
        if fontDict == None: err('can\'t make PdfFont(None)')

        self.font = fontDict
        self.glyphMap = glyphMap

        # set name
        self.name = self.get_font_name()

        # Set encoding
        self.encoding = PdfFontEncoding(font = self.font) if not self.is_cid() else None

        # Set widths
        self.cc2width, self.widthMissing, self.widthAverage = self.make_cc2width()
        self.scaleFactor = 1

        # Set fontMatrix
        self.fontMatrix = [float(x) for x in self.font.FontMatrix] if self.font.FontMatrix != None \
                            else [0.001, 0, 0, 0.001, 0, 0]
        
        # Set font's bounding box
        if self.font.Subtype == '/Type3':
            self.bbox = self.get_type3_bbox()
        else:
            try: self.bbox = [float(x) for x in self.font.FontDescriptor.FontBBox]
            except: self.bbox = [0, 0, 1000, 1000]

        # Set cmap
        if self.font.ToUnicode != None:
            try: self.cmap = PdfFontCMap(toUnicodeDict = self.font.ToUnicode) # /ToUnicode may be junk
            except: self.cmap = None
        else: self.cmap = None
        
        if self.cmap == None:
            if self.is_cid():
                # !!! REWRITE: add handling of CID encodings (beyond Identity-H) !!!
                self.cmap = PdfFontCMap(identity=True)
            else:
                # infer cmap based on the mappings from glyph names to Unicode points
                self.cmap = self.glyphMap.make_cmap(self.encoding)

        # spaceWidth is only needed by the class PdfState to regenerate spaces when extracting text
        self.spaceWidth = self.cc2width.get(self.cmap.unicode2cc.get(' ', None),0)
        if self.spaceWidth == 0:
            self.spaceWidth = self.widthAverage * self.scaleFactor

    def get_font_name(self):
        '''
        Returns N + 1, where N is the number of times this function was called previously
        with arguments, whose .Name was identical to the f.Name of the current argument.
        '''
        nDuplicates = lambda d, k: d.get(k) or d.update({k:len(d)+1}) or len(d) # duplicates counter
        f = self.font

        result = f.Name if f.Subtype == '/Type3' \
            else f.DescendantFonts[0].BaseFont if f.Subtype == '/Type0' and f.DescendantFonts != None \
            else f.BaseFont

        if result not in self.__fontNameCache: self.__fontNameCache[result] = {}
        idx = nDuplicates(self.__fontNameCache[result], id(f))

        return result + (f'-v{idx}' if idx > 1 and not PdfFontCore14.standard_fontname(result) else '') \
                if result != None else ('/T3Font' if f.Subtype == '/Type3' else '/NoName') + f'{idx}'

    def install(self, pdfPage, fontName, overwrite=False):
        '''
        Adds self.font to the pdfPage.Resources.Font dictionary under the name fontName.
        The font can then be referred to using Tf operators in the pdfPage.stream
        '''
        if pdfPage.Resources == None: pdfPage.Resources = PdfDict()
        if pdfPage.Resources.Font == None: pdfPage.Resources.Font = PdfDict()
        if PdfName(fontName) in pdfPage.Resources.Font:
            warn(f'font {fontName} already in page Resources')
            if not overwrite: return
        pdfPage.Resources.Font[PdfName(fontName)] = self.font

    def is_simple(self):
        '''
        Checks if the font is a simple font (i.e. if font.Subtype is one of
        '/Type1', '/Type3', '/TrueType', '/MMType1')
        '''
        return self.font.Subtype in ['/Type1', '/Type3', '/TrueType', '/MMType1']

    def is_cid(self):
        '''
        Checks if the font is a CID font (i.e. if font.Subtype is '/Type0')
        '''
        return self.font.Subtype == '/Type0' and not isinstance(self.font.Encoding, PdfDict)

    def is_embedded(self):
        '''
        Checks if the font is embedded
        '''
        return self.font.FontDescriptor != None or self.font.Subtype in ['/Type0','/Type3']

    def decodeCodeString(self, codeString:str):
        '''
        Decode a code string
        '''
        return self.cmap.decode(codeString)

    def decodePdfTextString(self, pdfTextString:str):
        '''
        Decode a PDF text string
        '''
        codes = PdfTextString(pdfTextString, forceCID=self.is_cid()).codes
        return self.cmap.decode(codes)

    def encodeCodeString(self, s:str):
        '''
        Encodes a text string as a code string
        '''
        return self.cmap.encode(s)

    def encodePdfTextString(self, s:str):
        '''
        Encodes a text string as a PdfTextString; this actually produces a PDF hex/literal string
        depending on the value of self.is_cid().
        '''
        return PdfTextString.from_codes(self.cmap.encode(s), forceCID=self.is_cid())

    def width(self, s:str, isEncoded = False):
        '''
        Returns the width of string if typeset with self.font. 
        '''
        if not isEncoded: s = self.cmap.encode(s)
        if s == None: return None
        return sum(self.cc2width.get(cc, self.widthMissing) for cc in s) * self.scaleFactor

    def make_cc2width(self):
        '''
        Returns a tuple (cc2width, missingWidth, averageWidth) where cc2width is a map of character codes (cc, as char) to widths.
        NB: the widths are in document units, not the font units.
        '''
        cc2width = {}
        font = self.font
        if font == None: return None

        if font.Subtype in ['/Type1', '/MMType1', '/Type3', '/TrueType']:

            # Set cc2width
            if font.Widths == None:
                name2width = PdfFontCore14.make_name2width(self.name) \
                    if font.Subtype in ['/Type1', '/TrueType'] else None
                if name2width == None: err(f'failed to make widths: {font}')
                cc2width = {cc:name2width[name] for cc,name in self.encoding.cc2glyphname.items()
                                if name in name2width != None}
            else:
                if None in [font.FirstChar, font.LastChar]: err(f'broken font: {font}')
                # For Type 3 fonts, widths should be scaled by the FontMatrix; PDF Ref Sec. 5.5.4 Type 3 fonts
                # scale = abs(float(font.FontMatrix[0])) \
                #     if font.Subtype == '/Type3' and font.FontMatrix != None else 1.0
                first, last = int(font.FirstChar), int(font.LastChar)
                # if (L:=len(font.Widths)) < last-first+1:
                #     warn(f'font.Widths too short ({L}) for [font.FirstChar,font.LastChar] = {[first,last]}')
                cc2width = {chr(cc):float(font.Widths[cc - first])
                                for cc in range(first, min(last+1, first+len(font.Widths)))}

            # Set defaultWidth
            missingWidth = 0 if font.Subtype == '/Type3' \
                else float(font.FontDescriptor.MissingWidth) \
                    if font.FontDescriptor != None and font.FontDescriptor.MissingWidth != None \
                else 0

        elif font.Subtype == '/Type0':

            # Set cc2width
            dFont = font.DescendantFonts[0]
            widthsArray = dFont.W if dFont.W != None else PdfArray([])
            assert isinstance(widthsArray, PdfArray)
            cc2width = {}
            start,end,chunk = None,None,None
            for token in widthsArray:
                if isinstance(token,PdfArray):
                    if chunk != None: err(f'failed to read widths: {widthsArray}')
                    chunk = [float(w) for w in token]
                elif start == None: start = int(token)
                elif end == None: end = int(token)
                else: chunk = [float(token)]*(end - start + 1)
                
                if chunk != None:
                    if start == None: err(f'failed to read widths: {widthsArray}')
                    cc2width = cc2width | {chr(i + start):width for i,width in enumerate(chunk)}
                    start, end, chunk = None, None, None

            # Set defaultWidth
            missingWidth = float(dFont.DW) if dFont.DW != None else 0

        else:
            raise ValueError(f'unrecognized font type: {font}')

        # Rescale from font units to to document units
        z = abs(float(self.font.FontMatrix[0])) if self.font.FontMatrix != None else 0.001
        cc2width = {cc: w*z for cc,w in cc2width.items()}
        # Get averageWidth
        w = [v for v in cc2width.values() if v != 0]
        averageWidth = sum(w)/ len(w) if len(w) > 0 else 1
        # Rescale missingWidth
        missingWidth *= z

        return cc2width, missingWidth, averageWidth

    def __str__(self):
        '''
        A string representation of a font and all of its font.DescendantFonts (if present)
        '''
        v = self.font
        if v == None: return 'None'
        s = f"{v.Subtype} {self.name} {self.encoding}"
        if v.ToUnicode != None: s += ' /ToUnicode'
        # if v.DescendantFonts != None:
        #     s += ' (' + ', '.join(f'{PdfFont(w)}' for w in v.DescendantFonts) + ')'
        return s

    def fontTableToPdfPages(self, t3scale = 'auto'):
        '''
        Returns a list of PDF pages which show the font tables: all the glyphs with glyph names and ToUnicode values.
        The scale argument can be either 'auto' or a float number. Any value of the scale only
        affects Type3 fonts.
        '''

        courier = PdfFont(PdfFontCore14.make_core14_font_dict('/Courier'))
        arial = PdfFont(PdfFontCore14.make_core14_font_dict('/Arial'))

        # Font size & scale
        fs = 14 
        scale = 1 if self.font.Subtype != '/Type3' else t3scale if t3scale != 'auto' \
                    else 1/(abs(self.fontMatrix[0]) * abs(self.bbox[2] - self.bbox[0]))
        z = fs*scale
        
        # Inversion flag
        invert = self.fontMatrix[3] < 0

        streams = {}

        # Print glyphs
        for cc in self.cc2width:

            # The math
            cid = ord(cc)
            col,row,n = cid % 16, (cid // 16 ) % 16, (cid // 256)
            if n not in streams: streams[n] = ''
            x,y = 10*(2*col), -2*10*row

            # Paint width
            width = self.cc2width[cc]
            streams[n] += f'0.9 g {x} {y} {z*width} {fs} re f\n'

            # Print glyph's unicode value
            if self.font.ToUnicode != None and cc in self.cmap.cc2unicode:
                uni = self.cmap.cc2unicode[cc]
                uniHex = f'{ord(uni):04X}'.encode('latin').hex()
                streams[n] += f'1 0 0 1 {10 * 2 * col} {-2 * 10 * row + 11} Tm /A 3 Tf 0.5 g <{uniHex}> Tj\n'

            # Print glyph
            Tm = ([1,0,0,-1] if invert else [1,0,0,1]) + [x, y]
            TmString = ' '.join(f'{x}' for x in Tm)
            cidHex = f'{cid:04X}' if self.is_cid() else f'{cid:02X}'
            streams[n] += f'{TmString} Tm /F {z:f} Tf 0 g <{cidHex}> Tj\n'

            # Print glyph name
            if self.encoding != None and cc in self.encoding.cc2glyphname:
                gnameHex = self.encoding.cc2glyphname[cc][1:].encode('latin').hex()
                streams[n] += f'1 0 0 1 {10 * 2 * col} {-2 * 10 * row - 4} Tm /A 3 Tf 0.5 g <{gnameHex}> Tj\n'

        pages = []
        for n in streams:

            # Set-up the page
            page = PdfDict(
                Type = PdfName.Page,
                MediaBox = [20,10,360, 360],
                Contents = IndirectPdfDict(),
                Resources = PdfDict(Font = PdfDict(C = courier.font, A = arial.font, F = self.font))
            )

            shift = '' if n == 0 else f' +{n:X}00'
            title = f'{self.font.Subtype} {self.name}{shift}'
            stream  = f'1 0 0 1 40 320 cm BT 1 0 0 1 -10 30 Tm /C 10 Tf ({title}) Tj\n'
            stream += '/C 6 Tf\n'
            for col in range(16): stream += f'1 0 0 1 {10*(2*col) + 5} 20 Tm ({col:X}) Tj\n'
            for row in range(16): stream += f'1 0 0 1 -10 {-10*(2*row) + 2} Tm ({row:X}) Tj\n'

            stream += streams[n]

            stream += 'ET\n'
            page.Contents.stream = stream
            pages.append(page)
 
        return pages

    def get_type3_bbox(self):
        '''
        '''
        from pdfrwx.pdffilter import PdfFilter
        from pdfrwx.pdfstreamparser import PdfStream
 
        get_bbox = lambda cmd: [0, 0, float(cmd[1][0]), float(cmd[1][0])] if cmd[0] == 'd0' \
                                    else [float(x) for x in cmd[1][2:]] if cmd[0] == 'd1' else None
        max_bbox = lambda b1,b2: [min(b1[0],b2[0]), min(b1[1],b2[1]), max(b1[2],b2[2]), max(b1[3],b2[3])]
        valid_bbox = lambda b: b[2] > b[0] and b[3] > b[1]

 
        font = self.font
        if font.Subtype != '/Type3': return [0,0,1,1]
        bbox = [1000000,1000000,-1000000,-1000000]

        for gname,proc in font.CharProcs.items():

            stream = PdfFilter.uncompress(proc).stream
            tree = PdfStream.stream_to_tree(stream)

            proc_bbox = get_bbox(tree[0])
            if proc_bbox == None: err(f'Font: {font}\nGlyph ({gname}) process stream has no d0/d1 operator:\n{proc.stream}')
            bbox = max_bbox(bbox, proc_bbox)

        return PdfArray(bbox) if valid_bbox(bbox) else PdfArray([0,0,1,1])


# =========================================================================== class PdfFontUtils

class PdfFontUtils:

    def __init__(self):
        self.fontsCache = {}

    def findFile(namePattern, dirList:list, findAll = False):
        '''
        For all files inside each rootDir in the rooDirList, returns the full path to the first file
        whose name matches namePattern, or None if no match is found.
        If findAll is True, returns a list of full paths to all matching files or [] if no match is found.
        '''
        import fnmatch,os
        r=[]
        for rootDir in dirList:
            for path, dirlist, filelist in os.walk(rootDir):
                for name in fnmatch.filter(filelist,namePattern):
                    fullPath = os.path.join(path,name)
                    if findAll: r.append(fullPath)
                    else: return fullPath
        return r if findAll else None

    def loadFont(self, fontNames:list[str], dirList:list[str], forceCID = False):
        '''
        Searches for font names in fontNames inside dirs in dirList and tries to load
        the first name match as a CID font. If forceCID == False, before doing any of the above,
        the font name is checked to see if it's a Core14 font name/alias and, if so,
        the corresponding Type1 font is loaded. The returned value is an instance of PdfFont or
        None if the requested font cannot be loaded.

        Note that this function is not static: we need an instance of PdfFontUtils class to
        cache successive calls to loadFont() with the same fontName-s.
        '''
        fontDict = None
        for name in fontNames:
            if name in self.fontsCache:
                return self.fontsCache[name]
            if not forceCID:
                fontDict = PdfFontCore14.make_core14_font_dict(PdfName(name))
                if fontDict != None: nameFound = name ; break
            path = PdfFontUtils.findFile(name+'.*', dirList)
            if path != None:
                fontDict,scaleFactor = PdfFontUtils.make_cid_font_dict(path)
                if fontDict != None: nameFound = name ; break

        if fontDict == None: warn(f'failed to load any font: {fontNames}') ; return None
        font = PdfFont(fontDict)
        self.fontsCache[nameFound] = font
        return font

    # --------------------------------------------------------------------- make_t1_font_dict()

    def make_t1FontDict_from_PFB(pdfFontFilePath:str):
        '''
        '''
        from fontTools.t1Lib import T1Font, findEncryptedChunks

        from fontTools.pens.basePen import NullPen
        pen = NullPen()

        font = T1Font(pdfFontFilePath)
        # print('Parse()')
        font.parse()
        
        fontName = font.font["FontName"]
        matrix = font.font["FontMatrix"]
        bbox = font.font["FontBBox"]

        info = font.font["FontInfo"]
        italicAngle = info.get('ItalicAngle',0)
        isFixedPitch = info.get('isFixedPitch',0)
        weight = info.get('Weight','Medium')
        weight_to_stemv = {'Light':68, 'Medium':100, 'Regular':100, 'Semibold':120, 'Bold':140}
        stemv = weight_to_stemv[weight]


        chars = font.font["CharStrings"]
        for char in chars.values(): char.draw(pen) # This initializes char.width-s

        widths = [chars[name].width for name in font.font["Encoding"]]
        encoding = [0] + [PdfName(name) for name in font.font["Encoding"]]

        # print(f'{encoding}')
        # print(f'{widths}')
        # print(len(encoding), len(widths))

        # This is required by PDF, but one might as well just set flags=32
        # The bits (from the 1st bit up):
        # FixedPitch,Serif,Symbolic,Script,0,Nonsymbolic,Italic,0,0,0,0,0,0,0,0,0,AllCap,SmallCap,ForceBold
        flags = 32 # This is just the pair of symbolic/non-symbolic bits setting (their XOR should always be 1)
        if isFixedPitch: flags += 1
        # if info.get('SerifStyle',0): flags += 2
        if italicAngle: flags += 64

        lengths = []
        fontProgram = b''
        with open(pdfFontFilePath,'rb') as f:
            for n in range(3):
                assert f.read(1) == b'\x80'
                assert f.read(1) in [b'\x01', b'\x02']
                l = int.from_bytes(f.read(4), 'little')
                lengths.append(l)
                fontProgram += f.read(l)
            assert f.read(2) == b'\x80\x03'                

        fontDict = IndirectPdfDict(
            Type = PdfName.Font,
            Subtype = PdfName.Type1,
            BaseFont = PdfName(fontName),
            FirstChar = 0,
            LastChar = 255,
            Widths = PdfArray(widths),
            Encoding = PdfDict(
                Type = PdfName.Encoding,
                # BaseEncoding = PdfName.WinAnsiEncoding,
                Differences = PdfArray(encoding)
            ),

            FontDescriptor = IndirectPdfDict(
                Type = PdfName.FontDescriptor,
                FontName = PdfName(fontName),
                Flags = flags,
                FontBBox = PdfArray(bbox),
                ItalicAngle = italicAngle,
                Ascent = bbox[3],
                Descent = bbox[1],
                # CapHeight = int(z*ttf['OS/2'].sCapHeight),
                CapHeight = bbox[3],
                StemV = stemv,
                FontFile = IndirectPdfDict(
                    Length1 = lengths[0],
                    Length2 = lengths[1],
                    Length3 = lengths[2],
                    stream = py23_diffs.convert_load(fontProgram)
                )
            )
        )

        return fontDict

    def save_t1FontDict_as_PFB(font:PdfDict, filePath:str):
        '''
        Saves Type 1 font dictionary to file (PFB file format).
        '''
        ff = PdfFilter.uncompress(font.FontDescriptor.FontFile)
        data = py23_diffs.convert_store(ff.stream)

        if int(ff.Length3) == 0:
            for _ in range(8):
                data += b'0' * 64 + b'\n'
            data += b"cleartomark\n"

        from fontTools.t1Lib import writePFB
        writePFB(filePath, data)

    # --------------------------------------------------------------------- make_cid_font_dict()

    def make_cid_font_dict(fontFilePath:str):
        '''
        Returns a tuple (fontDict, scaleFactor) where fontDict is the the dictionary that embeds the CID font
        and scaleFactor is 1000/unitsPerEm (the font's unitsPerEm parameter is most often equal to 1000, but not always).

        The created font's encoding is Identity-H which means that the 2-byte character codes
        and CIDs are the same thing, and that you need to use hex strings in text operators.
        Also, CIDs == 2-byte UVs (character's Unicode points), which means that
        a) the ToUnicode CMap is trivial; b) only 2-byte Unicode characters can be used; c) the hex string
        of the text operator is just a UTF-16BE encoded Unicode string.
        '''
        # Useful:
        # the 'ttx' command (part of fonttools) to inspect fonts
        # the code snippets: https://github.com/fonttools/fonttools/tree/main/Snippets
        # general references on fonttools (see comments in the imports section above)

        try: from fontTools.ttLib.ttFont import TTFont
        except: err(f'failed to load fonttools; run: pip3 install fonttools')

        with open(fontFilePath, 'rb') as font_file:
            ttf = TTFont(font_file)

        with open(fontFilePath, 'rb') as font_file:
            byteStream = font_file.read()
            stream = byteStream.decode('Latin-1')

        # Get font family, subfamily (normal/italic/bold/bolditalic) & full name
        # See: https://docs.microsoft.com/en-us/typography/opentype/spec/name
        FONT_FAMILY,FONT_SUBFAMILY,FONT_FULLNAME = 1,2,4
        fontFamily, fontSubfamily, fontName = None, None, None
        for record in ttf['name'].names:
            if record.nameID == FONT_FAMILY: fontFamily = record.toUnicode()
            if record.nameID == FONT_SUBFAMILY: fontSubfamily = record.toUnicode()
            if record.nameID == FONT_FULLNAME: fontName = re.sub(r' *','', record.toUnicode())

        # Number of glyphs (not used right now)
        numGlyphs = ttf['maxp'].numGlyphs

        # scaleFactor
        scaleFactor = 1000/ttf['head'].unitsPerEm

        # This is a hack to get the missing stemv; see:
        # https://stackoverflow.com/questions/35485179/stemv-value-of-the-truetype-font/
        weight = ttf['OS/2'].usWeightClass / 65
        stemv = 50 + int(weight*weight + 0.5)

        # This is required by PDF, but one might as well just set flags=32
        # The bits (from the 1st bit up):
        # FixedPitch,Serif,Symbolic,Script,0,Nonsymbolic,Italic,0,0,0,0,0,0,0,0,0,AllCap,SmallCap,ForceBold
        flags = 32 # This is just the symbolic/nonsymbol bits set (their XOR should always be 1)
        if ttf['post'].isFixedPitch: flags += 1
        if ttf['OS/2'].panose.bSerifStyle: flags += 2
        if ttf['post'].italicAngle: flags += 64

        # Set char widths
        widths = {}
        unicode2glyphName = ttf.getBestCmap()
        if unicode2glyphName == None: raise ValueError(f'font has no Unicode CMap: {fontFilePath}')
        glyphSet = ttf.getGlyphSet()
        for unicode, glyphName in unicode2glyphName.items():
            widths[unicode] = glyphSet[glyphName].width * scaleFactor

        # make CIDToGIDMap
        # By design, we identify Unicode values with CIDs, and so the map is actually from Unicode points to GIDs
        unicode2glyphName = ttf.getBestCmap() # A map from Unicode to glyph names
        maxCID = 65536 # Let's make this as large as possibly addressable for now, it will compress nicely.
        cid2gid = ["\x00"] * maxCID * 2
        for unicode, glyphName in unicode2glyphName.items():
            if unicode >= maxCID: continue # By our design, chars w. unicode > 2^16 will not be rendered
            glyphId = ttf.getGlyphID(glyphName)
            cid2gid[unicode * 2] = chr(glyphId >> 8)
            cid2gid[unicode * 2 + 1] = chr(glyphId & 0xFF)
        CIDToGIDMap =  IndirectPdfDict(stream=''.join(cid2gid))

        z = scaleFactor
        fontDict = IndirectPdfDict(
            Type=PdfName('Font'),
            Subtype=PdfName('Type0'),
            BaseFont=PdfName(fontName),
            Encoding=PdfName('Identity-H'),
            DescendantFonts=PdfArray([
                IndirectPdfDict(
                    Type = PdfName('Font'),
                    Subtype = PdfName('CIDFontType2'),
                    BaseFont = PdfName(fontName),
                    CIDSystemInfo=PdfDict(
                        Registry = PdfString('(Adobe)'),
                        Ordering = PdfString('(UCS)'),
                        Supplement = 0
                    ),
                    FontDescriptor = IndirectPdfDict(
                        Type=PdfName('FontDescriptor'),
                        FontName = PdfName(fontName),
                        Flags = flags,
                        FontBBox = PdfArray([
                            int(z*ttf['head'].xMin),
                            int(z*ttf['head'].yMin),
                            int(z*ttf['head'].xMax),
                            int(z*ttf['head'].yMax)
                        ]),
                        ItalicAngle = ttf['post'].italicAngle,
                        Ascent = int(z*ttf['OS/2'].sTypoAscender),
                        Descent = int(z*ttf['OS/2'].sTypoDescender),
                        # CapHeight = int(z*ttf['OS/2'].sCapHeight),
                        CapHeight = int(z*ttf['OS/2'].sTypoAscender),
                        StemV = stemv,
                        FontFile2 = IndirectPdfDict(
                            stream=stream
                        )
                    ),
                    # DW = int(round(ttf.metrics.defaultWidth, 0)),
                    # Widths = PdfArray(ttf.metrics.widths), # Required for simple TrueType fonts
                    W = PdfArray([x for u,w in widths.items() for x in [u,PdfArray([w])]]),
                    # W = PdfObject('[ ' + ' '.join([f'{u} [{w}]' for u,w in widths.items()]) + ' ]'),
                    CIDToGIDMap = CIDToGIDMap
                )
            ]),
            ToUnicode = IndirectPdfDict(
                stream = PdfFontCMap(identity=True).write_pdf_stream(CMapName='/Adobe-Identity-UCS', isCID=True))
        )

        return fontDict, scaleFactor

# ============================================================================= main()

if __name__ == '__main__':

    helpMessage='''\
pdffont.py -- embed/extract fonts into/from PDF

Usage:
pdffont.py font.pfb -- produces font.pdf which shows the font table
pdffont.py doc.pdf -- extracts all embedded Type1 fonts as pfb files
'''

    import sys
    from os.path import splitext
    from pdfrw import PdfReader, PdfWriter

    from pdfrwx.pdfobjects import PdfObjects
    from pprint import pprint

    if len(sys.argv) == 1:
        sys.exit(helpMessage)

    filePath = sys.argv[1]
    root, ext = splitext(filePath)

    if ext.lower() == '.pfb':
        fontDict = PdfFontUtils.make_t1FontDict_from_PFB(filePath)
        pdf = PdfWriter(root + '.pdf')
        pdf.addPage(PdfFont(fontDict).fontTableToPdfPage())
        pdf.write()
    
    if ext.lower() == '.pdf':
        pdf = PdfReader(filePath)
        fonts = PdfObjects()
        fonts.read_all(pdf, PdfObjects.fontType1Filter)
        for font in fonts.values():
            if font.FontDescriptor == None: continue
            PdfFontUtils.save_t1FontDict_as_PFB(font, root + '-' + font.BaseFont[1:] + '.pfb')


    # page = PdfDict(Type=PdfName.Page, MediaBox=[0,0,200,200], Contents=IndirectPdfDict())
    # # fontDict = PdfFontCore14.make_core14_font('/Times-Roman')
    # fontDict = PdfFontUtils.make_t1FontDict_from_PFB('./cmr10.pfb')
    # # print('fontDict: ',fontDict)
    # font = PdfFont(fontDict, PdfFontGlyphMap())
    # # print(f'font: {font}')
    # font.install(page, 'F1')
    # # print(page)
    # # text='''Hello…'''
    # text='''Hello, World!'''
    # pdfString = font.encodePdfTextString(text)
    # print("encoded string:", [pdfString])
    # page.Contents.stream = f'BT 10 50 Td /F1 10 Tf {pdfString} Tj ET'

    # pdf = PdfWriter('hello.pdf')
    # pdf.addPage(page)

    # pdf.write()

