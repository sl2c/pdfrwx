#!/usr/bin/env python3

# ================================================== imports

# pdfrw
try:
    from pdfrw import PdfDict, IndirectPdfDict, PdfString, PdfArray, PdfName, py23_diffs, PdfWriter
except:
    raise SystemError(f'import pdfrw failed; run: pip3 install pdfrw')

# pdfrwx
from pdfrwx.common import err, warn, msg
from pdfrwx.pdffontencoding import PdfFontEncoding
from pdfrwx.pdffontglyphmap import PdfFontGlyphMap
from pdfrwx.pdffontcore14 import PdfFontCore14
from pdfrwx.pdffontcmap import PdfFontCMap
from pdfrwx.pdffilter import PdfFilter
from pdfrwx.pdfgeometry import VEC, MAT, BOX

# fontLib
try:
    from fontTools.ttLib import TTFont
    from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
    from fontTools.t1Lib import T1Font, writePFB
    from fontTools.cffLib import CFFFontSet
    from fontTools.pens.basePen import NullPen
except:
    raise SystemError(f'import fonttools failed; run: pip3 install fonttools')

# misc
import sys, re, os, tempfile, argparse, random, string
from io import BytesIO
from pprint import pprint

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
        

# =========================================================================== class PdfFont

class PdfFont:
    '''
    '''

    __fontNameCache = {} # a map from id(fontDict) to its consecutive number; needed to number unnamed Type3 fonts

    # -------------------------------------------------------------------------------- __init__()

    def __init__(self,
                 pfb:bytes = None,
                 cff:bytes = None,
                 ttf:bytes = None,
                 otf:bytes = None,
                 font:PdfDict = None,
                 cc2gname:dict = None,
                 cc2unicode:dict = None,
                 force_CID:bool = False,
                 glyphMap:PdfFontGlyphMap = PdfFontGlyphMap(),
                 quiet:bool = False,
                 extractFontProgram:bool = True,
                 makeSyntheticCmap:bool = True):
        '''
        Creates an instance of PdfFont from a PFB(PFA)/CFF/TTF/OTF font file (bytes),
        or from a PDF font dictionary object (PdfDict).
        '''

        self.pfb = pfb
        self.cff = cff
        self.ttf = ttf
        self.otf = otf

        self.font = font

        self.glyphMap = glyphMap

        self.info = {}

        if self.pfb:
 
            self.info = self.read_pfb_info()
            self.encoding = self.make_encoding(cc2gname = cc2gname, cc2unicode = cc2unicode)
            self.font = self.make_font_dict(encoding = self.encoding, force_CID = force_CID)
 
        elif self.cff:

            self.info = self.read_cff_info()
            self.encoding = self.make_encoding(cc2gname = cc2gname, cc2unicode = cc2unicode)
            self.font = self.make_font_dict(encoding = self.encoding, force_CID = force_CID)

        elif self.ttf or self.otf:

            self.info = self.read_ttf_otf_info()
            self.encoding = self.make_encoding(cc2gname = cc2gname, cc2unicode = cc2unicode)
            self.font = self.make_font_dict(encoding = self.encoding, force_CID = force_CID)
 
        elif self.font:

            if extractFontProgram:
                self.extract_font_program()
                self.info = self.read_pfb_info() if self.pfb \
                                else self.read_ttf_otf_info() if (self.ttf or self.otf) \
                                else self.read_cff_info() if self.cff \
                                else self.read_core14()

            self.encoding = self.get_cid_encoding_from_type0_font() if self.is_cid() \
                                else PdfFontEncoding(font = self.font, gid2gname = self.info.get('gid2gname'))

        else:

            raise ValueError(f'at least one of the arguments must be specified: pfb/cff/ttf/otf/font')

        # Set font name
        self.name = self.get_font_name()

        # Set chars widths
        self.widthMissing = self.get_missingWidth()
        self.cc2width = self.get_cc2width_from_font()

        # Set fontMatrix
        self.fontMatrix = MAT(self.font.FontMatrix) if self.font.FontMatrix != None \
                            else MAT([0.001, 0, 0, 0.001, 0, 0])
        
        # Set font's bounding box
        if self.font.Subtype == '/Type3':
            self.bbox = self.get_type3_bbox()
        else:
            try: self.bbox = BOX(self.get_font_descriptor().FontBBox)
            except: self.bbox = BOX([0, 0, 1000, 1000])
            if self.bbox[2] == self.bbox[0] or self.bbox[3] == self.bbox[1]:
                warn(f'bad FontDescriptor.FontBBox {self.bbox} for font {self}; setting to [0,0,1000,1000]')
                self.bbox = BOX([0,0,1000,1000])

        # Set cmap_toUnicode (read from the font's ToUnicode dict)
        self.cmap_toUnicode = None
        if self.font.ToUnicode:
            try: self.cmap_toUnicode = PdfFontCMap(toUnicodeDict = self.font.ToUnicode) # /ToUnicode may be junk
            except: pass
            if self.cmap_toUnicode == None: warn(f'failed to read ToUnicode CMap for font: {self.name}')

        # An internal Unicode map (read from the font program)
        self.cmap_internal = None
        cid2unicode = self.get_cid2unicode(self.encoding)
        if cid2unicode and len(cid2unicode) > 0:
            self.cmap_internal = PdfFontCMap()
            self.cmap_internal.cc2unicode = cid2unicode
            self.cmap_internal.reset_unicode2cc()

        # Set cmap_synthetic: infer Unicode points based on glyph names using glyphMap
        self.cmap_synthetic = None
        if makeSyntheticCmap:
            self.cmap_synthetic = self.glyphMap.make_cmap(encoding = self.encoding, quiet = quiet)
            if self.cmap_synthetic and len(self.cmap_synthetic.cc2unicode) == 0:
                self.cmap_synthetic = None

        # Set the aggregated version of cmap which combines all 3 versions above
        self.cmap = PdfFontCMap()
        self.cmap.cc2unicode = (self.cmap_synthetic.cc2unicode if self.cmap_synthetic else {}) \
                                | (self.cmap_internal.cc2unicode if self.cmap_internal else {}) \
                                | (self.cmap_toUnicode.cc2unicode if self.cmap_toUnicode else {})
        self.cmap.reset_unicode2cc()


        # spaceWidth is only needed by the class PdfState to regenerate spaces when extracting text
        self.spaceWidth = self.cc2width.get(self.cmap.unicode2cc.get(' ', None),0)
        if self.spaceWidth == 0:
            w = [v for v in self.cc2width.values() if v != 0]
            averageWidth = sum(w)/ len(w) if len(w) > 0 else 1
            self.spaceWidth = averageWidth

        # if re.search('KozMin', self.name):
        #     print('DEBUG')
        #     pprint(self.font)
        #     sys.exit()

    # -------------------------------------------------------------------------------- read_pfb_info()

    def read_pfb_info(self):
        '''
        '''
        with tempfile.TemporaryDirectory() as tmp:
            T = lambda fileName: os.path.join(tmp, fileName)
            open(T('tmp.pfb'),'wb').write(self.pfb)
            t1 = T1Font(T('tmp.pfb'))

        try:
            t1.parse()
        except:
            warn(f'failed to parse a Type1 font')
            return {}

        info = {}

        info['FontName'] = t1.font['FontName']
        if 'FontInfo' in t1.font:
            info['FamilyName'] = t1.font['FontInfo'].get('FamilyName')

        # Geometry
        info['FontMatrix'] = t1.font["FontMatrix"]
        bbox = t1.font["FontBBox"]
        info['FontBBox'] = bbox
        info['Ascent']      = bbox[3]
        info['Descent']     = bbox[1]
        info['CapHeight']   = bbox[3]

        # Style
        if 'FontInfo' in t1.font:
            info['isFixedPitch'] = t1.font["FontInfo"].get('isFixedPitch',0)
            info['isSerif'] = t1.font["FontInfo"].get('SerifStyle',0)
            info['ItalicAngle']= t1.font["FontInfo"].get('ItalicAngle',0)

            # This is a rough sketch of what StemV values should be for various Weight classes
            weight = t1.font["FontInfo"].get('Weight','Medium')
            weight_to_stemv = {'Blond':50, 'Light':68,
                                'Medium':100, 'NORMAL':100, 'Normal':100, 'Regular':100,
                                'Italic':100, 'Roman':100, 'Book':100,
                                'Semibold':120, 'Bold':140, 'Bold Italic':140}
            info['StemV'] = weight_to_stemv[weight]

        # This initializes char.width-s
        chars = t1.font["CharStrings"]
        for char in chars.values(): char.draw(NullPen())

        # Set the maps
        # info['gid2gname'] = {chr(i):gname for i,gname in enumerate(chars.keys())}
        info['gid2gname'] = {chr(i):gname for i,gname in enumerate(t1.font['Encoding']) if gname != '.notdef'}
        info['gname2width'] = {gname:chars[gname].width for gname in chars.keys()}

        # Return info
        return info

    # -------------------------------------------------------------------------------- fix_ttf_font()

    def fix_ttf_font(ttf:bytes):
        '''
        '''

        ttFont = TTFont(BytesIO(ttf))
        tableIDs = [(table.platformID, table.platEncID) for table in ttFont['cmap'].tables]

        if (3,1) not in tableIDs:

            msg(f'Adding an empty (3,1) cmap sub-table to the TTF font')

            table = CmapSubtable.newSubtable(4)
            table.platformID = 3  # Microsoft platform
            table.platEncID = 1   # Unicode encoding
            table.language = 0    # Default language
            table.cmap = {}
            ttFont['cmap'].tables.append(table)

            bs = BytesIO() ; ttFont.save(bs) ; bs.seek(0)
            ttf = bs.read()
        
        return ttf

    # -------------------------------------------------------------------------------- read_ttf_otf_info()

    def read_ttf_otf_info(self):
        '''
        '''
        assert bool(self.ttf) ^ bool(self.otf)

        ttFont = TTFont(BytesIO(self.ttf or self.otf))

        # Add an empty (3,1) i.e. (Microsoft, Unicode) encoding table to the font's `cmap` if it is missing.
        # Without this table using a TrueType font is a pain.
        # See "Encodings for TrueType Fonts" in PDF Ref. sec. 5.5.5 for more details.

        info = {}

        # Get font family, subfamily (normal/italic/bold/bolditalic) & full name
        # See: https://docs.microsoft.com/en-us/typography/opentype/spec/name
        try:

            for record in ttFont['name'].names:

                if record.nameID == 1: info['FontFamily'] = record.toUnicode()
                if record.nameID == 2: info['FontSubfamily'] = record.toUnicode()
                if record.nameID == 4: info['FontName'] = record.toUnicode()

                if record.nameID == 6 and 'FontName' not in info: info['FontName'] = record.toUnicode()
                if record.nameID == 16 and 'FontFamily' not in info: info['FontFamily'] = record.toUnicode()
                if record.nameID == 17 and 'FontSubfamily' not in info: info['FontSubfamily'] = record.toUnicode()

            if 'FontName' not in info and 'FontFamily' in info and 'FontSubfamily' in info:
                info['FontName'] = info['FontFamily'] + '-' + info['FontSubfamily']

            # Remove all spaces from FontName
            info['FontName'] = re.sub(r' *','', info['FontName'])

        except:
            warn(f'missing or corrupt \'name\' table in font')


        info['numGlyphs'] = ttFont['maxp'].numGlyphs

        # Geometric parameters
        info['unitsPerEm'] = ttFont['head'].unitsPerEm
        z = float(1000/info['unitsPerEm'])
        minMax = [ttFont['head'].xMin, ttFont['head'].yMin, ttFont['head'].xMax, ttFont['head'].yMax]
        info['FontBBox'] = [int(z*x) for x in minMax]

        if os2 := ttFont.get('OS/2'):
            info['Ascent']      = int(z*os2.sTypoAscender)
            info['Descent']     = int(z*os2.sTypoDescender)
            # info['CapHeight']   = int(z*os2.sCapHeight)
            info['CapHeight']   = int(z*os2.sTypoAscender)

            try: info['XHeight']     = int(z*os2.sxHeight)
            except: pass

            # This is a hack to get the missing StemV; see:
            # https://stackoverflow.com/questions/35485179/stemv-value-of-the-truetype-font/
            weight = os2.usWeightClass / 65
            info['StemV'] = 50 + int(weight*weight + 0.5)

        # Stylistic parameters
        if post := ttFont.get('post'):
            info['ItalicAngle']     = post.italicAngle
            info['isFixedPitch']    = post.isFixedPitch

        if os2 := ttFont.get('OS/2'):
            info['isSerif']         = os2.panose.bFamilyType == 2 and os2.panose.bSerifStyle
            info['isScript']        = os2.panose.bFamilyType == 3

        # info['Widths'] = PdfArray(ttf.metrics.widths)
        # info['DefaultWidth'] = int(round(ttf.metrics.defaultWidth, 0))

        # Set maps
        try: glyphSet = ttFont.getGlyphSet()
        except:
            glyphSet = None
            warn(f'failed to get glyphSet from font: {info.get("FontName")}')
        if glyphSet:
            info['gid2gname'] = {chr(ttFont.getGlyphID(gname)):gname for gname in glyphSet}
            info['gname2width'] = {gname:glyphSet[gname].width * z for gname in glyphSet}
        
        if cmap := ttFont.get('cmap'):
            info['isSymbolic'] = any(table.platformID == 3 and table.platEncID == 0 for table in cmap.tables)
            info['unicode2gname'] = ttFont.getBestCmap()

        # Return the result
        return info

    # -------------------------------------------------------------------------------- read_cff_info()

    def read_cff_info(self):
        '''
        '''
        assert self.cff != None

        info = {}

        # Parse the CFF data
        cff = CFFFontSet()
        cff.decompile(file = BytesIO(self.cff), otFont = TTFont())

        # Access the first font in the CFF font set
        assert len(cff.fontNames) == 1
        info['FontName'] = cff.fontNames[0]
        font = cff[0]

        # Presence of ROS means a CID-encoded CFF font
        info['ROS'] = font.ROS if hasattr(font, 'ROS') else None

        # This initializes char.width-s
        try:
            chars = font.CharStrings    
            for char in chars.values(): char.draw(NullPen())
        except:
            warn(f'CFF font has no CharStrings: {info["FontName"]}')
            chars = {}

        # gid2gname
        if info['ROS']:
            info['gid2gname'] = {chr(int(gname[3:]) if gname != '.notdef' else 0):gname for gname in chars.keys()}
        else:
            try:
                info['gid2gname'] = {chr(i):gname for i,gname in enumerate(font.Encoding) if gname != '.notdef'}
            except:
                warn(f'CFF font has no CharStrings: {info["FontName"]}')
                info['gid2gname'] = {}
                
        # gname2width
        info['gname2width'] = {gname:chars[gname].width for gname in chars.keys()}

        # Set font params
        info['unitsPerEm'] = 1000
        info['FontMatrix'] = font.FontMatrix

        try:
            font.recalcFontBBox()
            info['FontBBox'] = font.FontBBox
        except:
            warn(f'failed to get FontBBox; setting to [0,0,1000,1000]: {info["FontName"]}')
            info['FontBBox'] = [0,0,1000,1000]

        info['ItalicAngle'] = font.ItalicAngle
        info['isFixedPitch'] = font.isFixedPitch

        info['Ascent'] = info['FontBBox'][3]
        info['Descent'] = info['FontBBox'][1]
        info['CapHeight'] = info['FontBBox'][3]

        try: info['StemV'] = font.Private.rawDict.get('StdVW', 100)
        except: info['StemV'] = 100

        # ................................................................................ DEBUG 

        # print('DEBUG:', 'chars.keys()', chars.keys())
        # print('-'*70)
        # print('DEBUG:', 'gid2gname', info['gid2gname'])

        # # Dump XML
        # from fontTools.misc.xmlWriter import XMLWriter
        # font.toXML(XMLWriter(f'__{info["FontName"]}.xml'))
        # open(f'__{info["FontName"]}.cff','wb').write(self.cff)

        # info['FontBBox'] = font.rawDict.get('FontBBox', None)
        # info['FontBBox'] = font.rawDict.get('FontBBox', None)
        # info['FontBBox'] = font.rawDict.get('FontBBox', None)

        # print('-'*70)
        # print(info)

        # pprint(info)
        # pprint(font.rawDict)
        # print(dir(font))
        # print(font.FDArray.items[0].Private.rawDict)
        # print(font.getGlyphOrder())
        # if re.search('VXUWDK', self.get_font_name()):
        #     print(cff.fontNames)
        #     print('***** dir(font)', dir(font))
        #     print('***** font.rawDict', font.rawDict)
        #     from fontTools.misc.xmlWriter import XMLWriter
        #     print('***** toXML()', font.toXML(XMLWriter('__debug.xml')))
        #     print('***** getGlyphOrder()', font.getGlyphOrder())
        #     print('***** topDictIndex', dir(cff.topDictIndex))
        #     print('***** strings', dir(cff.strings))
        #     open('__sample.cff','wb').write(self.cff)
        #     sys.exit()

        # Return info
        return info
    
    # -------------------------------------------------------------------------------- read_core14()

    def read_core14(self):
        '''
        '''
        assert not self.is_embedded()

        info = {}

        fontName = self.get_font_name()
        info['FontName'] = fontName

        # gid2gname
        baseEncodingName = PdfFontCore14.built_in_encoding(fontName)
        if baseEncodingName:
            baseEncoding = PdfFontEncoding(name = baseEncodingName)
            info['gid2gname'] = {cc:gname[1:] for cc,gname in baseEncoding.cc2glyphname.items()}

        # gname2width
        name2width = PdfFontCore14.make_name2width(fontName)
        if name2width:
            info['gname2width'] = {name[1:]:width for name,width in name2width.items()}

        return info

    # -------------------------------------------------------------------------------- make_encoding()

    def make_encoding(self, cc2gname:dict = None, cc2unicode:dict = None):
        '''
        '''
        assert not (bool(cc2gname) and bool(cc2unicode))


        if cc2unicode:
            unicode2gname = self.info.get('unicode2gname')
            if not unicode2gname:
                msg('failed to make encoding: no unicode2gname in font')
                return None
            cc2gname = {cc:unicode2gname[ord(u)] for cc,u in cc2unicode.items() if ord(u) in unicode2gname}
        
        if not cc2gname:
            cc2gname = self.info['gid2gname']

 
        encoding = PdfFontEncoding()
        encoding.cc2glyphname = {cc:PdfName(gname) for cc,gname in cc2gname.items()}
        encoding.reset_glyphname2cc()
        encoding.name = [None, PdfName('Differences')]

        return encoding

    def get_cidset(self):
        '''
        Returns a CIDSet of a CID font as a list of chars, or None if it does not exist
        '''
        assert self.is_cid()
        CIDSet = self.font.DescendantFonts[0].FontDescriptor.CIDSet
        if CIDSet:
            byteStream = py23_diffs.convert_store(PdfFilter.uncompress(CIDSet).stream)
            return [chr(i*8 + j) for i,byte in enumerate(byteStream) for j,bit in enumerate(f'{byte:08b}') if bit == '1']

    # -------------------------------------------------------------------------------- get_cidtogidmap()

    def get_cidtogidmap(self):
        '''
        '''
        CIDToGIDMap = self.font.DescendantFonts[0].CIDToGIDMap
        if CIDToGIDMap and isinstance(CIDToGIDMap, PdfDict):
            b = py23_diffs.convert_store(PdfFilter.uncompress(CIDToGIDMap).stream)
            return {chr(i):chr(b[2*i]*256 + b[2*i+1]) for i in range(len(b) // 2)}
        else:
            return None

    # -------------------------------------------------------------------------------- get_cid_encoding_from_font()

    def get_cid_encoding_from_type0_font(self):
        '''
        Returns an instance of PdfFontEncoding for a self.font that is a Type0 font dictionary.
        Strictly speaking, Type0 fonts do not have an encoding and use CID-to-GID maps.
        This functions uses the internal (written in the font program) gid2gname map to convert
        the CID-to-GID map to a cc2glyphname map that is then made part
        of an instance of the PdfFontEncoding class. This allows then to treat Type0 (CID-keyed)
        and simple fonts uniformly based on (real or surrogate) encoding.
        '''
        assert self.font
        assert self.font.Subtype == '/Type0'

        # Get gid2gname
        gid2gname = self.info.get('gid2gname')
        if not gid2gname: return None

        # Get cid2gid
        cid2gid = self.get_cidtogidmap()
        if not cid2gid:
            cid2gid = {cc:cc for cc in self.get_cc2width_from_font()}

        # Append cids from CIDSet if it exists
        CIDSet = self.get_cidset()
        if CIDSet:
            for cid in CIDSet:
                if cid not in cid2gid: cid2gid[cid] = cid

        # msg(f'{self.get_font_name()}: cc2width --> encoding')

        # Create cid2gname
        cid2gname = {cid:gid2gname.get(gid,'.notdef') for cid,gid in sorted(cid2gid.items())}

        # Create encoding
        encoding = PdfFontEncoding()
        encoding.name = PdfName('CIDEncoding')
        encoding.cc2glyphname = {cc:PdfName(gname) for cc,gname in cid2gname.items()}
        encoding.reset_glyphname2cc()

        return encoding

    # # -------------------------------------------------------------------------------- get_cid2gid_from_font()

    # def get_cid2gid_from_font(self):
    #     '''
    #     '''
    #     assert self.font
    #     font = self.font
    #     if font.Subtype != '/Type0':

    #         gid2gname = self.info.get('gid2gname')
    #         if not gid2gname: return None

    #         cid2gid = {gid:gid for gid in gid2gname}

    #         gname2gid = {gname:gid for gid,gname in gid2gname.items()}
    #         encoding = PdfFontEncoding(font = font)
    #         cid2gid = cid2gid | {cc:gname2gid.get(gname[1:],0) for cc,gname in encoding.cc2glyphname.items()}

    #         # if font.BaseFont and re.search(r'sfrm1095', font.BaseFont.lower()):
    #         #     pprint(gid2gname)
    #         #     # pprint(encoding.cc2glyphname)
    #         #     print(bool(self.pfb), bool(self.ttf))

    #         return cid2gid

    #     else:
    #         CIDToGIDMap = self.font.DescendantFonts[0].CIDToGIDMap
    #         if CIDToGIDMap and isinstance(CIDToGIDMap, PdfDict):
    #             b = py23_diffs.convert_store(PdfFilter.uncompress(CIDToGIDMap).stream)
    #             return {chr(i):chr(b[2*i]*256 + b[2*i+1]) for i in range(len(b) // 2)}
    #         else:
    #             assert CIDToGIDMap in [None, '/Identity']
    #             return {cc:cc for cc in self.get_cc2width_from_font()}


    # # -------------------------------------------------------------------------------- get_cid2gid()

    # def get_cid2gid(self):
    #     '''
    #     '''
    #     gid2gname = self.info.get('gid2gname')
    #     assert gid2gname
    #     return self.info.get('cid2gid') or {chr(gid):chr(gid) for gid in gid2gname.keys()}

    # # -------------------------------------------------------------------------------- make_cid2gname()

    # def get_cid2gname(self):
    #     '''
    #     '''
    #     gid2gname = self.info.get('gid2gname')
    #     if not gid2gname: return None
    #     return {cid:gid2gname.get(gid,'.notdef') for cid,gid in self.get_cid2gid().items()}

    # -------------------------------------------------------------------------------- make_cid2unicode()

    def get_cid2unicode(self, encoding:PdfFontEncoding = None):
        '''
        '''
        unicode2gname = self.info.get('unicode2gname')
        if not unicode2gname: return None
        gname2unicode = {gname:u for u,gname in unicode2gname.items()}

        cid2gname = {cid:gname[1:] for cid,gname in encoding.cc2glyphname.items()}

        return {cid:chr(gname2unicode[gname]) for cid,gname in cid2gname.items() if gname in gname2unicode}

    # -------------------------------------------------------------------------------- get_cid2width()

    def get_cid2width(self, encoding:PdfFontEncoding = None):
        '''
        '''
        cid2gname = {cid:gname[1:] for cid,gname in encoding.cc2glyphname.items()} if encoding \
                        else {gid:gid for gid in self.info['gid2gname'].items()}
        gname2width = self.info['gname2width']
        return {cid:gname2width.get(gname,0) for cid,gname in cid2gname.items()}

    # -------------------------------------------------------------------------------- make_font_dict()

    def make_font_dict(self, encoding:PdfFontEncoding = None, force_CID:bool = False):

        # ................................................................................ get_flags()

        def get_flags(info:dict):
            '''
            Calculates the `Flags` bit field.
            '''
            # This is required by PDF, but one might as well just set flags=32
            # The bits (from the 1st bit up):
            # FixedPitch,Serif,Symbolic,Script,0,Nonsymbolic,Italic,0,0,0,0,0,0,0,0,0,AllCap,SmallCap,ForceBold
            # The XOR of Symbolic & Nonsymbolic bits should always be 1

            flags = 0

            if info.get('isFixedPitch'): flags += 1
            if info.get('isSerif'): flags += 2
            flags += 4 if info.get('isSymbolic') else 32
            if info.get('ItalicAngle') : flags += 64

            return flags
                    
        # ................................................................................ Fix Font Name

        FontName = re.sub(r'\s+', '-', self.info['FontName'])

        if self.ttf or self.otf:
            randomPrefix = ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
            FontName = randomPrefix + '+' + FontName

        # ................................................................................ Determine CID/Type1

        maxCID = ord(max(encoding.cc2glyphname)) if encoding else max(self.info['gid2gname'])
        makeCID = maxCID > 255 or force_CID

        # Consistency checks
        if self.cff:
            cidEncoded = self.info['ROS'] != None
            if cidEncoded and not makeCID: makeCID = True
            elif not cidEncoded and makeCID:
                raise ValueError(f'cannot embed a non-CID-keyed CFF font program in a CID font: {self.info["FontName"]}')
        if self.pfb and makeCID:
            raise ValueError(f'cannot embed a Type1 (PFB) font program in a CID font: {self.info["FontName"]}')

        # PDF Ref.: "[A FontFile3 with the /OpenType Subtype can appear in] a Type1 font dictionary
        # [..] if the embedded font program contains a “CFF” table without CIDFont operators."
        # Since there's no simple way of checking for the presence of CIDFont operators
        # in a CFF font inside an OpenType font we simply impose a requirement:
        # unless there's a 'glyf' table in the OpenType font, it has to be a CID font,
        # and so we simply embed all such fonts in CIDFontType0 fonts.
        otfWithGlyf = bool(self.otf) and 'glyf' in TTFont(BytesIO(self.otf))
        if self.otf and not otfWithGlyf:
            makeCID = True

        # ................................................................................ Font Descriptor

        # Consult PDF Ref. sec. 5.8 Embedded Font Programs

        FontFile = FontFile2 = FontFile3 = None

        if self.pfb: # Make FontFile

            lengths = []
            fontProgram = b''
            with BytesIO(self.pfb) as f:
                for _ in range(3):
                    assert f.read(1) == b'\x80'
                    assert f.read(1) in [b'\x01', b'\x02']
                    l = int.from_bytes(f.read(4), 'little')
                    lengths.append(l)
                    fontProgram += f.read(l)
                assert f.read(2) == b'\x80\x03'

            FontFile = IndirectPdfDict(
                Length1 = lengths[0],
                Length2 = lengths[1],
                Length3 = lengths[2],
                stream = py23_diffs.convert_load(fontProgram)
            )

        elif self.ttf: # Make FontFile2

            fontProgram = PdfFont.fix_ttf_font(self.ttf)
 
            FontFile2 = IndirectPdfDict(
                Length1 = len(fontProgram),
                stream=py23_diffs.convert_load(fontProgram)
            )

        else: # Make FontFile3

            FontFileSubtype = PdfName('Type1C') if self.cff and not makeCID \
                        else PdfName('CIDFontType0C') if self.cff and makeCID \
                        else PdfName('OpenType') if self.otf \
                        else None

            FontFile3 = IndirectPdfDict(
                Subtype = FontFileSubtype,
                stream=py23_diffs.convert_load(self.cff or self.otf)
            )

        # Fix FontBBox if necessary
        bbox = [float(x) for x in self.info.get('FontBBox',[0,0,1000,1000])]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or bbox[1] > self.info['Descent'] or bbox[3] < self.info['Ascent']:
            bbox = [0,self.info['Descent'],1000, max(1000, self.info['Ascent'])]
        
        FontDescriptor = IndirectPdfDict(
            Type = PdfName('FontDescriptor'),
            FontName = PdfName(FontName),
            Flags = get_flags(self.info),
            # FontBBox = PdfArray(self.info['FontBBox']),
            FontBBox = PdfArray(bbox),
            ItalicAngle = self.info['ItalicAngle'],
            Ascent = self.info['Ascent'],
            Descent = self.info['Descent'],
            CapHeight = self.info['CapHeight'],
            # XHeight = self.info.get('XHeight'), --- optional
            StemV = self.info['StemV'],
            FontFile  = FontFile,
            FontFile2 = FontFile2,
            FontFile3 = FontFile3
        )

        # ................................................................................ ToUnicodeCMap

        # ToUnicodeCMap
        ToUnicodeCMap = None
        cc2unicode = self.get_cid2unicode(encoding)
        if cc2unicode:
            ToUnicodeCMap = PdfFontCMap()
            ToUnicodeCMap.cc2unicode = cc2unicode
            ToUnicodeCMap.reset_unicode2cc()

        # ................................................................................ Font Dictionary

        if makeCID:

            # ................................................................................ CID Font

            # CIDToGIDMap
            if not encoding:
                CIDToGIDMap = None
            else:
                gname2gid = {gname:gid for gid,gname in self.info['gid2gname'].items()}
                cid2gid = {cid:gname2gid.get(gname[1:],0) for cid,gname in encoding.cc2glyphname.items()}
                gids = [ord(cid2gid.get(chr(cid),chr(0))) for cid in range(maxCID + 1)]
                CIDToGIDMap = IndirectPdfDict(
                    stream=py23_diffs.convert_load(b''.join(bytes([gid >> 8, gid & 255]) for gid in gids))
                )
            
            # Widths
            W = PdfArray([x for cid,w in self.get_cid2width(encoding).items() for x in [ord(cid),PdfArray([w])]])

            # CIDFontSubtype; see PDF Ref. Sec. 5.8, Table 5.23 Embedded font organization for various font types
            CIDFontSubtype = PdfName('CIDFontType2') if otfWithGlyf or self.ttf else PdfName('CIDFontType0')

            fontDict = IndirectPdfDict(
                Type=PdfName('Font'),
                Subtype=PdfName('Type0'),
                BaseFont=PdfName(FontName),
                Encoding=PdfName('Identity-H'),
                DescendantFonts=PdfArray([
                    IndirectPdfDict(
                        Type = PdfName('Font'),
                        Subtype = CIDFontSubtype,
                        BaseFont = PdfName(FontName),
                        CIDSystemInfo=PdfDict(
                            Registry = PdfString('(Adobe)'),
                            Ordering = PdfString('(UCS)'),
                            Supplement = 0
                        ),
                        FontDescriptor = FontDescriptor,
                        W = W,
                        CIDToGIDMap = CIDToGIDMap
                    )
                ]),
                ToUnicode = IndirectPdfDict(
                    stream = ToUnicodeCMap.write_pdf_stream(PdfName(FontName),isCID = True)
                ) if ToUnicodeCMap else None
            )

        else:

            # ................................................................................ Simple Font

            # OpenType (.otf) fonts are embedded as TrueType fonts if they contain a 'glyf' table;
            # OpenType fonts with a CFF font inside require checking for the absence of
            # CID Font operators if we want to embed them in Type1 fonts
            # (see PDF Ref. Sec. 5.8, Table 5.23 Embedded font organization for various font types)
            # which is hard to do, and so we simply embed all such fonts in CIDFontType0 fonts.
            # This check is made above (see "Determine CID/Type1" section above), and so here
            # (in the simple font section) we are left with only those OpenType fonts that have a 'glyf' table.
            simpleFontSubtype = PdfName('Type1') if self.pfb or self.cff else PdfName('TrueType')

            Differences, FirstChar, LastChar = PdfFontEncoding.cc2glyphname_to_differences(encoding.cc2glyphname)

            Widths = PdfArray([self.get_cid2width(encoding).get(chr(i),0) for i in range(FirstChar, LastChar+1)])

            fontDict = IndirectPdfDict(
                Type = PdfName('Font'),
                Subtype = simpleFontSubtype,
                BaseFont = PdfName(FontName),
                Widths = Widths,
                FirstChar = FirstChar,
                LastChar = LastChar,
                Encoding = PdfDict(
                    Type = PdfName.Encoding,
                    Differences = PdfArray(Differences)
                ),
                FontDescriptor = FontDescriptor,
                ToUnicode = IndirectPdfDict(
                    stream = ToUnicodeCMap.write_pdf_stream(PdfName(FontName),isCID = False)
                ) if ToUnicodeCMap else None
            )

        return fontDict

    # -------------------------------------------------------------------------------- extract_font_program()

    def extract_font_program(self):
        '''
        Extracts the font program from self.font and populates
        one of: self.pfb, self.ttf, self.cff, self.otf
        '''
        assert self.font != None

        fd = self.get_font_descriptor()

        if fd == None: return

        FontFile = fd.FontFile or fd.FontFile2 or fd.FontFile3
        if not FontFile: return None

        FontFile = PdfFilter.uncompress(FontFile)
        FontProgram = py23_diffs.convert_store(FontFile.stream)

        if fd.FontFile:

            # Add the missing zero-filled chunk at the end
            if int(FontFile.Length3) == 0:
                FontProgram += (b'0' * 64 + b'\n')*8 + b'cleartomark\n'
            with tempfile.TemporaryDirectory() as tmp:
                T = lambda fileName: os.path.join(tmp, fileName)
                writePFB(T('tmp.pfb'), FontProgram)
                self.pfb = open(T('tmp.pfb'),'rb').read()

        elif fd.FontFile2:

            self.ttf = FontProgram

        else:

            subtype = FontFile.Subtype
            if subtype in ['/Type1C', '/CIDFontType0C']:
                self.cff = FontProgram
            elif subtype == '/OpenType':
                self.otf = FontProgram
            else:
                raise ValueError(f'invalid FontFile3.Subtype: {subtype}')

    # -------------------------------------------------------------------------------- get_font_name()

    def get_font_name(self):
        '''
        '''
        nDuplicates = lambda d, k: d.get(k) or d.update({k:len(d)+1}) or len(d) # duplicates counter
        f = self.font

        result = f.Name if f.Subtype == '/Type3' \
            else f.DescendantFonts[0].BaseFont if f.Subtype == '/Type0' and f.DescendantFonts != None \
            else f.BaseFont

        if result not in self.__fontNameCache: self.__fontNameCache[result] = {}
        idx = nDuplicates(self.__fontNameCache[result], id(f))

        result = result + (f'-v{idx}' if idx > 1 and not PdfFontCore14.standard_fontname(result) else '') \
                    if result != None else ('/T3Font' if f.Subtype == '/Type3' else '/NoName') + f'{idx}'
        
        return PdfName(result[1:])

    # -------------------------------------------------------------------------------- install()

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

    # -------------------------------------------------------------------------------- is_simple()

    def is_simple(self):
        '''
        Checks if the font is a simple font (i.e. if font.Subtype is one of
        '/Type1', '/Type3', '/TrueType', '/MMType1')
        '''
        return self.font.Subtype in ['/Type1', '/Type3', '/TrueType', '/MMType1']

    # -------------------------------------------------------------------------------- is_cid()

    def is_cid(self):
        '''
        Checks if the font is a CID font (i.e. if font.Subtype is '/Type0')
        '''
        return self.font.Subtype == '/Type0' and not isinstance(self.font.Encoding, PdfDict)

    # -------------------------------------------------------------------------------- get_subtype()

    def get_subtype_string(self):
        '''
        '''
        suffixes = {'/Type1C':'C', '/CIDFontType0C':'C', '/OpenType':'-OpenType'}

        if not self.font: return None

        f = self.font if self.font.Subtype != '/Type0' else self.font.DescendantFonts[0]

        prefix = f.Subtype

        fd = f.FontDescriptor
        if not fd: return prefix
 
        suffix = suffixes.get(fd.FontFile3.Subtype) if fd.FontFile3 else ''
        return prefix + suffix

    # -------------------------------------------------------------------------------- get_encoding_string()

    def get_encoding_string(self):
        '''
        '''
        if not self.font: return None
        if not self.is_cid():
            enc = self.font.Encoding
            if not isinstance(enc, PdfDict): return enc
            return [enc.BaseEncoding, '/Differences']
        else:
            dFont = self.font.DescendantFonts[0]
            s = []
            if dFont.CIDToGIDMap: s.append('/CIDToGIDMap')
            if dFont.FontDescriptor.CIDSet: s.append('/CIDSet')
            s = ' + '.join(s)
            if self.encoding == None: s += ' = None'
            return None if len(s) == 0 else s

    # -------------------------------------------------------------------------------- get_font_descriptor()

    def get_font_descriptor(self):
        '''
        Get the font's descriptor dictionary
        '''
        if not self.font: return None
        return self.font.FontDescriptor if not self.is_cid() else self.font.DescendantFonts[0].FontDescriptor
    
    # -------------------------------------------------------------------------------- is_embedded()

    def is_embedded(self):
        '''
        Checks if the font is embedded
        '''
        fd = self.get_font_descriptor()
        return fd != None and (fd.FontFile or fd.FontFile2 or fd.FontFile3)

    # -------------------------------------------------------------------------------- decodeCodeString()

    def decodeCodeString(self, codeString:str):
        '''
        Decode a code string
        '''
        return self.cmap.decode(codeString)

    # -------------------------------------------------------------------------------- decodePdfTextString()

    def decodePdfTextString(self, pdfTextString:str):
        '''
        Decode a PDF text string
        '''
        codes = PdfTextString(pdfTextString, forceCID=self.is_cid()).codes
        return self.cmap.decode(codes)

    # -------------------------------------------------------------------------------- encodeCodeString()

    def encodeCodeString(self, s:str):
        '''
        Encodes a text string as a code string
        '''
        return self.cmap.encode(s)

    # -------------------------------------------------------------------------------- encodePdfTextString()

    def encodePdfTextString(self, s:str):
        '''
        Encodes a text string as a PdfTextString; this actually produces a PDF hex/literal string
        depending on the value of self.is_cid().
        '''
        return PdfTextString.from_codes(self.cmap.encode(s), forceCID=self.is_cid())

    # -------------------------------------------------------------------------------- width()

    def width(self, s:str, isEncoded = False):
        '''
        Returns the width of string if typeset with self.font. The value isEncoded == False means
        the string s is a unicode string; the value of True means it is an (internal part of) PDF
        encoded string.

        NB: the widths are in document units, not the font units.
        For example, widths of single chars are normally < 1 for Type 1 fonts.
        '''
        if not isEncoded: s = self.cmap.encode(s)
        if s == None: return None
        return sum(self.cc2width.get(cc, self.widthMissing) for cc in s)

    # -------------------------------------------------------------------------------- get_missingWidth()

    def get_missingWidth(self):
        '''
        Returns the MissingWidth (in document units)
        '''
        assert self.font
        font = self.font

        z = abs(float(font.FontMatrix[0])) if font.FontMatrix != None else 0.001
        if font.Subtype == '/Type3': return 0
        elif font.Subtype == '/Type0':
            return z * float(font.DescendantFonts[0].DW or '0')
        else:
            fd = font.FontDescriptor
            return z * float(fd.MissingWidth or '0') if fd else 0


    # -------------------------------------------------------------------------------- get_cc2width_from_font()

    def get_cc2width_from_font(self):
        '''
        Returns a map of character codes (cc, as char) to widths.
        NB: the widths are in document units, not the font units. For example, widths are normally < 1 for Type 1 fonts.
        '''
        assert self.font
        font = self.font

        if font.Subtype != '/Type0': # simple fonts

            cc2width = {}
            if font.Widths != None:
                if None in [font.FirstChar, font.LastChar]:
                    raise ValueError(f'broken font: {font}')
                first, last = int(font.FirstChar), int(font.LastChar)
                cc2width = {chr(cc):float(font.Widths[cc - first])
                                for cc in range(first, min(last+1, first+len(font.Widths)))}
            else:
                # Absent Widths array means it's a Core14 font
                name2width = PdfFontCore14.make_name2width(font.BaseFont)
                if not name2width: raise ValueError(f'failed to get Widths for font:\n{font}')
                encoding = PdfFontEncoding(font = font)
                cc2width = {cc:name2width[gname] for cc,gname in encoding.cc2glyphname.items() if gname in name2width}

            # Add default widths to cc's whose widths are not explicitly specified
            if self.encoding:
                fd = font.FontDescriptor
                missingWidth = float(fd.MissingWidth or '0') if fd else 0
                for cc in self.encoding.cc2glyphname:
                    if cc not in cc2width:
                        cc2width[cc] = missingWidth

        else: # CID fonts

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
                    for i,width in enumerate(chunk):
                        cc2width[chr(i + start)] = width
                    start, end, chunk = None, None, None

            # Add default widths to cids whose widths are not explicitly specified
            defaultWidth = int(dFont.DW) if dFont.DW else 1000
            if cid2gid := self.get_cidtogidmap():
                for cc in cid2gid:
                    if cc not in cc2width: cc2width[cc] = defaultWidth
            if cidset := self.get_cidset():
                for cc in cidset:
                    if cc not in cc2width: cc2width[cc] = defaultWidth

        # Rescale from font units to document units
        # For Type 3 fonts, widths should be scaled by the FontMatrix; PDF Ref Sec. 5.5.4 Type 3 fonts
        z = abs(float(font.FontMatrix[0])) if font.FontMatrix != None else 0.001
        cc2width = {cc: w*z for cc,w in sorted(cc2width.items())}

        return cc2width

    # -------------------------------------------------------------------------------- __str__()

    def __str__(self):
        '''
        A string representation of a font
        '''
        return f'{self.get_subtype_string():14s} {self.name} {self.get_encoding_string()}'

    # -------------------------------------------------------------------------------- fontTableToPdfPages()

    def fontTableToPdfPages(self, t3scale = 'auto'):
        '''
        Returns a list of PDF pages which show the font tables: all the glyphs with glyph names and ToUnicode values.
        The scale argument can be either 'auto' or a float number. Any value of the scale only
        affects Type3 fonts.
        '''

        # Font size & scale
        fs = 14 
        scale = 1 if self.font.Subtype != '/Type3' else t3scale if t3scale != 'auto' \
                    else 1/(abs(self.fontMatrix[0]) * abs(self.bbox[2] - self.bbox[0]))
        z = fs*scale
        
        # Inversion flag
        invert = self.fontMatrix[3] < 0

        streams = {}

        cc2g = {cc:gname[1:] for cc,gname in self.encoding.cc2glyphname.items()} if self.encoding else {}
        cc2w = {cc:w for cc,w in self.cc2width.items()}

        # Print glyphs

        # Paint glyph's box
        for cc,width in cc2w.items():

            # if cc2g.get(cc,'.notdef') == '.notdef': continue

            # The math
            cid = ord(cc)
            col,row,n = cid % 16, (cid // 16 ) % 16, (cid // 256)
            if n not in streams: streams[n] = '0.9 g\n'
            x,y = 10*(2*col), -2*10*row

            streams[n] += f'{x} {y} {z*width} {fs} re f\n'

        # Switch to text mode
        for n in streams: streams[n] += 'BT\n'

        # Print glyph's Unicode value
        counter = {}
        for cc in cc2w:

            # if cc2g.get(cc,'.notdef') == '.notdef': continue

            # The math
            cid = ord(cc)
            col,row,n = cid % 16, (cid // 16 ) % 16, (cid // 256)
            if n not in counter: streams[n] += f'/A 3 Tf 0.5 g\n' ; counter[n] = 1
            x,y = 10*(2*col), -2*10*row

            u1 = self.cmap_toUnicode.cc2unicode.get(cc,'') if self.cmap_toUnicode else ''
            u2 = self.cmap_internal.cc2unicode.get(cc,'')  if self.cmap_internal  else ''
            # u3 = self.cmap_synthetic.cc2unicode.get(cc,'') if self.cmap_synthetic else ''

            red, green, blue, gray = '1 0.25 0.25 rg', '0 0.75 0 rg', '0.25 0.25 1 rg', '0.5 g'
            toHex = lambda s: ''.join(f'{ord(u):04X}' for u in s)

            u1 = toHex(u1)
            u2 = toHex(u2)
            # u3 = toHex(u3)

            s = ''
            if u1 + u2 != '':

                if u1 != '':
                    s = u1
                    if u2 != '':
                        if u2 != u1:
                            s += '|' + u2
                            color = blue
                        else:
                            color = gray
                    else:
                        color = blue
                elif u2 != '':
                    s = u2
                    color = green
                # else:
                #     s = u3
                #     color = red
                    
                streams[n] += f'1 0 0 1 {x} {y + 11} Tm {color} ({s}) Tj 0.5 g\n'


        # Print glyph name
        counter = {}
        for cc in cc2w:

            gname = cc2g.get(cc,'.notdef')
            if gname == '.notdef': continue

            # The math
            cid = ord(cc)
            col,row,n = cid % 16, (cid // 16 ) % 16, (cid // 256)
            if n not in counter: streams[n] += f'/A 3 Tf 0.75 g\n' ; counter[n] = 1
            x,y = 10*(2*col), -2*10*row
            
            if gname != '':
                if len(gname) > 17: gname = gname[:16] + '..'
                gnameHex = gname.encode('latin').hex()
                streams[n] += f'1 0 0 1 {x} {y - 4} Tm 75 Tz <{gnameHex}> Tj 100 Tz\n'

        # Print glyph
        counter = {}
        for cc in cc2w:

            # if cc2g.get(cc,'.notdef') == '.notdef': continue

            # The math
            cid = ord(cc)
            col,row,n = cid % 16, (cid // 16 ) % 16, (cid // 256)
            if n not in counter: streams[n] += f'/F {z:f} Tf 0 g\n' ; counter[n] = 1
            x,y = 10*(2*col), -2*10*row

            Tm = ([1,0,0,-1] if invert else [1,0,0,1]) + [x, y]
            TmString = ' '.join(f'{x}' for x in Tm)
            cidHex = f'{cid:04X}' if self.is_cid() else f'{cid:02X}'
            streams[n] += f'{TmString} Tm <{cidHex}> Tj\n'

        # Switch back from text mode
        for n in streams: streams[n] += 'ET\n'

        def make_page_template(n:int):
            '''
            Makes a page template — a page an empty font table.
            n is the code range page number: n = 0 is 0-255 etc.
            '''

            courier = PdfFont(font = PdfFontCore14.make_core14_font_dict('/Courier'))
            arial = PdfFont(font = PdfFontCore14.make_core14_font_dict('/Arial'))

            # Set-up the page
            page = PdfDict(
                Type = PdfName.Page,
                MediaBox = [20,10,360, 370],
                Contents = IndirectPdfDict(),
                Resources = PdfDict(Font = PdfDict(C = courier.font, A = arial.font, F = self.font))
            )

            shift = '' if n == 0 else f' +{n:X}00'
            title = f'{self.name}{shift}'
            subtype = self.font.Subtype if self.font.Subtype != '/Type0' else self.font.DescendantFonts[0].Subtype
            ToUnicode = ''
            if self.font.ToUnicode != None:
                ToUnicode = ', ToUnicode'
            if self.info.get('unicode2gname'):
                ToUnicode += ', internal Unicode CMap'
            Embedded = '' if self.is_embedded() else ', Not embedded'
            subtitle = f'{self.get_subtype_string()}, Encoding: {self.get_encoding_string()}{ToUnicode}{Embedded}'
            stream  = f'1 0 0 1 40 320 cm BT\n'
            stream += f'1 0 0 1 -10 40 Tm /C 10 Tf ({title}) Tj\n'
            stream += f'1 0 0 1 -10 30 Tm /C 6 Tf ({subtitle}) Tj\n'
            stream += '/C 6 Tf\n'
            for col in range(16): stream += f'1 0 0 1 {10*(2*col) + 5} 20 Tm ({col:X}) Tj\n'
            for row in range(16): stream += f'1 0 0 1 -10 {-10*(2*row) + 2} Tm ({row:X}) Tj\n'
            stream += 'ET\n'

            page.Contents.stream = stream

            return page

        pages = []
        for n in streams:
            page = make_page_template(n)
            page.Contents.stream += streams[n]
            pages.append(page)
        
        if len(pages) == 0:
            pages.append(make_page_template(0))
 
        return pages

    # -------------------------------------------------------------------------------- get_type3_bbox()

    def get_type3_bbox(self):
        '''
        '''
        from pdfrwx.pdffilter import PdfFilter
        from pdfrwx.pdfstreamparser import PdfStream

        get_bbox = lambda cmd: BOX([0, 0, cmd[1][0], cmd[1][0]]) if cmd[0] == 'd0' \
                                    else BOX(cmd[1][2:6]) if cmd[0] == 'd1' else None

        isValid = lambda bbox: bbox[2] != bbox[0] and bbox[3] != bbox[1]

        font = self.font
        assert font.Subtype == '/Type3'

        bbox = None

        if font.FontBBox:
            bbox = BOX([float(x) for x in font.FontBBox])
            if isValid(bbox): return bbox

        for gname,proc in font.CharProcs.items():

            stream = PdfFilter.uncompress(proc).stream
            tree = PdfStream.stream_to_tree(stream)

            proc_bbox = get_bbox(tree[0])
            if proc_bbox == None:
                err(f'Font: {font}\nGlyph ({gname}) process stream has no d0/d1 operator:\n{proc.stream}')
            bbox = bbox + proc_bbox if bbox != None else proc_bbox

        if not isValid(bbox):
            warn(f'bad Type3 bbox: {bbox}, font: {font}')
            bbox = BOX([0,0,1,1])

        return bbox

    # -------------------------------------------------------------------------------- subset()

    def subset(self):
        '''
        Subsets a /TrueType font: removes all glyphs from the font program that are not used by the font.
        '''
        if self.font.Subtype != '/TrueType': return

        # Read the font program into a TTFont
        font = self.font
        fd = font.FontDescriptor
        ff = PdfFilter.uncompress(fd.FontFile2)
        import io
        from fontTools.ttLib import TTFont
        ttf = TTFont(io.BytesIO(py23_diffs.convert_store(ff.stream)))

        # Subset the TTFont
        from fontTools import subset
        subsetter = subset.Subsetter()
        allGlyphs = ttf.getGlyphNames()
        glyphs = [g[1:] for g in self.encoding.cc2glyphname.values() if g[1:] in allGlyphs]
        # print(glyphs)
        # print(self.encoding.cc2glyphname)
        subsetter.populate(glyphs = glyphs)
        subsetter.subset(ttf)

        # Save the subset font program
        bs = io.BytesIO() ; ttf.save(bs) ; bs.seek(0)
        stream = bs.read().decode('Latin-1')
        fd.FontFile2 = IndirectPdfDict(Length1 = len(stream), stream=stream)

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
        font = None

        for name in fontNames:

            if name in self.fontsCache:
                return self.fontsCache[name]

            if not forceCID:
                fontDict = PdfFontCore14.make_core14_font_dict(PdfName(name))
                if fontDict != None:
                    font = PdfFont(font = fontDict)
                    nameFound = name
                    break
                
            path = PdfFontUtils.findFile(name+'.*', dirList)
            if path != None:
                name,ext = os.path.splitext(path)
                data = open(path, 'rb').read()
                cc2unicode = {chr(i):chr(i) for i in range(65536)}
        
                font = PdfFont(ttf = data, cc2unicode = cc2unicode) if ext == '.ttf' \
                        else PdfFont(otf = data, cc2unicode = cc2unicode) if ext == '.otf' \
                        else None
                if font:
                    nameFound = name
                    break

        if font == None:
            warn(f'failed to load any font: {fontNames}') ; return None
        self.fontsCache[nameFound] = font
        return font

# ============================================================================= main()

if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument('filePath', metavar='FILE', help='input file: a font (TrueType, OpenType or Type1) or a PDF file')

    options = ap.parse_args()
    name,ext = os.path.splitext(options.filePath)
    ext = ext.lower()

    file = open(options.filePath, 'rb').read()

    if ext in ['.pfa', '.pfb']:
        font = PdfFont(pfb=file)
    elif ext == '.cff':
        font = PdfFont(cff=file, force_CID=False)
    elif ext == '.ttf':
        font = PdfFont(ttf=file)
    elif ext == '.otf':
        font = PdfFont(otf=file)
    else:
        raise ValueError(f'bad file extension: {ext}')
    
    outPath = options.filePath + '.pdf'
    pdf = PdfWriter(outPath, compress=True)
    for page in font.fontTableToPdfPages():
        pdf.addPage(page)
    pdf.write()

    print(f'Output written to {outPath}')

    # pprint(font.info)
