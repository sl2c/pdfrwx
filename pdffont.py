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
from pdfrwx.pdfobjects import PdfObjects

# fontLib
try:
    from fontTools.ttLib import TTFont, newTable
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


# ================================================== class PdfFontDictFunc

class PdfFontDictFunc:
    '''
    Miscellaneous static font utility functions that operate on PdfDict-s
    '''

    # -------------------------------------------------------------------------------- is_simple()

    @staticmethod
    def is_simple(font:PdfDict):
        '''
        Checks if the font is a simple font (i.e. if font.Subtype is one of
        '/Type1', '/Type3', '/TrueType', '/MMType1')
        '''
        return font.Subtype in ['/Type1', '/Type3', '/TrueType', '/MMType1']

    # -------------------------------------------------------------------------------- is_cid()

    @staticmethod
    def is_cid(font:PdfDict):
        '''
        Checks if the font is a CID font (i.e. if font.Subtype is '/Type0')
        '''
        # return font.Subtype == '/Type0' and not isinstance(font.Encoding, PdfDict) --- ???
        return font.Subtype == '/Type0'

    # -------------------------------------------------------------------------------- is_embedded()

    @staticmethod
    def is_embedded(font:PdfDict):
        '''
        Checks if the font is embedded
        '''
        fd = PdfFontDictFunc.get_font_descriptor(font)
        return fd and (fd.FontFile or fd.FontFile2 or fd.FontFile3)

    # -------------------------------------------------------------------------------- is_symbolic()

    @staticmethod
    def is_symbolic(font:PdfDict):
        '''
        '''
        try: Flags = int(PdfFontDictFunc.get_font_descriptor(font).Flags)
        except: Flags = None

        if Flags is not None:
            isSymbolic = (Flags & 4 == 4)
            isNonsymbolic = (Flags & 32 == 32)
            assert isSymbolic ^ isNonsymbolic
        else:
            isSymbolic = None

        return isSymbolic

    # -------------------------------------------------------------------------------- get_font_name()

    @staticmethod
    def get_font_name(font:PdfDict):
        '''
        Returns font's name: `font.Name` for a Type3 font and `font.BaseFont` for all others.
        Note that a font may have no name, in which case `None` is returned.
        '''
        return font.Name if font.Subtype == '/Type3' else font.BaseFont

    # -------------------------------------------------------------------------------- get_font_descriptor()

    @staticmethod
    def get_font_descriptor(font:PdfDict):
        '''
        Get the font's descriptor dictionary
        '''
        return font.FontDescriptor if not PdfFontDictFunc.is_cid(font) else font.DescendantFonts[0].FontDescriptor

    # -------------------------------------------------------------------------------- get_cidset()

    @staticmethod
    def get_cidset(font:PdfDict):
        '''
        Returns a CIDSet of a CID font as a list of chars, or None if it does not exist
        '''
        assert PdfFontDictFunc.is_cid(font)
        CIDSet = font.DescendantFonts[0].FontDescriptor.CIDSet
        if CIDSet:
            byteStream = py23_diffs.convert_store(PdfFilter.uncompress(CIDSet).stream)
            return [chr(i*8 + j) for i,byte in enumerate(byteStream) for j,bit in enumerate(f'{byte:08b}') if bit == '1']
        else:
            return None

    # -------------------------------------------------------------------------------- get_cidtogidmap()

    @staticmethod
    def get_cidtogidmap(font:PdfDict):
        '''
        Returns a `CIDToGIDMap` as a dictionary `{cid:gid}`.
        '''
        assert PdfFontDictFunc.is_cid(font)
        CIDToGIDMap = font.DescendantFonts[0].CIDToGIDMap
        if CIDToGIDMap and isinstance(CIDToGIDMap, PdfDict):
            b = py23_diffs.convert_store(PdfFilter.uncompress(CIDToGIDMap).stream)
            cid2gid = {chr(i):chr(b[2*i]*256 + b[2*i+1]) for i in range(len(b) // 2)}
            cid2gid = {cid:gid for cid,gid in cid2gid.items() if gid != chr(0)}
            return cid2gid
        else:
            return None

    # -------------------------------------------------------------------------------- get_missingWidth()

    @staticmethod
    def get_missingWidth(font:PdfDict):
        '''
        Returns the MissingWidth (in document units)
        '''
        assert font
 
        z = abs(float(font.FontMatrix[0])) if font.FontMatrix != None else 0.001
        if font.Subtype == '/Type3': return 0
        elif font.Subtype == '/Type0':
            return z * float(font.DescendantFonts[0].DW or '0')
        else:
            fd = font.FontDescriptor
            return z * float(fd.MissingWidth or '0') if fd else 0

    # -------------------------------------------------------------------------------- get_type3_bbox()

    @staticmethod
    def get_type3_bbox(font:PdfDict):
        '''
        '''
        from pdfrwx.pdffilter import PdfFilter
        from pdfrwx.pdfstreamparser import PdfStream

        get_bbox = lambda cmd: BOX([0, 0, cmd[1][0], cmd[1][0]]) if cmd[0] == 'd0' \
                                    else BOX(cmd[1][2:6]) if cmd[0] == 'd1' else None

        isValid = lambda bbox: bbox[2] != bbox[0] and bbox[3] != bbox[1]

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

    # -------------------------------------------------------------------------------- get_subtype_string()

    @staticmethod
    def get_subtype_string(font:PdfDict):
        '''
        Returns `subtype + suffix` where `subtype` is one of:
        
        `/Type1, /Type3, /TTF, /MMF, /CID0, /CID2`

        and suffix is:
        
        - `'C'` if `FontFile3.Subtype` is `/Type1C` or `/CIDFontType0C`;
        - `'O'` if `FontFile3.Subtype` is `/OpenType`;
        - absent if `FontFile3` font program is absent.
        
        See PDF Ref. v1.7 sec. 5.8, table 5.23.
        '''
        suffixes = {'/Type1C':'C', '/CIDFontType0C':'C', '/OpenType':'O'}
        subtypes = {'/Type1':'/Type1', '/Type3':'/Type3', '/TrueType':'/TrueType', '/MMType1':'/MMType1', '/CIDFontType0':'/CIDType0', '/CIDFontType2':'/CIDType2'}

        if not font: return None

        f = font if font.Subtype != '/Type0' else font.DescendantFonts[0]

        subtype = subtypes.get(f.Subtype)

        try: suffix = suffixes.get(f.FontDescriptor.FontFile3.Subtype)
        except: suffix = ''

        return subtype + suffix

    # -------------------------------------------------------------------------------- get_encoding_string()

    @staticmethod
    def get_encoding_string(font:PdfDict):
        '''
        String representation of the font's encoding
        '''
        if not font: return None
        if not PdfFontDictFunc.is_cid(font):
            enc = font.Encoding
            if not isinstance(enc, PdfDict): return enc
            return f'[{enc.BaseEncoding}, /Differences]'
        else:
            dFont = font.DescendantFonts[0]
            s = []
            if dFont.CIDToGIDMap: s.append('/CIDToGIDMap')
            if dFont.FontDescriptor.CIDSet: s.append('/CIDSet')
            s = ' + '.join(s)
            return None if len(s) == 0 else s


# =========================================================================== class PdfFontFile

class PdfFontFile:
    '''
    Utility class for font files input/output operations
    '''

    # -------------------------------------------------------------------------------- read_pfb_info()

    @staticmethod
    def read_pfb_info(font:bytes):
        '''
        '''
        with tempfile.TemporaryDirectory() as tmp:
            T = lambda fileName: os.path.join(tmp, fileName)
            open(T('tmp.pfb'),'wb').write(font)
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
        info['gid2gname'] = {i:gname for i,gname in enumerate(t1.font['Encoding']) if gname != '.notdef'}
        info['gname2width'] = {gname:chars[gname].width for gname in chars.keys()}

        # Return info
        return info

    # -------------------------------------------------------------------------------- read_ttf_otf_info()

    @staticmethod
    def read_ttf_otf_info(font:bytes):
        '''
        Returns font info as a dictionary.
        The `ttf` argument is any TrueType/OpenType font as a bytes string.
        '''
        ttFont = TTFont(BytesIO(font))

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

        fontName = info.get('FontName')

        info['numGlyphs'] = ttFont['maxp'].numGlyphs

        # Geometric parameters
        info['unitsPerEm'] = ttFont['head'].unitsPerEm
        z = float(1000/info['unitsPerEm'])
        minMax = [ttFont['head'].xMin, ttFont['head'].yMin, ttFont['head'].xMax, ttFont['head'].yMax]
        info['FontBBox'] = [int(z*x) for x in minMax]

        try:
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
        except:
            os2 = None
            warn(f'failed to get OS/2 table from font: {fontName}')

        # Stylistic parameters
        if post := ttFont.get('post'):
            info['ItalicAngle']     = post.italicAngle
            info['isFixedPitch']    = post.isFixedPitch

        if os2:
            info['isSerif']         = os2.panose.bFamilyType == 2 and os2.panose.bSerifStyle
            info['isScript']        = os2.panose.bFamilyType == 3

        # info['Widths'] = PdfArray(ttf.metrics.widths)
        # info['DefaultWidth'] = int(round(ttf.metrics.defaultWidth, 0))

        # Set maps
        try:
            glyphSet = ttFont.getGlyphSet()
        except:
            glyphSet = None
            warn(f'failed to get glyphSet from font: {fontName}')

        if glyphSet is not None:
            try: info['gid2gname'] = {ttFont.getGlyphID(gname):gname for gname in glyphSet}
            except: warn(f'failed to get gid2gname from font: {fontName}')
            try: info['gname2width'] = {gname:glyphSet[gname].width * z for gname in glyphSet}
            except: warn(f'failed to get from gname2width font: {fontName}')
        
        if cmap := ttFont.get('cmap'):
            combo = lambda table: (table.platformID, table.platEncID)
            info['isSymbolic'] = any(combo(table) == (3,0) for table in cmap.tables)
            for table in cmap.tables:
                m = {chr(code):gname for code,gname in table.cmap.items()}
                if combo(table) == (1,0): info['cmap10'] = m
                if combo(table) == (3,1): info['cmap30'] = m

            info['unicode2gname'] = ttFont.getBestCmap()

        # if re.search('MBBIVI', info['FontName']):
        #     ttFont.save('dump.ttf')
        #     pprint(info)
        #     sys.exit()

        # Return the result
        return info

    # -------------------------------------------------------------------------------- merge_cff_font_sets()

    @staticmethod
    def merge_cff_font_sets(cffFont1:CFFFontSet, cffFont2:CFFFontSet):
        '''
        Attempts to merge two CFF fonts. If the fonts are compatible, cffFont2 is updated with entries
        from cffFont1, and the function returns True; otherwise, the two fonts are unchanged and
        the function returns False.
        '''
        cs1 = cffFont1[0].CharStrings
        cs2 = cffFont2[0].CharStrings

        for s in cs1.values(): s.compile()
        for s in cs2.values(): s.compile()

        # check compatibility
        if any(gname in cs2.keys() and cs1[gname].bytecode != cs2[gname].bytecode for gname in cs1.keys()):
            return False

        # check non-zero overlap
        if not any(gname in cs2.keys() for gname in cs1.keys()):
            return False
        
        # update cffFont2 with entries from cffFont1
        for gname in cs1.keys():
            if gname not in cs2.keys():
                cs2[gname] = cs1[gname]
                cffFont2[0].charset.append(gname)

        return True

    # -------------------------------------------------------------------------------- read_cff_info()

    @staticmethod
    def read_cff_info(cff:bytes):
        '''
        '''
        assert cff

        info = {}

        # Parse the CFF data
        cffFont = CFFFontSet()
        cffFont.decompile(file = BytesIO(cff), otFont = TTFont())

        # Access the first font in the CFF font set
        assert len(cffFont.fontNames) == 1
        info['FontName'] = cffFont.fontNames[0]
        font = cffFont[0]

        # Presence of ROS means a CID-encoded CFF font
        info['ROS'] = font.ROS if hasattr(font, 'ROS') else None

        try:
            chars = font.CharStrings    
        except:
            warn(f'CFF font has no CharStrings: {info["FontName"]}')
            chars = {}

        # This initializes char.width-s
        for char in chars.values():
            char.draw(NullPen())

        # gid2gname
        if info['ROS']:
            info['gid2gname'] = {(int(gname[3:]) if gname != '.notdef' else 0):gname for gname in chars.keys()}
        else:
            try:
                info['gid2gname'] = {i:gname for i,gname in enumerate(font.Encoding) if gname != '.notdef'}
            except:
                warn(f'CFF font has no Encoding: {info["FontName"]}')
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

        # Return info
        return info

    # -------------------------------------------------------------------------------- read_core14_info()

    @staticmethod
    def read_core14_info(fontName:str):
        '''
        '''
        info = {'FontName':fontName}

        # gid2gname
        baseEncodingName = PdfFontCore14.built_in_encoding(fontName)
        if baseEncodingName:
            baseEncoding = PdfFontEncoding(name = baseEncodingName)
            info['gid2gname'] = {ord(cc):gname[1:] for cc,gname in baseEncoding.cc2glyphname.items()}

        # gname2width
        name2width = PdfFontCore14.make_name2width(fontName)
        if name2width:
            info['gname2width'] = {name[1:]:width for name,width in name2width.items()}

        return info

    # -------------------------------------------------------------------------------- fix_ttf_font()

    @staticmethod
    def fix_ttf_font(font:bytes):
        '''
        Adds an empty (3,1) cmap to the TTF font
        When such font is then used in a PDF this essentially means
        an internal encoding such that "cid == gid"
        '''

        ttFont = TTFont(BytesIO(font))
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
            font = bs.read()
        
        return font

    # -------------------------------------------------------------------------------- print_xml()

    @staticmethod
    def cff_to_xml(cff:bytes, filePath:str):
        '''
        Writes a CFF font program to file.
        '''
        from fontTools.misc.xmlWriter import XMLWriter
        cffFont = CFFFontSet()
        cffFont.decompile(file = BytesIO(cff), otFont = TTFont())
        cffFont.toXML(XMLWriter(filePath))

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
 
            self.info = PdfFontFile.read_pfb_info(self.pfb)
            self.encoding = self.make_encoding(cc2gname = cc2gname, cc2unicode = cc2unicode)
            self.font = self.make_font_dict(encoding = self.encoding, force_CID = force_CID)
 
        elif self.cff:

            self.info = PdfFontFile.read_cff_info(self.cff)
            self.encoding = self.make_encoding(cc2gname = cc2gname, cc2unicode = cc2unicode)
            self.font = self.make_font_dict(encoding = self.encoding, force_CID = force_CID)

        elif self.ttf or self.otf:

            self.info = PdfFontFile.read_ttf_otf_info(self.ttf or self.otf)
            self.encoding = self.make_encoding(cc2gname = cc2gname, cc2unicode = cc2unicode)
            self.font = self.make_font_dict(encoding = self.encoding, force_CID = force_CID)
 
        elif self.font:

            if extractFontProgram:
 
                self.extract_font_program()

                self.info = PdfFontFile.read_pfb_info(self.pfb) if self.pfb \
                                else PdfFontFile.read_ttf_otf_info(self.ttf or self.otf) if (self.ttf or self.otf) \
                                else PdfFontFile.read_cff_info(self.cff) if self.cff \
                                else PdfFontFile.read_core14_info(self.get_font_name())

            self.encoding = self.get_cid_encoding_from_type0_font() if self.is_cid() \
                                else PdfFontEncoding(font = self.font, fontInfo = self.info)

        else:

            raise ValueError(f'at least one of the arguments must be specified: pfb/cff/ttf/otf/font')

        # Set font name
        self.name = self.get_font_name()

        # Set chars widths
        self.widthMissing = PdfFontDictFunc.get_missingWidth(self.font)
        self.cc2width = self.get_cc2width_from_font()

        # Set fontMatrix
        self.fontMatrix = MAT(self.font.FontMatrix) if self.font.FontMatrix != None \
                            else MAT([0.001, 0, 0, 0.001, 0, 0])
        
        # Set font's bounding box
        if self.font.Subtype == '/Type3':
            self.bbox = PdfFontDictFunc.get_type3_bbox(self.font)
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
            cc2gname = {chr(gid):gname for gid, gname in self.info['gid2gname'].items()}

 
        encoding = PdfFontEncoding()
        encoding.cc2glyphname = {cc:PdfName(gname) for cc,gname in cc2gname.items()}
        encoding.reset_glyphname2cc()
        encoding.name = [None, PdfName('Differences')]

        return encoding

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

        # Get cid2gid (str -> str)
        cid2gid = PdfFontDictFunc.get_cidtogidmap(self.font)
        if not cid2gid:
            cid2gid = {cc:cc for cc in self.get_cc2width_from_font()}

        # Append cids from CIDSet if it exists
        CIDSet = PdfFontDictFunc.get_cidset(self.font)
        if CIDSet:
            for cid in CIDSet:
                if cid not in cid2gid: cid2gid[cid] = cid

        if len(cid2gid) == 0:
            name = self.font.BaseFont or self.DescendantFonts[0].BaseFont
            cid2gid = {chr(gid):chr(gid) for gid in gid2gname}

        # Create cid2gname
        cc2gname = {cid:gid2gname.get(ord(gid),'.notdef') for cid,gid in sorted(cid2gid.items())}

        # Create encoding
        encoding = PdfFontEncoding()
        encoding.name = PdfName('CIDEncoding')
        encoding.cc2glyphname = {cc:PdfName(gname) for cc,gname in cc2gname.items()}
        encoding.reset_glyphname2cc()

        return encoding

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

    def get_cid2width_from_info(self, encoding:PdfFontEncoding = None):
        '''
        Get a map form CIDs to widths (int -> float) from self.info
        '''
        if encoding:
            cid2gname = {ord(cc):gname[1:] for cc,gname in encoding.cc2glyphname.items()}
        else:
            cid2gname = self.info['gid2gname']
        gname2width = self.info['gname2width']
        return {cid:gname2width.get(gname,0) for cid,gname in cid2gname.items()}

    # -------------------------------------------------------------------------------- make_font_dict()

    def make_font_dict(self, encoding:PdfFontEncoding = None, force_CID:bool = False):

        # ................................................................................ make_flags()

        def make_flags(info:dict):
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

            fontProgram = PdfFontFile.fix_ttf_font(self.ttf)
 
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
            Flags = make_flags(self.info),
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
                cid2gid = {ord(cc):gname2gid.get(gname[1:],0) for cc,gname in encoding.cc2glyphname.items()}
                gids = [cid2gid.get(cid,0) for cid in range(maxCID + 1)]
                CIDToGIDMap = IndirectPdfDict(
                    stream=py23_diffs.convert_load(b''.join(bytes([gid >> 8, gid & 255]) for gid in gids))
                )
            
            # Widths
            W = PdfArray([x for cid,w in self.get_cid2width_from_info(encoding).items() for x in [cid,PdfArray([w])]])

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

            Widths = PdfArray([self.get_cid2width_from_info(encoding).get(i,0) for i in range(FirstChar, LastChar+1)])

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

        fd = PdfFontDictFunc.get_font_descriptor(self.font)

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
            
    
    # -------------------------------------------------------------------------------- save()

    def save(self, basePath:str = None):
        '''
        The basePath arguments is is the intended path to the saved font file without the file's extension.
        If it's None then the self.name is chosen as the file name, and the file is saved in the current folder.
        '''
        fontProgram = self.pfb or self.ttf or self.otf or self.cff
        if fontProgram is None:
            raise ValueError(f'no font program in font: {self.name}')
        if basePath is None:
            basePath = self.name[1:]
        ext = 'pfb' if self.pfb else 'ttf' if self.ttf else 'otf' if self.otf else 'cff' if self.cff else None

        # if ext == 'cff':
        #     PdfFontFile.cff_to_xml(self.cff, basePath + '.xml')
        #     return

        open(basePath + '.' + ext, 'wb').write(fontProgram)
        return

    # -------------------------------------------------------------------------------- get_font_name()

    def get_font_name(self):
        '''
        This is a wrapper function around `PdfFontDictFunc.get_font_name()` which does some extra work:

        * if `self.font` doesn't have a name, this function will return `f'/T3Font{N}'`, where `N`
        is a consecutive integer (note: Type3 font type is the only one that is allowed not to have a name);
        * if the name of `self.font` has been encountered in previous calls to this function, it will
        return a "versioned" variant of the name `f'{name}-v{N}'`, where `N` is a consecutive integer.

        Altogether, this ensures that names returned for different fonts never coincide.
        '''
        nDuplicates = lambda d, k: d.get(k) or d.update({k:len(d)+1}) or len(d) # duplicates counter
        f = self.font

        result = PdfFontDictFunc.get_font_name(f)

        if result not in self.__fontNameCache: self.__fontNameCache[result] = {}
        idx = nDuplicates(self.__fontNameCache[result], id(f))

        result = result + (f'-v{idx}' if idx > 1 and not PdfFontCore14.standard_fontname(result) else '') \
                    if result != None else ('/T3Font' if f.Subtype == '/Type3' else '/NoName') + f'{idx}'
        
        return PdfName(result[1:])

    # -------------------------------------------------------------------------------- install()

    def install(self, pdfPage:PdfDict, fontName:str, overwrite:bool = False):
        '''
        Adds self.font to the pdfPage.Resources.Font dictionary under the name fontName.
        The font can then be referred to using Tf operators in the pdfPage.stream
        '''
        if pdfPage.Resources == None: pdfPage.Resources = PdfDict()
        if pdfPage.Resources.Font == None: pdfPage.Resources.Font = PdfDict()
        if PdfName(fontName) in pdfPage.Resources.Font:
            warn(f'font {fontName} already in page Resources; overwrite = {overwrite}')
            if not overwrite: return
        pdfPage.Resources.Font[PdfName(fontName)] = self.font

    # -------------------------------------------------------------------------------- is_simple()

    def is_simple(self):
        '''
        Checks if the font is a simple font (i.e. if font.Subtype is one of
        '/Type1', '/Type3', '/TrueType', '/MMType1')
        '''
        return PdfFontDictFunc.is_simple(self.font)

    # -------------------------------------------------------------------------------- is_cid()

    def is_cid(self):
        '''
        Checks if the font is a CID font (i.e. if font.Subtype is '/Type0')
        '''
        return PdfFontDictFunc.is_cid(self.font)

    # -------------------------------------------------------------------------------- is_embedded()

    def is_embedded(self):
        '''
        Checks if the font is embedded
        '''
        return PdfFontDictFunc.is_embedded(self.font)

    # -------------------------------------------------------------------------------- is_embedded()

    def is_symbolic(self):
        '''
        Checks if the font is symbolic
        '''
        return PdfFontDictFunc.is_symbolic(self.font)

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

            # Exclude code that are actually not in the font.
            # Assumption: whether a code (cc) is in the font is first checked by
            # mapping to glyph name and looking up the glyph name in the font, and if that
            # fails then just looking up a symbol in the font with gid equal to cc
            gname2width = self.info.get('gname2width')
            gid2gname = self.info.get('gid2gname')
            cc2width = {cc:ww for cc,ww in cc2width.items() \
                        if gname2width and self.encoding.cc2glyphname.get(cc, '/None')[1:] in gname2width \
                            or gid2gname and ord(cc) in gid2gname \
                            or gid2gname is None and gname2width is None}

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

            if cid2gid := PdfFontDictFunc.get_cidtogidmap(self.font):
                for cc in cid2gid:
                    if cc not in cc2width: cc2width[cc] = defaultWidth
            elif cidset := PdfFontDictFunc.get_cidset(self.font):
                for cc in cidset:
                    if cc not in cc2width: cc2width[cc] = defaultWidth

            if len(cc2width) == 0:
                gid2gname = self.info.get('gid2gname')
                if gid2gname:
                    cc2width = {chr(gid):defaultWidth for gid in gid2gname}


        # Rescale from font units to document units
        # For Type 3 fonts, widths should be scaled by the FontMatrix; PDF Ref Sec. 5.5.4 Type 3 fonts
        z = abs(float(font.FontMatrix[0])) if font.FontMatrix != None else 0.001
        cc2width = {cc: w*z for cc,w in sorted(cc2width.items())}

        return cc2width

    # -------------------------------------------------------------------------------- __repr__()

    def __repr__(self):
        '''
        A string representation of a font
        '''
        subtype = PdfFontDictFunc.get_subtype_string(self.font)
        encoding = PdfFontDictFunc.get_encoding_string(self.font)
        return f'{subtype} {self.name} {encoding}'

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

            gname = cc2g.get(cc, None)
            if gname == None: continue

            # The math
            cid = ord(cc)
            col,row,n = cid % 16, (cid // 16 ) % 16, (cid // 256)
            if n not in counter: streams[n] += f'/A 3 Tf 0.75 g\n' ; counter[n] = 1
            x,y = 10*(2*col), -2*10*row

            # Replace non-printable chars according to name syntax (see PDF Ref. sec. 3.2.4)           
            gname = ''.join(f'#{ord(c):02x}' if ord(c) <= 0x20 or 0x7f <= ord(c) < 0xa0 or c == '#' else c for c in gname)

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
            subtype = PdfFontDictFunc.get_subtype_string(self.font)
            encoding = PdfFontDictFunc.get_encoding_string(self.font)
            subtitle = f'{subtype}, Encoding: {encoding}{ToUnicode}{Embedded}'
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

                glyphMap = PdfFontGlyphMap(loadAdobeGlyphList = True)
                font = PdfFont(ttf = data, glyphMap = glyphMap, cc2unicode = cc2unicode) if ext == '.ttf' \
                        else PdfFont(otf = data, glyphMap = glyphMap, cc2unicode = cc2unicode) if ext == '.otf' \
                        else None
                if font:
                    nameFound = name
                    break

        if font == None:
            warn(f'failed to load any font: {fontNames}') ; return None
        self.fontsCache[nameFound] = font
        return font

# ------------------------------------------------------- get_object_fonts()

def get_object_fonts(xobj:PdfDict,
                        fontTypes:list[str] = None,
                        regex:str = None):
    '''
    Returns a `{id(font):font}` dictionary of fonts used by `xobj` whose Subtypes match those in the `fontTypes` list.
    Setting `fontTypes = None` has the same effect as setting it to:

    `['/Type1', '/MMType1', '/TrueType', '/Type3', '/Type0']`.

    Example:
    
    `get_object_fonts(pdf, fontTypes = ['/Type3']).`
    
    If the `regex` argument is provided only the fonts whose names match the regex are selected. Fonts
    that have no name are always included.
    '''
    cache = set()

    fontName = PdfFontDictFunc.get_font_name

    if fontTypes is None:
        fontTypes = ['/Type1', '/MMType1', '/TrueType', '/Type3', '/Type0']

    fontFilter = lambda obj: \
        isinstance(obj, PdfDict) \
        and obj.Type == PdfName.Font \
        and obj.Subtype in fontTypes \
        and (regex in [None, '.'] or fontName(obj) is not None and re.search(regex, fontName(obj), re.IGNORECASE))

    if xobj.pages:
        objectTuples = [t for page in xobj.pages for t in PdfObjects(page, cache=cache)]
    else:
        objectTuples = PdfObjects(xobj, cache=cache)

    return {id(obj):obj for name, obj in objectTuples if fontFilter(obj)}

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
