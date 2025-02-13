#!/usr/bin/env python3

# ================================================== imports

# pdfrw
try:
    from pdfrw import PdfDict, IndirectPdfDict, PdfString, PdfArray, PdfName, py23_diffs, PdfWriter
except:
    raise SystemError(f'import pdfrw failed; run: pip3 install pdfrw')

# pdfrwx
from pdfrwx.common import err, warn, msg, getExecPath
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
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.misc.xmlWriter import XMLWriter
except:
    raise SystemError(f'import fonttools failed; run: pip3 install fonttools')

# misc
import sys, re, os, tempfile, argparse, random, string, zlib, base64, struct
from io import BytesIO
from pprint import pprint
from typing import Union

# ================================================== Typedef
    
ENC_TYPE = Union[PdfName, PdfDict]

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

    # --------------------------------------------------------------------------- format()

    def format(self):
        '''
        Returns the format of self: 'literal', 'hex', or None if format is invalid.
        '''
        if self.startswith('(') and self.endswith(')'): return 'literal'
        elif self.startswith('<') and self.endswith('>'): return 'hex'
        else: None

    # --------------------------------------------------------------------------- to_codes()

    def to_codes(self, isCID = False):
        return self.to_bytes().decode('utf-16-be' if isCID else 'latin1')

    # --------------------------------------------------------------------------- from_codes()

    @classmethod
    def from_codes(cls, codeString:str, format:str = 'auto', forceCID = False):
        if format not in ['auto', 'hex', 'literal']:
            raise ValueError(f'format should be one of: auto, hex, literal')
        isCID = any(ord(c)>255 for c in codeString) or forceCID
        suitableFormat = 'hex' if isCID else 'literal'
        format = format if format != 'auto' else suitableFormat # Do not let from_bytes() decide on the format
        bytes = codeString.encode('utf-16-be' if isCID else 'latin1')
        return cls.from_bytes(bytes, bytes_encoding=format)

    # --------------------------------------------------------------------------- to_bytes()

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

    # --------------------------------------------------------------------------- from_bytes()

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

    # --------------------------------------------------------------------------- UTF16BE_to_Unicode()

    @staticmethod
    def UTF16BE_to_Unicode(hexStr:str):
        '''Converts a utf-16be-encoded hex string to a Unicode string;
        hex strings of length < 4 are treated as Unicode hex values;
        for badly formatted hexStr the returned value is None.
        '''
        # try: return bytes.fromhex(hexStr).decode('utf-16-be') if len(hexStr) >= 4 else chr(int(hexStr,16))
        # except: return None
        try: return bytes.fromhex(hexStr).decode('utf-16-be')
        except:
            try: return chr(int(hexStr,16))
            except: return None

    # --------------------------------------------------------------------------- Unicode_to_UTF16BE()

    @staticmethod
    def Unicode_to_UTF16BE(unicodeStr:str):
        '''Converts a Unicode string to a utf-16be-encoded hex string
        '''
        # return unicodeStr.encode('utf-16-be','surrogatepass').hex()
        return unicodeStr.encode('utf-16-be').hex().upper()

# =========================================================================== AdobeGlyphMap

def readGlyphList(glyphListNames:list[str]):
    '''
    Returns a Glyph Map read from the list of glyphLists located in `EXEC_PATH/glyph-lists/`.
    The glyphLists are plain text files which contain maps from glyph names to Unicode values
    in the Adobe Glyph List format. The original Adobe Glyph List can be found here:
    https://github.com/adobe-type-tools/agl-aglfn
    '''
    # if decodeType3GlyphNames: glyphMapNames.append(glyphMapNameType3)
    glyphMapPaths = [os.path.join(getExecPath(), 'glyph-lists', name) for name in glyphListNames]

    GL = {} # a map from the (Adobe) standard glyph names to Unicode points

    # Initialize glyphMap
    for path in glyphMapPaths:
        with open(path, 'r') as file:
            for k,line in enumerate(file):
                s = line.strip(' \t\r\n')
                if s == '' or s[0] == '#': continue
                try: glyphName, codePointHex = re.split(';',s)
                except: err(f'malformed line # {k} in a glyph list: {path}')
                codePointHex = re.sub(r'[^0-9a-fA-F].*$','',codePointHex) # remove non-hex chars from 1st occurrence to end
                if len(codePointHex) < 4: err(f'malformed line # {k} in a glyph list: {path}')
                GL[PdfName(glyphName)] = PdfTextString.UTF16BE_to_Unicode(codePointHex)

    return GL

AdobeGlyphMap = readGlyphList(['glyphlist.txt'])
MyGlyphMap = readGlyphList(['glyphlist.txt', 'glyphs-nimbus.txt', 'wingdings.txt', 'my-glyphlist.txt'])

# =========================================================================== class PdfCore14Fonts

class PdfFontCore14:
        
   # ascent/descent font parameters
    ASCENT_DESCENT = {
        '/Courier': (629, -157),
        '/Courier-Bold': (626, -142),
        '/Courier-BoldOblique': (626, -142),
        '/Courier-Oblique': (629, -157),
        '/Helvetica': (718, -207),
        '/Helvetica-Bold': (718, -207),
        '/Helvetica-BoldOblique': (718, -207),
        '/Helvetica-Oblique': (718, -207),
        '/Times-Roman': (683, -217),
        '/Times-Bold': (676, -205),
        '/Times-BoldItalic': (699, -205),
        '/Times-Italic': (683, -205),
        '/Symbol': (0, 0),
        '/ZapfDingbats': (0, 0)
        }

   # A map from aliases to stanard names
    CORE14_FONTNAMES_ALIASES = {
        '/CourierNew':'/Courier', '/CourierNew,Italic':'/Courier-Oblique',
        '/CourierNew,Bold':'/Courier-Bold', '/CourierNew,BoldItalic':'/Courier-BoldOblique',
        '/Arial':'/Helvetica', '/Arial,Italic':'/Helvetica-Oblique',
        '/Arial,Bold':'/Helvetica-Bold', '/Arial,BoldItalic':'/Helvetica-BoldOblique',
        '/TimesNewRoman':'/Times-Roman', '/TimesNewRoman,Italic':'/Times-Italic',
        '/TimesNewRoman,Bold':'/Times-Bold', '/TimesNewRoman,BoldItalic':'/Times-BoldItalic',
        '/Symbol':'/Symbol', '/ZapfDingbats':'/ZapfDingbats'
    }

    @staticmethod
    def standard_fontname(fontname:PdfName):
        '''
        If the fontname is a standard core14 font name returns fontname. Otherwise, if
        it is an alias for a standard core14 font name, returns the corresponding standard core 14 font name.
        If none of the above, returns None. The standard core14 font names and their aliases are in
        PdfFontUtils.CORE14_FONTNAMES_ALIASES; see PDF Reference 1.7 Sec. 5.5.1; Sec. H.3 Implementation notes
        for Sec. 5.5.1.
        '''
        aliases = PdfFontCore14.CORE14_FONTNAMES_ALIASES
        name = re.sub(r' *', '', fontname) # remove spaces
        name = re.sub(r'-v[0-9]+', '', name) # remove versioning
        return name if name in aliases.values() else aliases[name] if name in aliases else None
        
    @staticmethod
    def make_core14_font_dict(fontname:PdfName, encoding:ENC_TYPE = None):
        '''
        Returns a core14 font dictionary for the specified font name or None if the specified
        font name is neither a standard core14 font name nor an alias for a standard core14 font name.
        See help for PdfFontCore14.standard_fontname() for more info.

        The `encoding` argument should be one of: `/WinAnsiEncoding`, `/MacRomanEncoding` or `/StandardEncoding`.
        '''
        if PdfFontCore14.standard_fontname(fontname) == None: return None
        assert encoding is None or isinstance(encoding, ENC_TYPE)
        return IndirectPdfDict(
            Type = PdfName.Font,
            Subtype = PdfName.Type1,
            Encoding = encoding,
            BaseFont = PdfName(fontname[1:]),
        )

    @staticmethod
    def built_in_encoding(fontname:PdfName):
        '''
        If fontname is the name of one of Core 14 fonts
        (see PdfFontCore14.CORE14_FONTNAMES_ALIASES for Core 14 font names and aliases),
        returns the name of the built-in encoding of that font, otherwise returns None.
        Examples: '/Times' --> '/StandardEncoding', '/Symbol' --> '/SymbolEncoding',
        '/ZapfDingbats' --> '/ZapfDingbatsEncoding'.
        '''

        # PDF Ref. v1.7 sec. 5.5.5:
        # The base encoding [is] the encoding from which the Differences
        # entry (if present) describes differences—specified as the name of a predefined
        # encoding MacRomanEncoding, MacExpertEncoding, or WinAnsiEncoding (see Appendix D).
        # If this entry is absent, the Differences entry describes differences from an implicit
        # base encoding. For a font program that is embedded in the PDF file, the implicit base
        # encoding is the font program’s built-in encoding, as described above and further
        # elaborated in the sections on specific font types below. Otherwise, for a nonsymbolic font,
        # it is StandardEncoding, and for a symbolic font, it is the font’s built-in encoding.

        standard_fontname = PdfFontCore14.standard_fontname(fontname)
        if standard_fontname == None: return None
        return '/ZapfDingbatsEncoding' if standard_fontname == '/ZapfDingbats' \
            else '/SymbolEncoding' if standard_fontname == '/Symbol' \
            else '/StandardEncoding'

    @staticmethod
    def make_name2width(fontname:PdfName):
        '''
        '''
        standard_fontname = PdfFontCore14.standard_fontname(fontname)
        if standard_fontname == None: return None

        # Creates the names vector
        if standard_fontname == '/Symbol':
            names ='''Alpha Beta Chi Delta Epsilon Eta Euro Gamma Ifraktur Iota Kappa Lambda Mu Nu Omega Omicron
                    Phi Pi Psi Rfraktur Rho Sigma Tau Theta Upsilon Upsilon1 Xi Zeta aleph alpha ampersand angle
                    angleleft angleright apple approxequal arrowboth arrowdblboth arrowdbldown arrowdblleft arrowdblright arrowdblup arrowdown arrowhorizex arrowleft arrowright arrowup arrowvertex
                    asteriskmath bar beta braceex braceleft braceleftbt braceleftmid bracelefttp braceright bracerightbt bracerightmid bracerighttp bracketleft bracketleftbt bracketleftex bracketlefttp
                    bracketright bracketrightbt bracketrightex bracketrighttp bullet carriagereturn chi circlemultiply circleplus club colon comma congruent copyrightsans copyrightserif degree
                    delta diamond divide dotmath eight element ellipsis emptyset epsilon equal equivalence eta exclam existential five florin
                    four fraction gamma gradient greater greaterequal heart infinity integral integralbt integralex integraltp intersection iota kappa lambda
                    less lessequal logicaland logicalnot logicalor lozenge minus minute mu multiply nine notelement notequal notsubset nu numbersign
                    omega omega1 omicron one parenleft parenleftbt parenleftex parenlefttp parenright parenrightbt parenrightex parenrighttp partialdiff percent period perpendicular
                    phi phi1 pi plus plusminus product propersubset propersuperset proportional psi question radical radicalex reflexsubset reflexsuperset registersans
                    registerserif rho second semicolon seven sigma sigma1 similar six slash space spade suchthat summation tau therefore
                    theta theta1 three trademarksans trademarkserif two underscore union universal upsilon weierstrass xi zero zeta'''
        elif standard_fontname == '/ZapfDingbats':
            names ='''a1 a10 a100 a101 a102 a103 a104 a105 a106 a107 a108 a109 a11 a110 a111 a112 a117 a118 a119 a12 a120
                    a121 a122 a123 a124 a125 a126 a127 a128 a129 a13 a130 a131 a132 a133 a134 a135 a136 a137 a138 a139
                    a14 a140 a141 a142 a143 a144 a145 a146 a147 a148 a149 a15 a150 a151 a152 a153 a154 a155 a156 a157
                    a158 a159 a16 a160 a161 a162 a163 a164 a165 a166 a167 a168 a169 a17 a170 a171 a172 a173 a174 a175
                    a176 a177 a178 a179 a18 a180 a181 a182 a183 a184 a185 a186 a187 a188 a189 a19 a190 a191 a192 a193
                    a194 a195 a196 a197 a198 a199 a2 a20 a200 a201 a202 a203 a204 a205 a206 a21 a22 a23 a24 a25 a26 a27
                    a28 a29 a3 a30 a31 a32 a33 a34 a35 a36 a37 a38 a39 a4 a40 a41 a42 a43 a44 a45 a46 a47 a48 a49 a5 a50
                    a51 a52 a53 a54 a55 a56 a57 a58 a59 a6 a60 a61 a62 a63 a64 a65 a66 a67 a68 a69 a7 a70 a71 a72 a73
                    a74 a75 a76 a77 a78 a79 a8 a81 a82 a83 a84 a85 a86 a87 a88 a89 a9 a90 a91 a92 a93 a94 a95 a96 a97
                    a98 a99 space'''
        else:
            names = '''A AE Aacute Acircumflex Adieresis Agrave Aring Atilde B C Ccedilla D E Eacute Ecircumflex Edieresis
                    Egrave Eth Euro F G H I Iacute Icircumflex Idieresis Igrave J K L Lslash M
                    N Ntilde O OE Oacute Ocircumflex Odieresis Ograve Oslash Otilde P Q R S Scaron T
                    Thorn U Uacute Ucircumflex Udieresis Ugrave V W X Y Yacute Ydieresis Z Zcaron a aacute
                    acircumflex acute adieresis ae agrave ampersand aring asciicircum asciitilde asterisk at atilde b backslash bar braceleft
                    braceright bracketleft bracketright breve brokenbar bullet c caron ccedilla cedilla cent circumflex colon comma copyright currency
                    d dagger daggerdbl degree dieresis divide dollar dotaccent dotlessi e eacute ecircumflex edieresis egrave eight ellipsis
                    emdash endash equal eth exclam exclamdown f fi five fl florin four fraction g germandbls grave
                    greater guillemotleft guillemotright guilsinglleft guilsinglright h hungarumlaut hyphen i iacute icircumflex idieresis igrave j k l
                    less logicalnot lslash m macron minus mu multiply n nine ntilde numbersign o oacute ocircumflex odieresis
                    oe ogonek ograve one onehalf onequarter onesuperior ordfeminine ordmasculine oslash otilde p paragraph parenleft parenright percent
                    period periodcentered perthousand plus plusminus q question questiondown quotedbl quotedblbase quotedblleft quotedblright quoteleft quoteright quotesinglbase quotesingle
                    r registered ring s scaron section semicolon seven six slash space sterling t thorn three threequarters
                    threesuperior tilde trademark two twosuperior u uacute ucircumflex udieresis ugrave underscore v w x y yacute
                    ydieresis yen z zcaron zero'''
                            
        names = names.split()
        names = [PdfName(name) for name in names]

        # Create the widths vector
        if standard_fontname in ['/Courier','/Courier-Bold','/Courier-BoldOblique','/Courier-Oblique']:
            widthsArray = [600]*229
        else:
            # The widths vectors are store zlib- and base64-compressed to save space and prevent accidental mods
            widthsDict = {
                '/Helvetica' : b'eJxVUbENwyAQ9POpPABSdmAJu6IhI6ShTpPOA6RP7wHSeoPsYCmtJXceAClt7h+MHJ0wz/F3z8lm5M2Mf5gVx7MzkVszk61I4J1x3Ekvt7xhVZhRulUbq5968lRdI+AEFOAzoIIjfYynB3+Vt3SiCyDzAuorJexJ+rGDNWt2wPkJ1md+5/R9G9I5vbGqSaiS3DQvSZVVqsuuoVn2jKhSs+DeNwt3pdMf3XXCJErsPffQ3+lWcp3FjQdxwht8UUaKwsscxRvqVVPJy2zJbfdaPYNkyFNq3oS/kTLkJN8fS7RbrA==',
                '/Helvetica-Bold' : b'eJxtkKERwzAMRSOJZQBfhvASDTLJCiXGJWUZILw8A5Rmk94ZloUZFXWASt+OW9D7JxD5/ycpnCRz+qO1KbHnKD0ncoe0kzhq92Re6SVrNfFqbmQj8o0pW6NqWikqmtjLDKLnwIEWedhEnTPQopqggc7FW0rleDcfGDftBvRdoRbpXhnUWChwI9HdbYOSKrzqcb9XatlGTmY4I8rXgmQrVBll1PSVLvUFTM0pR3cISNrEJ70hTOhetPBuFx1/wmZicqWCkuv3Ibus/15pPPYfQchXzQ==',
                '/Times-Roman' : b'eJxlkCEOwlAMhl9bhyZz3ODJBc8BdgUMp+AAeCRkB8AikLMk6AU8cmrqiWWKv+02FsiXt/1b27/t41r2XM8ojZp3EzUljhqjYuLgGWPtj0c0nwh2Xjki10mjmiqjoIpLqmQB/eIVJTlDJcrCg97AOkJvPRcn4WSh5w7KOVLBuf8fXJUkjTSY3iOAo+6iFeFidV7lUX9mMzAD58jIMJtlDtkz7Fb03XJLkZa0HiJ3dZNN6EMvjbkkm+k0ba1b3eDb6X2CZB11737U5lnIEzr+9f1+2a4fxRyQKg==',
                '/Times-Bold' : b'eJxNkDEOwjAMRWNnQT1AJm5ATtCNgQUpV2DpWViY2LuxsHKAnqIzB4iExJQpE992G6onFNf+/rbh2WeeN4ztXZmp8OA739G1UXwnNf8SLWpZ6iuiRj6CkYet+2YWuqkoCf6Y4Y94T7xD9EAUKbg33UASEF9oQiS/opnKH9WJx50S95oPUKwU7JWhsQoQtXW4p0bWVZZa0v4VyQTuoQjYDUrc0/8dDGye9P3ylyI461fkvbj57KqruLq3TlQO7WqbNuGKZP+pOeHuKrFOEs8kN1DkxjK37WC3/gCycoQ+',
                '/Times-Italic' : b'eJxVkDEOwjAMReN4gxGUjRtkZu8BcgWWnKIH6M6I1LEIVg7QUyDEGZg6ZUCd+HbS0upLiWM//8SxkWsbV2qh5+L8pCQr9jCrB+Ot5yrT/JD6LKEjJfT53DnlwUfpyxICCuBaSrxBfLMHRBfNO9rRGWqggPhEvdwsvLzBjPZbHBKoYDvNOxCTEtdcwz1XGtQmPpl7vltr6ldWt1BPznYgnMwuZKEXwsuD7oMdyNOejiW/FTeuzGhG/qhL7riW2fUfzZsaTBF0xpSdyJlRp1dXeAZ+Ifare/H//ylz9w8xy4v0',
                '/Times-BoldItalic' : b'eJxlkLsNg0AMhs++jgFYIjcBFRnAG0Q0TJEB6NPfAGkzAFNQZwCkSFRXUeW3HR5S9Ik7nx+/bTjHF+c/prNNRc9Y0bCjnp77eEcmgMZ0orf6BHqzNzLyrc6hYgjU0AP6hRtuYLXmr8ObbkAU2B2NsPQr5ln581Mo9CDRSvhrZGyUOMdZVRHRqYUTXkkrwtN7W8z04BerP8AMmKijGrMJd9zitSs4mFzsXnihRBUdMajFa1jDGmffy/IuR9S6DdhC/J+6EvZe9XRV6InuQIl38Bo56+lbevUXENqI9A==',
                '/Symbol' : b'eJxdT6FOA0EU3JlXU+SJM9Xr8LVFVjSpX4EoEt+AwuBJ7X1A3QlMDR/QTVBnz2BIMGxyySUkxTC7oQkhk9037+17s/PYsWHHG250Jwa2WIm1dsfOnNi3OZvZjJdcq+edz7zlIzc259YuzGGJpdX0k2pSMVhvvTSCfZwZRnekR8KbTvoTV3goOMdX7uj1l+OA2p3Ea6vxhMSB3p0wMkp1jgO9EOTSY8xwezSMpTowomKbkZ2VWunWe5CSd19c5LnSHzkVazOXzn8/ya7kYseppouKNo4FLV5+NWL2KN6gVh6yNyll94N4xMGuNb/FaJXdZ6/aLnJhPT6VpR+pg2TA',
                '/ZapfDingbats' : b'eJxtkL1LglEUxt/7/BbHPvzIJpegrTahD8SIoMWloTmcbKylIRpFGnLKIQWjcKkhlDCXoCiIHHSosT8gpBpqeN+KoGNr8oNz7n2ewznnXjo6V0VPSijBHkWVFRgbXKmpFUJqECPKKeH/aGKQyupAtThQrbHDEhUC17ZYosqhUWeBHHOm7HPEseuyySIdsmqxroAGa9yRspqqPmhT0ov5sGXZIlmuyfRr6ehLvvtxB5Y/Nauavu11vt4Z4fFvgyhxxpkmSYRlkuqStluKsp71oFfzx3RhXgjPTr5udKsTXTLDlHszPWLKmVqqW8dR3RPTJMP2ZzmbZ7PZNsIMKe6a3q6LuGp/G6/n9VxeaZd3YWPeyBgFV1DFRX8BrA5Wog=='
            }
            widthsDict['/Helvetica-Oblique'] = widthsDict['/Helvetica']
            widthsDict['/Helvetica-BoldOblique'] = widthsDict['/Helvetica-Bold']
            d = zlib.decompress(base64.b64decode(widthsDict[standard_fontname]))
            widthsArray = struct.unpack('>' + 'H'*(len(d)//2), d)

        # Return the names and widths vectors zipped up into a map
        return dict(zip(names,widthsArray))

# =========================================================================== class PdfFontEncodingStandards

class PdfFontEncodingStandards:

    def get_cc2glyphname(encodingName:str):
        '''
        Returns a cc2glyphname map (with cc as char) for the given encodingName,
        or None if encodingName is not a known encoding.
        '''
        if encodingName not in PdfFontEncodingStandards.encodingVectors: return None
        encodingVector = PdfFontEncodingStandards.encodingVectors[encodingName]
        return {chr(i):PdfName(encodingVector[i]) for i in range(256) if encodingVector[i] != None}
    
    def invert_cc2glyphname(cc2glyphname:dict[str, PdfName], encodingName:str):
        '''
        Invert cc2glyphname map; special care is taken when encodingName == '/WinAnsiEncoding'
        to properly map the /space, /bullet and /hyphen glyph names.
        '''
        glyphname2cc = {gname:cc for cc,gname in cc2glyphname.items()}
        if encodingName == '/WinAnsiEncoding':
            glyphname2cc |= {'/space':chr(0x20), '/hyphen':chr(0x2D), '/bullet':chr(0x95)}
        return glyphname2cc


    # --------------------------------------------------------------------------- encodingVectors

    encodingVectors = {}

    NONE_VEC = lambda n: [None] * n

    ASCII_VEC = [
        'space','exclam','quotedbl','numbersign','dollar','percent','ampersand','quotesingle','parenleft','parenright','asterisk','plus','comma','hyphen','period','slash',
        'zero','one','two','three','four','five','six','seven','eight','nine','colon','semicolon','less','equal','greater','question',
        'at','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
        'P','Q','R','S','T','U','V','W','X','Y','Z','bracketleft','backslash','bracketright','asciicircum','underscore',
        'grave','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
        'p','q','r','s','t','u','v','w','x','y','z','braceleft','bar','braceright','asciitilde',None]

    encodingVectors['/PDFDocEncoding'] = NONE_VEC(24) \
    +  ['breve','caron','circumflex','dotaccent','hungarumlaut','ogonek','ring','tilde'] \
    +  ASCII_VEC \
    +  ['bullet','dagger','daggerdbl','ellipsis','emdash','endash','florin','fraction','guilsinglleft','guilsinglright','minus','perthousand','quotedblbase','quotedblleft','quotedblright','quoteleft',
        'quoteright','quotesinglbase','trademark','fi','fl','Lslash','OE','Scaron','Ydieresis','Zcaron','dotlessi','lslash','oe','scaron','zcaron',None,
        'Euro','exclamdown','cent','sterling','currency','yen','brokenbar','section','dieresis','copyright','ordfeminine','guillemotleft','logicalnot',None,'registered','macron',
        'degree','plusminus','twosuperior','threesuperior','acute','mu','paragraph','periodcentered','cedilla','onesuperior','ordmasculine','guillemotright','onequarter','onehalf','threequarters','questiondown',
        'Agrave','Aacute','Acircumflex','Atilde','Adieresis','Aring','AE','Ccedilla','Egrave','Eacute','Ecircumflex','Edieresis','Igrave','Iacute','Icircumflex','Idieresis',
        'Eth','Ntilde','Ograve','Oacute','Ocircumflex','Otilde','Odieresis','multiply','Oslash','Ugrave','Uacute','Ucircumflex','Udieresis','Yacute','Thorn','germandbls',
        'agrave','aacute','acircumflex','atilde','adieresis','aring','ae','ccedilla','egrave','eacute','ecircumflex','edieresis','igrave','iacute','icircumflex','idieresis',
        'eth','ntilde','ograve','oacute','ocircumflex','otilde','odieresis','divide','oslash','ugrave','uacute','ucircumflex','udieresis','yacute','thorn','ydieresis']

    encodingVectors['/StandardEncoding'] = NONE_VEC(32) + ASCII_VEC + NONE_VEC(32) \
    +  [None,'exclamdown','cent','sterling','fraction','yen','florin','section','currency','quotesingle','quotedblleft','guillemotleft','guilsinglleft','guilsinglright','fi','fl',
        None,'endash','dagger','daggerdbl','periodcentered',None,'paragraph','bullet','quotesinglbase','quotedblbase','quotedblright','guillemotright','ellipsis','perthousand',None,'questiondown',
        None,'grave','acute','circumflex','tilde','macron','breve','dotaccent','dieresis',None,'ring','cedilla',None,'hungarumlaut','ogonek','caron',
        'emdash',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,
        None,'AE',None,'ordfeminine',None,None,None,None,'Lslash','Oslash','OE','ordmasculine',None,None,None,None,
        None,'ae',None,None,None,'dotlessi',None,None,'lslash','oslash','oe','germandbls',None,None,None,None]
    encodingVectors['/StandardEncoding'][0x27] = 'quoteright'
    encodingVectors['/StandardEncoding'][0x60] = 'quoteleft'

    encodingVectors['/WinAnsiEncoding'] = NONE_VEC(32) + ASCII_VEC \
    +  ['Euro','bullet','quotesinglbase','florin','quotedblbase','ellipsis','dagger','daggerdbl','circumflex','perthousand','Scaron','guilsinglleft','OE','bullet','Zcaron','bullet',
        'bullet','quoteleft','quoteright','quotedblleft','quotedblright','bullet','endash','emdash','tilde','trademark','scaron','guilsinglright','oe','bullet','zcaron','Ydieresis',
        'space','exclamdown','cent','sterling','currency','yen','brokenbar','section','dieresis','copyright','ordfeminine','guillemotleft','logicalnot','hyphen','registered','macron',
        'degree','plusminus','twosuperior','threesuperior','acute','mu','paragraph','periodcentered','cedilla','onesuperior','ordmasculine','guillemotright','onequarter','onehalf','threequarters','questiondown',
        'Agrave','Aacute','Acircumflex','Atilde','Adieresis','Aring','AE','Ccedilla','Egrave','Eacute','Ecircumflex','Edieresis','Igrave','Iacute','Icircumflex','Idieresis',
        'Eth','Ntilde','Ograve','Oacute','Ocircumflex','Otilde','Odieresis','multiply','Oslash','Ugrave','Uacute','Ucircumflex','Udieresis','Yacute','Thorn','germandbls',
        'agrave','aacute','acircumflex','atilde','adieresis','aring','ae','ccedilla','egrave','eacute','ecircumflex','edieresis','igrave','iacute','icircumflex','idieresis',
        'eth','ntilde','ograve','oacute','ocircumflex','otilde','odieresis','divide','oslash','ugrave','uacute','ucircumflex','udieresis','yacute','thorn','ydieresis']
    encodingVectors['/WinAnsiEncoding'][0x7f] = 'bullet'

    encodingVectors['/MacRomanEncoding'] = NONE_VEC(32) + ASCII_VEC \
    +  ['Adieresis','Aring','Ccedilla','Eacute','Ntilde','Odieresis','Udieresis','aacute','agrave','acircumflex','adieresis','atilde','aring','ccedilla','eacute','egrave',
        'ecircumflex','edieresis','iacute','igrave','icircumflex','idieresis','ntilde','oacute','ograve','ocircumflex','odieresis','otilde','uacute','ugrave','ucircumflex','udieresis',
        'dagger','degree','cent','sterling','section','bullet','paragraph','germandbls','registered','copyright','trademark','acute','dieresis',None,'AE','Oslash',
        None,'plusminus',None,None,'yen','mu',None,None,None,None,None,'ordfeminine','ordmasculine',None,'ae','oslash',
        'questiondown','exclamdown','logicalnot',None,'florin',None,None,'guillemotleft','guillemotright','ellipsis','space','Agrave','Atilde','Otilde','OE','oe',
        'endash','emdash','quotedblleft','quotedblright','quoteleft','quoteright','divide',None,'ydieresis','Ydieresis','fraction','currency','guilsinglleft','guilsinglright','fi','fl',
        'daggerdbl','periodcentered','quotesinglbase','quotedblbase','perthousand','Acircumflex','Ecircumflex','Aacute','Edieresis','Egrave','Iacute','Icircumflex','Idieresis','Igrave','Oacute','Ocircumflex',
        None,'Ograve','Uacute','Ucircumflex','Ugrave','dotlessi','circumflex','tilde','macron','breve','dotaccent','ring','cedilla','hungarumlaut','ogonek','caron']

    # PDF Ref. 1.7, page 431
    encodingVectors['/StandardRomanEncoding'] = encodingVectors['/MacRomanEncoding'].copy()
    for k,v in zip((173,176,178,179,182,183,184,185,186,189,195,197,198,215,219,240),
                    ('notequal','infinity','lessequal','greaterequal','partialdiff','summation','product','pi',
                     'integral','Omega','radical','approxequal','Delta','lozenge','Euro','apple')):
        encodingVectors['/StandardRomanEncoding'][k] = v 

    encodingVectors['/MacExpertEncoding'] = NONE_VEC(32) \
    +  ['space','exclamsmall','Hungarumlautsmall','centoldstyle','dollaroldstyle','dollarsuperior','ampersandsmall','Acutesmall','parenleftsuperior','parenrightsuperior','twodotenleader','onedotenleader','comma','hyphen','period','fraction',
        'zerooldstyle','oneoldstyle','twooldstyle','threeoldstyle','fouroldstyle','fiveoldstyle','sixoldstyle','sevenoldstyle','eightoldstyle','nineoldstyle','colon','semicolon',None,'threequartersemdash',None,'questionsmall',
        None,None,None,None,'Ethsmall',None,None,'onequarter','onehalf','threequarters','oneeighth','threeeighths','fiveeighths','seveneighths','onethird','twothirds',
        None,None,None,None,None,None,'ff','fi','fl','ffi','ffl','parenleftinferior',None,'parenrightinferior','Circumflexsmall','hypheninferior',
        'Gravesmall','Asmall','Bsmall','Csmall','Dsmall','Esmall','Fsmall','Gsmall','Hsmall','Ismall','Jsmall','Ksmall','Lsmall','Msmall','Nsmall','Osmall',
        'Psmall','Qsmall','Rsmall','Ssmall','Tsmall','Usmall','Vsmall','Wsmall','Xsmall','Ysmall','Zsmall','colonmonetary','onefitted','rupiah','Tildesmall',None,
        None,'asuperior','centsuperior',None,None,None,None,'Aacutesmall','Agravesmall','Acircumflexsmall','Adieresissmall','Atildesmall','Aringsmall','Ccedillasmall','Eacutesmall','Egravesmall',
        'Ecircumflexsmall','Edieresissmall','Iacutesmall','Igravesmall','Icircumflexsmall','Idieresissmall','Ntildesmall','Oacutesmall','Ogravesmall','Ocircumflexsmall','Odieresissmall','Otildesmall','Uacutesmall','Ugravesmall','Ucircumflexsmall','Udieresissmall',
        None,'eightsuperior','fourinferior','threeinferior','sixinferior','eightinferior','seveninferior','Scaronsmall',None,'centinferior','twoinferior',None,'Dieresissmall',None,'Caronsmall','osuperior',
        'fiveinferior',None,'commainferior','periodinferior','Yacutesmall',None,'dollarinferior',None,None,'Thornsmall',None,'nineinferior','zeroinferior','Zcaronsmall','AEsmall','Oslashsmall',
        'questiondownsmall','oneinferior','Lslashsmall',None,None,None,None,None,None,'Cedillasmall',None,None,None,None,None,'OEsmall',
        'figuredash','hyphensuperior',None,None,None,None,'exclamdownsmall',None,'Ydieresissmall',None,'onesuperior','twosuperior','threesuperior','foursuperior','fivesuperior','sixsuperior',
        'sevensuperior','ninesuperior','zerosuperior',None,'esuperior','rsuperior','tsuperior',None,None,'isuperior','ssuperior','dsuperior',None,None,None,None,
        None,'lsuperior','Ogoneksmall','Brevesmall','Macronsmall','bsuperior','nsuperior','msuperior','commasuperior','periodsuperior','Dotaccentsmall','Ringsmall',None,None,None,None]

    encodingVectors['/SymbolEncoding'] = NONE_VEC(32) \
    +  ['space','exclam','universal','numbersign','existential','percent','ampersand','suchthat','parenleft','parenright','asteriskmath','plus','comma','minus','period','slash',
        'zero','one','two','three','four','five','six','seven','eight','nine','colon','semicolon','less','equal','greater','question',
        'congruent','Alpha','Beta','Chi','Delta','Epsilon','Phi','Gamma','Eta','Iota','theta1','Kappa','Lambda','Mu','Nu','Omicron',
        'Pi','Theta','Rho','Sigma','Tau','Upsilon','sigma1','Omega','Xi','Psi','Zeta','bracketleft','therefore','bracketright','perpendicular','underscore',
        'radicalex','alpha','beta','chi','delta','epsilon','phi','gamma','eta','iota','phi1','kappa','lambda','mu','nu','omicron',
        'pi','theta','rho','sigma','tau','upsilon','omega1','omega','xi','psi','zeta','braceleft','bar','braceright','similar',None] \
    + NONE_VEC(32) \
    +  ['Euro','Upsilon1','minute','lessequal','fraction','infinity','florin','club','diamond','heart','spade','arrowboth','arrowleft','arrowup','arrowright','arrowdown',
        'degree','plusminus','second','greaterequal','multiply','proportional','partialdiff','bullet','divide','notequal','equivalence','approxequal','ellipsis','arrowvertex','arrowhorizex','carriagereturn',
        'aleph','Ifraktur','Rfraktur','weierstrass','circlemultiply','circleplus','emptyset','intersection','union','propersuperset','reflexsuperset','notsubset','propersubset','reflexsubset','element','notelement',
        'angle','gradient','registerserif','copyrightserif','trademarkserif','product','radical','dotmath','logicalnot','logicaland','logicalor','arrowdblboth','arrowdblleft','arrowdblup','arrowdblright','arrowdbldown',
        'lozenge','angleleft','registersans','copyrightsans','trademarksans','summation','parenlefttp','parenleftex','parenleftbt','bracketlefttp','bracketleftex','bracketleftbt','bracelefttp','braceleftmid','braceleftbt','braceex',
        None,'angleright','integral','integraltp','integralex','integralbt','parenrighttp','parenrightex','parenrightbt','bracketrighttp','bracketrightex','bracketrightbt','bracerighttp','bracerightmid','bracerightbt',None]

    encodingVectors['/ZapfDingbatsEncoding'] = NONE_VEC(32) \
    +  ['space','a1','a2','a202','a3','a4','a5','a119','a118','a117','a11','a12','a13','a14','a15','a16',
        'a105','a17','a18','a19','a20','a21','a22','a23','a24','a25','a26','a27','a28','a6','a7','a8',
        'a9','a10','a29','a30','a31','a32','a33','a34','a35','a36','a37','a38','a39','a40','a41','a42',
        'a43','a44','a45','a46','a47','a48','a49','a50','a51','a52','a53','a54','a55','a56','a57','a58',
        'a59','a60','a61','a62','a63','a64','a65','a66','a67','a68','a69','a70','a71','a72','a73','a74',
        'a203','a75','a204','a76','a77','a78','a79','a81','a82','a83','a84','a97','a98','a99','a100',None,
        'a89','a90','a93','a94','a91','a92','a205','a85','a206','a86','a87','a88','a95','a96',None,None,
        None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,
        None,'a101','a102','a103','a104','a106','a107','a108','a112','a111','a110','a109','a120','a121','a122','a123',
        'a124','a125','a126','a127','a128','a129','a130','a131','a132','a133','a134','a135','a136','a137','a138','a139',
        'a140','a141','a142','a143','a144','a145','a146','a147','a148','a149','a150','a151','a152','a153','a154','a155',
        'a156','a157','a158','a159','a160','a161','a163','a164','a196','a165','a192','a166','a167','a168','a169','a170',
        'a171','a172','a173','a162','a174','a175','a176','a177','a178','a179','a193','a180','a199','a181','a200','a182',
        None,'a201','a183','a184','a197','a185','a194','a198','a186','a195','a187','a188','a189','a190','a191',None]

# =========================================================================== class PdfFontGlyphMap

class PdfFontGlyphMap:
    '''
    The class provides fonts ToUnicode CMaps inference/reencoding functions that are
    based on the Adobe Glyph Lists
    (maps from the more-or-less standardized -- by Adobe -- font glyph names to Unicode points,
    see: https://github.com/adobe-type-tools/agl-aglfn) and the heuristics based on the statistical analysis
    of the so-called 'composite' (see the help of the CompositeNames class for more info) glyph names
    based on the entire set of document's fonts, the results of which are set in self.composites_global.
    '''

    # Suffix/prefix types; higher values mean higher specificity
    DEX=1 # Digital or hex, like '12'
    HEX=2 # Hex, but not decimal, like '1F'

    def __init__(self,
                    loadAdobeGlyphList:bool = True,
                    fontsForTraining:dict = {},
                    knownPrefixes = {}
                ):
        '''Initializes PdfFontGlyphMap by doing the following:
        
        * sets `self.adobeMap = MyGlypMap (superset of AdobeGlyphMap with extra mappings) if loadAdobeGlyphList else {}`.
        * sets `self.glyphMap` to a map that translates composite glyph names to Unicode values by
        training the composite_glyphname_to_cc() algo on glyph names from the `fontsForTraining` dict;
        this also sets `self.knownPrefixes` which is a map from prefixes (strings)
        to the corresponding suffix type (see class CompositeNames), for example: {'uni':'hex', 'c':'dex', ...};
        this map can then be used by the `composite_gname_to_cc()` translation algo.
        * `self.knownPrefixes` is updated based on `knownPrefixes`.

        For more info on composite names, please see the help for `composite_glyphname_to_cc()`.
        '''

        # A map for glyph names to Unicode chars based on the Adobe Glyph List (+other lists)
        self.adobeMap = MyGlyphMap if loadAdobeGlyphList else {}

        # Map from composite glyph names to Unicode chars
        # Produced as a result of training by calling composite_glyphname_to_unicode(gname) repeatedly
        self.glyphMap = {}

        # A map from prefixes to suffix types for composite glyph names
        self.prefix_types = {}

        # train the composite_glyphname_to_cc() algo on glyph names from the fonts list
        fonts = [f for f in fontsForTraining.values() if f.font.Subtype != '/Type0']
        for font in fonts:
            strippedGlyphNames = [PdfFontGlyphMap.strip_dot_endings(g) for g in font.cc2g.values()]
            for gname in strippedGlyphNames:
                if gname in self.adobeMap: continue
                self.composite_gname_to_cc(gname)

        # add prefix to prefix type mapping from the knownPrefixes argument
        for prefix,t in knownPrefixes.items():
            self.prefix_types[prefix] = PdfFontGlyphMap.HEX if t.lower() == 'hex' else PdfFontGlyphMap.DEX

    # --------------------------------------------------------------------------- composite_gname_to_cc()

    def composite_gname_to_cc(self, gname:str):
        '''
        Maps composite glyph names of the form `prefix + suffix` form, where suffix is a decimal/hex number,
        to character codes by using `prefix` to infer whether `suffix` is decimal/hex based on
        past encounters with such composite glyph names.
        
        Examples of composite names include `c13`, `glyph10H`, etc.

        The usage scenario is to first run this function through all available glyph names
        to train the algorithm, and then call it on any particular glyph name to get results.
        '''
        suffix_type = lambda suffix: self.DEX if all(c in string.digits for c in suffix) else self.HEX

        # Abb is a double-struck A, not a composite glyph name!
        if re.match(r'[a-zA-Z]b{2,}', gname): return None

        gname_marked = re.sub(r'^([a-zA-Z]|#|FLW|uni|cid|Char|char|glyph|MT|.*\.g)([0-9a-fA-F]{2,}|[0-9])$',r'\1|||\2',gname)
        gname_split = re.split(r'\|\|\|',gname_marked)
        prefix,suffix = gname_split if len(gname_split) == 2 else (None,None)
        if prefix == None: return None

        suffix_t = suffix_type(suffix) if prefix != 'uni' else self.HEX
        if prefix not in self.prefix_types or suffix_t > self.prefix_types[prefix]:
            self.prefix_types[prefix] = suffix_t

        return int(suffix,16) if self.prefix_types[prefix] == self.HEX else int(suffix)

    # --------------------------------------------------------------------------- decode_gname()

    def decode_gname(self,
                         gname,
                         stdGlyphMap:dict[str,str] = {},
                         isType3:bool = False,
                         baseInvMap:dict[PdfName,str] = {},
                         explicitMap:dict[PdfName,str] = {},
                         mapComposites:bool = True,
                         mapSemicolons:bool = True):
        '''
        Convert a glyph name either to a character code (int) or a unicode value (str), depending on the glyph name.
        '''
        assert gname[0] == '/'

        # We don't map .notdef (i.e., the if below never fulfills), but if we would this would be the mapping:
        if gname == '/.notdef': return chr(0xFFFD) # Unicode: Replacement Char

        result = None

        if gname in baseInvMap:
            return ord(baseInvMap[gname])

        if gname in explicitMap:
            return explicitMap[gname]
        
        if isType3:
            result = PdfFontGlyphMap.type3_gname_to_cc(gname=gname,
                                                        mapSemicolons=mapSemicolons)
            
        if result is None:

            # This is how symbol fonts should be encoded according to the PDF Ref.
            if len(gname) == 8 and gname[:5] == '/uniF' and gname[5] in ['0', '1', '2']:

                try: result = int(gname[4:],16)
                except: result = None

        gname = PdfFontGlyphMap.strip_dot_endings(gname)

        if result is None:

            # When glyphs encode the corresponding Unicode points in their names
            if gname[:4] == '/uni':

                # try: result = chr(int(gname[4:],16))
                try: result = PdfTextString.UTF16BE_to_Unicode(gname[4:])
                except: result = None

        if result is None:

            # When glyphs encode the corresponding Unicode points in their names
            if gname[:2] == '/u':

                try: result = chr(int(gname[2:],16))
                except: result = None

        if result is None:

            g = PdfFontGlyphMap.strip_prefixes(gname)

            # Order of checking is important
            result = self.adobeMap.get(g) or self.glyphMap.get(g) or stdGlyphMap.get(g) \
                    or self.adobeMap.get(gname)

        if result == None and mapComposites:

            composite = gname[1:]
            result = int(composite) if all(c in string.digits for c in composite) \
                else ord(composite) if len(composite) == 1 \
                else self.composite_gname_to_cc(composite)
            
            if isinstance(result, int) and result > 0x110000:
                result = None

        return result

    # --------------------------------------------------------------------------- gname_to_cc_type3()

    @staticmethod
    def type3_gname_to_cc(gname:PdfName,
                            mapSemicolons = True):
        '''
        Convert a Type3 font glyphname to a character code (int), or None if conversion fails.
        '''

        t3map1 = {c:i for i,c in enumerate('ABCDEFGH')}
        t3map2 = {c:i for i,c in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
        # semicolons = {';':';', ';;':chr(0), ';;;;':chr(0), ';;;;;;;;':';'}
        semicolons = {';':0, ';;':ord(';'), ';;;;':0, ';;;;;;;;':ord(';')}

        if gname == '/BnZr':
            return 0 # This is how chr(0) is sometimes denoted in Type3 fonts
        elif mapSemicolons and gname[1:] in semicolons:
            cc = semicolons[gname[1:]]
            warn(f'mapping {gname} --> {cc:04X}')
        elif len(gname) == 3:
            try: return t3map1[gname[1]] * 36 + t3map2[gname[2]]
            except: return None
        elif len(gname) == 4 and gname[1] == '#':
            try: return int(gname[2:],16)
            except: return None
        else:
            return None

    # --------------------------------------------------------------------------- strip_dot_endings()

    @staticmethod
    def strip_dot_endings(s:str):
        '''
        Strips ._, .sc & .cap endings from a string.
        Useful to match variants (small caps etc) of glyph names in glyph lists
        '''
        assert s[0] == '/'

        s1 = s.rstrip(';')
        if len(s1) != len(s) - 1 or len(s1) == 1:
            s1 = s

        s1 = re.sub(r'(\.(_|sc|cap|alt[0-9]*|vsize[0-9]*|hsize[0-9]*|disp|big|small|sm|upsm|short|ts1|lf|tf|swash|endl))+$','', s1)
 
        # Unicode points of Arabic letter forms in Times New Roman are mapped to logical Unicode points
        s1 = re.sub(r'(\.(init|medi|fina|isol|morocco|finamorocco|urdu))+$','', s1)
        return s1 if len(s1)>1 else s

    # --------------------------------------------------------------------------- strip_prefixes()

    @staticmethod
    def strip_prefixes(s:str):
        '''Strips \\rm and the like from the beginning of the glyph name
        '''
        s1 = re.sub(r'(\\rm|\\mathchardef)', '', s)
        return s1 if len(s1)>1 else s


# =========================================================================== class PdfFontGlyphMapStandards

class PdfFontGlyphMapStandards:

    def get_glyphName2unicodeMap(encodingName:str):
        '''
        Returns the appropriate glyph2unicode map chosen from the standard unicode maps for standard PDF encodings.
        For unknown encodings, returns a union of the standard unicode maps except the map for Zapf's Dingbats.
        
        The function does not just return a union of all the standard unicode maps because:
        a) some of the glyph names in the /ZapfDingbatsEncoding may be used by non-core14 fonts, in which case
        these glyph names may denote different glyphs; b) the '/mu' in the /WinAnsiEncoding stands for the 'micro'
        character U+00B5 (like, for example, the one used in writing 'micrometer' as 'mu m'), while the '/mu'
        in the '/SymbolEncoding' stands for the greek lower-case 'mu' U+03BC, the one also used in mathematical texts.

        For mappings beyond these standard maps, please use Adobe's glyphlist.txt and similar map lists.
        '''
        encNameToEncSet = {
            '/WinAnsiEncoding':'LatinSet',
            '/MacRomanEncoding':'LatinSet',
            '/StandardEncoding':'LatinSet',
            '/PDFDocEncoding':'LatinSet',
            '/SymbolEncoding':'SymbolSet',
            '/ZapfDingbatsEncoding':'ZapfDingbatsSet',
            '/MacExpertEncoding':'MacExpertSet'
        }
        glyphName2unicodeMapUnion = PdfFontGlyphMapStandards.glyphName2unicodeMaps['LatinSet'] \
                    | PdfFontGlyphMapStandards.glyphName2unicodeMaps['MacExpertSet'] \
                    | PdfFontGlyphMapStandards.glyphName2unicodeMaps['SymbolSet']
        glyphname2unicode =  PdfFontGlyphMapStandards.glyphName2unicodeMaps.get(encNameToEncSet.get(encodingName)) \
                                or glyphName2unicodeMapUnion
        glyphname2unicode = {PdfName(name):unicode for name,unicode in glyphname2unicode.items()}
        return glyphname2unicode

    # --------------------------------------------------------------------------- glyphName2unicodeMaps

    glyphName2unicodeMaps = {}

    glyphName2unicodeMaps['LatinSet'] = {
        'A': 'A', 'AE': 'Æ', 'Aacute': 'Á', 'Acircumflex': 'Â', 'Adieresis': 'Ä', 'Agrave': 'À', 'Aring': 'Å',
        'Atilde': 'Ã', 'B': 'B', 'C': 'C', 'Ccedilla': 'Ç', 'D': 'D', 'E': 'E', 'Eacute': 'É', 'Ecircumflex': 'Ê',
        'Edieresis': 'Ë', 'Egrave': 'È', 'Eth': 'Ð', 'Euro': '€', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I',
        'Iacute': 'Í', 'Icircumflex': 'Î', 'Idieresis': 'Ï', 'Igrave': 'Ì', 'J': 'J', 'K': 'K', 'L': 'L',
        'Lslash': 'Ł', 'M': 'M', 'N': 'N', 'Ntilde': 'Ñ', 'O': 'O', 'OE': 'Œ', 'Oacute': 'Ó', 'Ocircumflex': 'Ô',
        'Odieresis': 'Ö', 'Ograve': 'Ò', 'Oslash': 'Ø', 'Otilde': 'Õ', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S',
        'Scaron': 'Š', 'T': 'T', 'Thorn': 'Þ', 'U': 'U', 'Uacute': 'Ú', 'Ucircumflex': 'Û', 'Udieresis': 'Ü',
        'Ugrave': 'Ù', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Yacute': 'Ý', 'Ydieresis': 'Ÿ', 'Z': 'Z',
        'Zcaron': 'Ž', 'a': 'a', 'aacute': 'á', 'acircumflex': 'â', 'acute': '´', 'adieresis': 'ä', 'ae': 'æ',
        'agrave': 'à', 'ampersand': '&', 'aring': 'å', 'asciicircum': '^', 'asciitilde': '~', 'asterisk': '*',
        'at': '@', 'atilde': 'ã', 'b': 'b', 'backslash': '\\', 'bar': '|', 'braceleft': '{', 'braceright': '}',
        'bracketleft': '[', 'bracketright': ']', 'breve': '˘', 'brokenbar': '¦', 'bullet': '•', 'c': 'c',
        'caron': 'ˇ', 'ccedilla': 'ç', 'cedilla': '¸', 'cent': '¢', 'circumflex': 'ˆ', 'colon': ':', 'comma': ',',
        'copyright': '©', 'currency': '¤', 'd': 'd', 'dagger': '†', 'daggerdbl': '‡', 'degree': '°', 'dieresis': '¨',
        'divide': '÷', 'dollar': '$', 'dotaccent': '˙', 'dotlessi': 'ı', 'e': 'e', 'eacute': 'é', 'ecircumflex': 'ê',
        'edieresis': 'ë', 'egrave': 'è', 'eight': '8', 'ellipsis': '…', 'emdash': '—', 'endash': '–', 'equal': '=',
        'eth': 'ð', 'exclam': '!', 'exclamdown': '¡', 'f': 'f', 'fi': 'ﬁ', 'five': '5', 'fl': 'ﬂ', 'florin': 'ƒ',
        'four': '4', 'fraction': '⁄', 'g': 'g', 'germandbls': 'ß', 'grave': '`', 'greater': '>', 'guillemotleft': '«',
        'guillemotright': '»', 'guilsinglleft': '‹', 'guilsinglright': '›', 'h': 'h', 'hungarumlaut': '˝',
        'hyphen': '-', 'i': 'i', 'iacute': 'í', 'icircumflex': 'î', 'idieresis': 'ï', 'igrave': 'ì', 'j': 'j',
        'k': 'k', 'l': 'l', 'less': '<', 'logicalnot': '¬', 'lslash': 'ł', 'm': 'm', 'macron': '¯', 'minus': '−',
        'mu': 'µ', 'multiply': '×', 'n': 'n', 'nine': '9', 'ntilde': 'ñ', 'numbersign': '#', 'o': 'o', 'oacute': 'ó',
        'ocircumflex': 'ô', 'odieresis': 'ö', 'oe': 'œ', 'ogonek': '˛', 'ograve': 'ò', 'one': '1', 'onehalf': '½',
        'onequarter': '¼', 'onesuperior': '¹', 'ordfeminine': 'ª', 'ordmasculine': 'º', 'oslash': 'ø', 'otilde': 'õ',
        'p': 'p', 'paragraph': '¶', 'parenleft': '(', 'parenright': ')', 'percent': '%', 'period': '.',
        'periodcentered': '·', 'perthousand': '‰', 'plus': '+', 'plusminus': '±', 'q': 'q', 'question': '?',
        'questiondown': '¿', 'quotedbl': '"', 'quotedblbase': '„', 'quotedblleft': '“', 'quotedblright': '”',
        'quoteleft': '‘', 'quoteright': '’', 'quotesinglbase': '‚', 'quotesingle': "'", 'r': 'r', 'registered': '®',
        'ring': '˚', 's': 's', 'scaron': 'š', 'section': '§', 'semicolon': ';', 'seven': '7', 'six': '6', 'slash': '/',
        'space': ' ', 'sterling': '£', 't': 't', 'thorn': 'þ', 'three': '3', 'threequarters': '¾', 'threesuperior': '³',
        'tilde': '˜', 'trademark': '™', 'two': '2', 'twosuperior': '²', 'u': 'u', 'uacute': 'ú', 'ucircumflex': 'û',
        'udieresis': 'ü', 'ugrave': 'ù', 'underscore': '_', 'v': 'v', 'w': 'w', 'x': 'x', 'y': 'y', 'yacute': 'ý',
        'ydieresis': 'ÿ', 'yen': '¥', 'z': 'z', 'zcaron': 'ž', 'zero': '0'
    }

    # The phi/phi1 codes were swapped in Unicode 3.0 (Unicode Technical Report #25)

    glyphName2unicodeMaps['SymbolSet'] = {
        'Alpha': 'Α', 'Beta': 'Β', 'Chi': 'Χ', 'Delta': 'Δ', 'Epsilon': 'Ε', 'Eta': 'Η',
        'Euro': '€', 'Gamma': 'Γ', 'Ifraktur': 'ℑ', 'Iota': 'Ι', 'Kappa': 'Κ', 'Lambda': 'Λ', 'Mu': 'Μ', 'Nu': 'Ν',
        'Omega': 'Ω', 'Omicron': 'Ο', 'Phi': 'Φ', 'Pi': 'Π', 'Psi': 'Ψ', 'Rfraktur': 'ℜ', 'Rho': 'Ρ', 'Sigma': 'Σ',
        'Tau': 'Τ', 'Theta': 'Θ', 'Upsilon': 'Υ', 'Upsilon1': 'ϒ', 'Xi': 'Ξ', 'Zeta': 'Ζ', 'aleph': 'ℵ', 'alpha': 'α',
        'ampersand': '&', 'angle': '∠', 'angleleft': '〈', 'angleright': '〉', 'apple': '', 'approxequal': '≈',
        'arrowboth': '↔', 'arrowdblboth': '⇔', 'arrowdbldown': '⇓', 'arrowdblleft': '⇐', 'arrowdblright': '⇒',
        'arrowdblup': '⇑', 'arrowdown': '↓', 'arrowhorizex': '⎯', 'arrowleft': '←', 'arrowright': '→', 'arrowup': '↑',
        'arrowvertex': '⏐', 'asteriskmath': '∗', 'bar': '|', 'beta': 'β', 'braceex': '⎪', 'braceleft': '{',
        'braceleftbt': '⎩', 'braceleftmid': '⎨', 'bracelefttp': '⎧', 'braceright': '}', 'bracerightbt': '⎭',
        'bracerightmid': '⎬', 'bracerighttp': '⎫', 'bracketleft': '[', 'bracketleftbt': '⎣', 'bracketleftex': '⎢',
        'bracketlefttp': '⎡', 'bracketright': ']', 'bracketrightbt': '⎦', 'bracketrightex': '⎥', 'bracketrighttp': '⎤',
        'bullet': '•', 'carriagereturn': '↵', 'chi': 'χ', 'circlemultiply': '⊗', 'circleplus': '⊕', 'club': '♣',
        'colon': ':', 'comma': ',', 'congruent': '≅', 'copyrightsans': '©', 'copyrightserif': '©', 'degree': '°',
        'delta': 'δ', 'diamond': '♦', 'divide': '÷', 'dotmath': '⋅', 'eight': '8', 'element': '∈', 'ellipsis': '…',
        'emptyset': '∅', 'epsilon': 'ε', 'equal': '=', 'equivalence': '≡', 'eta': 'η', 'exclam': '!', 'existential':
        '∃', 'five': '5', 'florin': 'ƒ', 'four': '4', 'fraction': '⁄', 'gamma': 'γ', 'gradient': '∇', 'greater': '>',
        'greaterequal': '≥', 'heart': '♥', 'infinity': '∞', 'integral': '∫', 'integralbt': '⌡', 'integralex': '⎮',
        'integraltp': '⌠', 'intersection': '∩', 'iota': 'ι', 'kappa': 'κ', 'lambda': 'λ', 'less': '<',
        'lessequal': '≤', 'logicaland': '∧', 'logicalnot': '¬', 'logicalor': '∨', 'lozenge': '◊', 'minus': '−',
        'minute': '′', 'mu': 'μ', 'multiply': '×', 'nine': '9', 'notelement': '∉', 'notequal': '≠', 'notsubset': '⊄',
        'nu': 'ν', 'numbersign': '#', 'omega': 'ω', 'omega1': 'ϖ', 'omicron': 'ο', 'one': '1', 'parenleft': '(',
        'parenleftbt': '⎝', 'parenleftex': '⎜', 'parenlefttp': '⎛', 'parenright': ')', 'parenrightbt': '⎠',
        'parenrightex': '⎟', 'parenrighttp': '⎞', 'partialdiff': '∂', 'percent': '%', 'period': '.',
        'perpendicular': '⊥', 'phi': 'ϕ', 'phi1': 'φ', 'pi': 'π', 'plus': '+', 'plusminus': '±', 'product': '∏',
        'propersubset': '⊂', 'propersuperset': '⊃', 'proportional': '∝', 'psi': 'ψ', 'question': '?', 'radical': '√',
        'radicalex': '‾', 'reflexsubset': '⊆', 'reflexsuperset': '⊇', 'registersans': '®', 'registerserif': '®',
        'rho': 'ρ', 'second': '″', 'semicolon': ';', 'seven': '7', 'sigma': 'σ', 'sigma1': 'ς', 'similar': '∼',
        'six': '6', 'slash': '/', 'space': ' ', 'spade': '♠', 'suchthat': '∋', 'summation': '∑', 'tau': 'τ',
        'therefore': '∴', 'theta': 'θ', 'theta1': 'ϑ', 'three': '3', 'trademarksans': '™', 'trademarkserif': '™',
        'two': '2', 'underscore': '_', 'union': '∪', 'universal': '∀', 'upsilon': 'υ', 'weierstrass': '℘',
        'xi': 'ξ', 'zero': '0', 'zeta': 'ζ'}

    glyphName2unicodeMaps['ZapfDingbatsSet'] = {
        'a1': '✁', 'a2': '✂', 'a202': '✃', 'a3': '✄', 'a4': '☎', 'a5': '✆', 'a119': '✇', 'a118': '✈',
        'a117': '✉', 'a11': '☛', 'a12': '☞', 'a13': '✌', 'a14': '✍', 'a15': '✎', 'a16': '✏', 'a105': '✐', 'a17': '✑',
        'a18': '✒', 'a19': '✓', 'a20': '✔', 'a21': '✕', 'a22': '✖', 'a23': '✗', 'a24': '✘', 'a25': '✙', 'a26': '✚',
        'a27': '✛', 'a28': '✜', 'a6': '✝', 'a7': '✞', 'a8': '✟', 'a9': '✠', 'a10': '✡', 'a29': '✢', 'a30': '✣',
        'a31': '✤', 'a32': '✥', 'a33': '✦', 'a34': '✧', 'a35': '★', 'a36': '✩', 'a37': '✪', 'a38': '✫', 'a39': '✬',
        'a40': '✭', 'a41': '✮', 'a42': '✯', 'a43': '✰', 'a44': '✱', 'a45': '✲', 'a46': '✳', 'a47': '✴', 'a48': '✵',
        'a49': '✶', 'a50': '✷', 'a51': '✸', 'a52': '✹', 'a53': '✺', 'a54': '✻', 'a55': '✼', 'a56': '✽', 'a57': '✾',
        'a58': '✿', 'a59': '❀', 'a60': '❁', 'a61': '❂', 'a62': '❃', 'a63': '❄', 'a64': '❅', 'a65': '❆', 'a66': '❇',
        'a67': '❈', 'a68': '❉', 'a69': '❊', 'a70': '❋', 'a71': '●', 'a72': '❍', 'a73': '■', 'a74': '❏', 'a203': '❐',
        'a75': '❑', 'a204': '❒', 'a76': '▲', 'a77': '▼', 'a78': '◆', 'a79': '❖', 'a81': '◗', 'a82': '❘', 'a83': '❙',
        'a84': '❚', 'a97': '❛', 'a98': '❜', 'a99': '❝', 'a100': '❞', 'a89': '❨', 'a90': '❩', 'a93': '❪', 'a94': '❫',
        'a91': '❬', 'a92': '❭', 'a205': '❮', 'a85': '❯', 'a206': '❰', 'a86': '❱', 'a87': '❲', 'a88': '❳', 'a95': '❴',
        'a96': '❵', 'a101': '❡', 'a102': '❢', 'a103': '❣', 'a104': '❤', 'a106': '❥', 'a107': '❦', 'a108': '❧',
        'a112': '♣', 'a111': '♦', 'a110': '♥', 'a109': '♠', 'a120': '①', 'a121': '②', 'a122': '③', 'a123': '④',
        'a124': '⑤', 'a125': '⑥', 'a126': '⑦', 'a127': '⑧', 'a128': '⑨', 'a129': '⑩', 'a130': '❶', 'a131': '❷',
        'a132': '❸', 'a133': '❹', 'a134': '❺', 'a135': '❻', 'a136': '❼', 'a137': '❽', 'a138': '❾', 'a139': '❿',
        'a140': '➀', 'a141': '➁', 'a142': '➂', 'a143': '➃', 'a144': '➄', 'a145': '➅', 'a146': '➆', 'a147': '➇',
        'a148': '➈', 'a149': '➉', 'a150': '➊', 'a151': '➋', 'a152': '➌', 'a153': '➍', 'a154': '➎', 'a155': '➏',
        'a156': '➐', 'a157': '➑', 'a158': '➒', 'a159': '➓', 'a160': '➔', 'a161': '→', 'a163': '↔', 'a164': '↕',
        'a196': '➘', 'a165': '➙', 'a192': '➚', 'a166': '➛', 'a167': '➜', 'a168': '➝', 'a169': '➞', 'a170': '➟',
        'a171': '➠', 'a172': '➡', 'a173': '➢', 'a162': '➣', 'a174': '➤', 'a175': '➥', 'a176': '➦', 'a177': '➧',
        'a178': '➨', 'a179': '➩', 'a193': '➪', 'a180': '➫', 'a199': '➬', 'a181': '➭', 'a200': '➮', 'a182': '➯',
        'a201': '➱', 'a183': '➲', 'a184': '➳', 'a197': '➴', 'a185': '➵', 'a194': '➶', 'a198': '➷', 'a186': '➸',
        'a195': '➹', 'a187': '➺', 'a188': '➻', 'a189': '➼', 'a190': '➽', 'a191': '➾'
    }

    glyphName2unicodeMaps['MacExpertSet'] = {
        'space': ' ', 'exclamsmall': '﹗', 'Hungarumlautsmall': '˝', 'centoldstyle': '¢', 'dollaroldstyle': '$',
        'dollarsuperior': '$', 'ampersandsmall': '﹠', 'Acutesmall': '´', 'parenleftsuperior': '⁽',
        'parenrightsuperior': '⁾', 'twodotenleader': '‥', 'onedotenleader': '․', 'comma': ',', 'hyphen': '-',
        'period': '.', 'fraction': '⁄', 'zerooldstyle': '0', 'oneoldstyle': '1', 'twooldstyle': '2',
        'threeoldstyle': '3', 'fouroldstyle': '4', 'fiveoldstyle': '5', 'sixoldstyle': '6', 'sevenoldstyle': '7',
        'eightoldstyle': '8', 'nineoldstyle': '9', 'colon': ':', 'semicolon': ';', 'threequartersemdash': '—',
        'questionsmall': '﹖', 'Ethsmall': 'ᴆ', 'onequarter': '¼', 'onehalf': '½', 'threequarters': '¾',
        'oneeighth': '⅛', 'threeeighths': '⅜', 'fiveeighths': '⅝', 'seveneighths': '⅞', 'onethird': '⅓',
        'twothirds': '⅔', 'ff': 'ﬀ', 'fi': 'ﬁ', 'fl': 'ﬂ', 'ffi': 'ﬃ', 'ffl': 'ﬄ', 'parenleftinferior': '₍',
        'parenrightinferior': '₎', 'Circumflexsmall': 'ˆ', 'hypheninferior': '-', 'Gravesmall': '`', 'Asmall': 'ᴀ',
        'Bsmall': 'ʙ', 'Csmall': 'ᴄ', 'Dsmall': 'ᴅ', 'Esmall': 'ᴇ', 'Fsmall': 'ꜰ', 'Gsmall': 'ɢ', 'Hsmall': 'ʜ',
        'Ismall': 'ɪ', 'Jsmall': 'ᴊ', 'Ksmall': 'ᴋ', 'Lsmall': 'ʟ', 'Msmall': 'ᴍ', 'Nsmall': 'ɴ', 'Osmall': 'ᴏ',
        'Psmall': 'ᴘ', 'Qsmall': chr(0xA7Af), 'Rsmall': 'ʀ', 'Ssmall': 'ꜱ', 'Tsmall': 'ᴛ', 'Usmall': 'ᴜ', 'Vsmall': 'ᴠ',
        'Wsmall': 'ᴡ', 'Xsmall': 'x', 'Ysmall': 'ʏ', 'Zsmall': 'ᴢ', 'colonmonetary': '₡', 'onefitted': '1',
        'rupiah': '₨', 'Tildesmall': '˜', 'asuperior': 'ᵃ', 'centsuperior': '¢', 'Aacutesmall': 'Á',
        'Agravesmall': 'À', 'Acircumflexsmall': 'Â', 'Adieresissmall': 'Ä', 'Atildesmall': 'Ã', 'Aringsmall': 'Å',
        'Ccedillasmall': 'Ç', 'Eacutesmall': 'É', 'Egravesmall': 'È', 'Ecircumflexsmall': 'Ê', 'Edieresissmall': 'Ë',
        'Iacutesmall': 'Í', 'Igravesmall': 'Ì', 'Icircumflexsmall': 'Î', 'Idieresissmall': 'Ï', 'Ntildesmall': 'Ñ',
        'Oacutesmall': 'Ó', 'Ogravesmall': 'Ò', 'Ocircumflexsmall': 'Ô', 'Odieresissmall': 'Ö', 'Otildesmall': 'Õ',
        'Uacutesmall': 'Ú', 'Ugravesmall': 'Ù', 'Ucircumflexsmall': 'Û', 'Udieresissmall': 'Ü', 'eightsuperior': '⁸',
        'fourinferior': '₄', 'threeinferior': '₃', 'sixinferior': '₆', 'eightinferior': '₈', 'seveninferior': '₇',
        'Scaronsmall': 'Š', 'centinferior': '¢', 'twoinferior': '₂', 'Dieresissmall': '¨', 'Caronsmall': 'ˇ',
        'osuperior': 'ᵒ', 'fiveinferior': '₅', 'commainferior': ',', 'periodinferior': '.', 'Yacutesmall': 'Ý',
        'dollarinferior': '$', 'Thornsmall': 'Þ', 'nineinferior': '₉', 'zeroinferior': '₀', 'Zcaronsmall': 'Ž',
        'AEsmall': 'Æ', 'Oslashsmall': 'Ø', 'questiondownsmall': '¿', 'oneinferior': '₁', 'Lslashsmall': 'ᴌ',
        'Cedillasmall': '¸', 'OEsmall': 'Œ', 'figuredash': '‒', 'hyphensuperior': '-', 'exclamdownsmall': '¡',
        'Ydieresissmall': 'Ÿ', 'onesuperior': '¹', 'twosuperior': '²', 'threesuperior': '³', 'foursuperior': '⁴',
        'fivesuperior': '⁵', 'sixsuperior': '⁶', 'sevensuperior': '⁷', 'ninesuperior': '⁹', 'zerosuperior': '⁰',
        'esuperior': 'ᵉ', 'rsuperior': 'ʳ', 'tsuperior': 'ᵗ', 'isuperior': 'ⁱ', 'ssuperior': 'ˢ', 'dsuperior': 'ᵈ',
        'lsuperior': 'ˡ', 'Ogoneksmall': '˛', 'Brevesmall': '˘', 'Macronsmall': '¯', 'bsuperior': 'ᵇ',
        'nsuperior': 'ⁿ', 'msuperior': 'ᵐ', 'commasuperior': ',', 'periodsuperior': '.', 'Dotaccentsmall': '˙',
        'Ringsmall': '˚'
    }

# =========================================================================== class PdfFontCMap

class PdfFontCMap:

    PUP15 = "\U000F0000" # Start of PUP (15): https://en.wikipedia.org/wiki/Private_Use_Areas
    SURROGATES_START = "\uD800" # Maximum unicode point in the Basic Multilingual Plane (BMP) == U+D7FF
    SURROGATES_END = "\uDFFF"

    # --------------------------------------------------------------------------- __init__()

    def __init__(self, identity=False, toUnicodeDict=None, htfFilePath=None, bfrFilePath=None, diffBlockList=None):
        '''Implementation of the PDF ToUnicode CMap and the associated text conversion functions'''
        self.cc2unicode = {} # a map from cc (char) to unicode (str of len 1-3 <- ligatures)
        self.unicode2cc = {} # a reverse map from unicode to cc

        if identity == True: self.set_to_identity_map()
        if htfFilePath != None: self.read_htf_file(htfFilePath)
        if bfrFilePath != None: self.read_bfr_file(bfrFilePath)
        if diffBlockList != None: self.read_diff_block_list(diffBlockList)
        if toUnicodeDict != None: self.read_to_unicode_dict(toUnicodeDict)

    # --------------------------------------------------------------------------- set_to_identity_map()

    def set_to_identity_map(self):
        '''Sets self.cc2unicode equal to the identity map: cc -> cc for cc in range(self.UNICODE_BMP_MAX+1)
        '''
        self.cc2unicode = {}
        sur_start, sur_end = ord(self.SURROGATES_START), ord(self.SURROGATES_END)
        for cc in range(65536):
            if sur_start <= cc <= sur_end: continue
            self.cc2unicode[chr(cc)] = chr(cc)
        self.reset_unicode2cc()

    # --------------------------------------------------------------------------- read_to_unicode_stream()

    def read_to_unicode_dict(self, ToUnicode:IndirectPdfDict):
        '''
        Create a CMap from the PDF ToUnicode CMap stream
        '''

        self.cc2unicode = {}
        self.unicode2cc = {}

        stream = PdfFilter.uncompress(ToUnicode).stream

        if stream == None:
            warn(f'no stream in font\'s ToUnicode object: {ToUnicode}')
            return
                
        # process bfranges

        bfRanges = re.split(r'beginbfrange',stream); del bfRanges[0]
        bfRanges = [re.split(r'endbfrange',range)[0] for range in bfRanges]

        for block in bfRanges:

            block = re.sub('[\r\n]+','',block)
            tokens = re.split(r'(<|>|\[|\])',block)
            isHex,isValueList = False,False
            rangeList,valueList = [],[]

            for t in tokens:
                if t == '<': isHex = True
                elif t == '>': isHex = False
                elif t == '[': isValueList = True
                elif t == ']': isValueList = False; rangeList.append(valueList); valueList = []
                elif isHex:
                    u = PdfTextString.UTF16BE_to_Unicode(t)
                    if u == None:
                        warn(f'bad token in ToUnicode CMap: {t}')
                    if isValueList:
                        valueList.append(u if u != None else chr(0))
                    else:
                        rangeList.append(u if u != None else chr(0))

            if len(rangeList) % 3 != 0: err(f'bfrange: not a whole number of triplets in a bfrange block: {block}')
 
            # Increment last char of a string (sames as incrementing the last byte, but faster; see PDF Ref. 1.7 p. 474)
            INCREMENT = lambda s,i: (s[:-1] + chr(ord(s[-1]) + i)) if ord(s[-1])//256 == (ord(s[-1]) + i)//256 else None

            for i in range(0,len(rangeList),3):
                start,end,u = rangeList[i:i+3]
                start,end = ord(start),ord(end)
                if isinstance(u, str):
                    if INCREMENT(u,end-start) is None:
                        raise ValueError(f'string increment crosses the 256-byte boundary; see PDF Ref. 1.7 p. 474')
                    for k in range(start,end+1):
                        self.cc2unicode[chr(k)] =  INCREMENT(u,k - start)
                else:
                    assert len(u) == end-start+1
                    for k in range(start, end+1):
                        self.cc2unicode[chr(k)] = u[k - start]
    
        # process bfchars

        bfChars = re.split(r'beginbfchar',stream); del bfChars[0]
        bfChars = [re.split(r'endbfchar',range)[0] for range in bfChars]

        for block in bfChars:

            block = re.sub('[\r\n]+','',block)
            tokens = re.split(r'(<|>)',block)
            isHex = False
            charList = []

            for t in tokens:
                if t == '<': isHex = True
                elif t == '>': isHex = False
                elif isHex:
                    u = PdfTextString.UTF16BE_to_Unicode(t)
                    # if u == None: u = chr(0) # This is a quick and dirty fix for errors in existing ToUnicode CMaps
                    if u == None: warn('bad token in ToUnicode CMap: ' + t) 
                    charList.append(u if u != None else chr(0))

            if len(charList) % 2 != 0: err(f'bfchar: not a whole number of pairs in a bfchar block: {block}')

            for i in range(0,len(charList),2):
                start,u = charList[i:i+2]
                self.cc2unicode[start] = u

        self.reset_unicode2cc()

    # --------------------------------------------------------------------------- read_htf_file()

    def read_htf_file(self, htfFilePath:str):
        '''Create a CMap from an htf file
        '''
        self.cc2unicode = {}

        with open(htfFilePath, 'r') as f:
            # Read htf file header
            headerString = f.readline().strip(' \t\r\n')
            headerList = re.split(r"\s+", headerString)
            if len(headerList) != 3: err(f'invalid htf file header in {htfFilePath}')

            startPos = int(headerList[1])
            endPos   = int(headerList[2])
            if startPos < 0 or startPos > endPos or endPos > 255:
                err(f'invalid startPos or endPos in the htf file header in {htfFilePath}')

            # This iteration starts with the second line of the file (bc of readline() above), however n starts with 0!
            nLines = 0
            footerFound = False
            for n,s in enumerate(f):
                s = s.strip(' \t\r\n')
                if s == headerString: footerFound = True; break # an htf footer is just a copy of the header
                if s != '': nLines += 1
                if n > endPos-startPos: continue 
                cid = n + startPos

                # Character codes are in the first column
                lineList = re.split(r"\s+", s)
                c=lineList[0]
                str=c[1:-1]
                l=len(str)
                bs = chr(92) # Backslash character used for escaping

                if c[0] != c[-1]: u = chr(0)
                elif l == 1 or (l == 2 and str[0] == bs): u = str[-1]
                elif l == 8 and str[0:3] == "&#x" and str[7] == ';': u = chr(int(str[3:7],16))
                elif l == 9 and str[0:3] == "&#x" and str[8] == ';': u = chr(int(str[3:8],16))
                else: u = None

                if u is not None:
                    self.cc2unicode[chr(cid)] = u
 
        # Sanity checks
        if not footerFound: err(f'footer != header in htf file: {htfFilePath}')
        if nLines != endPos-startPos+1:
            err(f'number of lines in {htfFilePath} is {nLines} while the htf header specifies {endPos-startPos+1}')

        self.reset_unicode2cc()

    # --------------------------------------------------------------------------- read_bfr_file()

    def read_bfr_file(self, bfrFilePath:str):
        '''Create a CMap from a bfr file
        '''
        INCREMENT = lambda s,i: (s[:-1] + chr(ord(s[-1]) + i)) if ord(s[-1])//256 == (ord(s[-1]) + i)//256 else None

        self.cc2unicode = {}
        start,start2, stop = -1,-1,-1
        with open(bfrFilePath, 'r') as f:
            for lineNo, line in enumerate(f):
                line = re.sub(r'#[ ]+.*', '', line) # strip comments
                line = line.rstrip('\r\n ') # strip end-of-line chars
                if line == '': continue
                if line[0] == '<' and line[-1]== '>':
                    try:
                        lineSplit = re.split(r'>\s*<',line.strip('<>'))
                        if len(lineSplit) == 3:
                            start,stop,hex = lineSplit
                            start,stop = int(start,16),int(stop,16)
                            u = PdfTextString.UTF16BE_to_Unicode(hex)
                            if start == stop:
                                self.cc2unicode[chr(start)] = u
                            elif INCREMENT(u, stop-start) is None:
                                err(f'bad line {lineNo} in a BFR file: string increment crosses the 256-byte boundary; file: {bfrFilePath}, line: {line}')
                            else:
                                for cc in range(start,stop+1):
                                    self.cc2unicode[chr(cc)] = INCREMENT(u, cc-start)
                            start2 = ((stop // 16 ) + 1) * 16
                        elif len(lineSplit) == 1:
                            start2 = int(lineSplit[0],16)
                            if start2 % 16 != 0: err(f'start position is not a multiple of 16: {lineSplit[0]}; file: {bfrFilePath}')
                        else:
                            err(f'bad line {lineNo} in a BFR file: {bfrFilePath}, line: {line}')
                    except:
                        err(f'bad line {lineNo} in a bfr file {bfrFilePath}: {line}')
                elif len(line) <= 16 and start2 != -1:
                    if len(line) < 16: line = line + ' '*(16-len(line))
                    for i in range(len(line)):
                        if line[i] != '.': self.cc2unicode[chr(start2 + i)] = line[i]
                    start2 += 16
                else:
                    err(f'bad line {lineNo} in a bfr file {bfrFilePath}: {line}')
        self.reset_unicode2cc()

    # --------------------------------------------------------------------------- write_bfr_file()

    def write_bfr_file(self, bfrFilePath:str):
        '''
        Write CMap to a bfr file using the compact format
        '''
        cc2u = self.cc2unicode
        s = "# dots '.' mean '.notdef'; to encode the dot per se, use e.g.: <002E><002E><002E>\n"
        s = ''
        skip = False
        firstLine = True
        keys = sorted(cc2u.keys())
        if len(keys) == 0: return s

        # Combining marks, spaces and other special chars that are hard to deal with
        special = [i for i in range(32)] \
                + [i for i in range(0x007f,0x00a1)] \
                + [i for i in range(0x0300,0x0370)] \
                + [i for i in range(0x0483,0x048a)] \
                + [i for i in range(0x0590,0x05d0)] \
                + [0x0649] \
                + [i for i in range(0x0600,0x0700)] \
                + [i for i in range(0x1dc0,0x1e00)] \
                + [i for i in range(0x2000,0x2010)] \
                + [i for i in range(0x2028,0x2030)] \
                + [i for i in range(0x205F,0x2070)] \
                + [i for i in range(0x20d0,0x2100)] \
                + [i for i in range(0xa700,0xa720)] \
                + [i for i in range(0xe000,0xf900)] \
                + [0xfb1e] \
                + [i for i in range(0xfc5e,0xfc63)] \
                + [i for i in range(0xfe00,0xfe10)] \
                + [i for i in range(0xfe20,0xfe30)] \
                + [0xfeff] \
                + [i for i in range(0xfff0,0x10000)] \
                + [0xad]
        special = [chr(i) for i in special]

        triples = {}
        row_first, row_last = ord(keys[0])//16, (ord(keys[-1])+15)//15
        for row in range(row_first, row_last):
            line = ''
            for col in range(16):
                cc = chr(row*16 + col)
                if cc not in cc2u: line += '.'; continue
                u = cc2u[cc]
                if u in special or len(u) != 1: line += '.'; triples[cc] = u; continue
                line += u
            if line == '.'*16: skip = True; continue
            if skip or row % 16 == 0 or firstLine: s += f'<{row*16:04X}>\n'
            s += line + '\n'
            firstLine = False
            skip = False
 
        dotLines = ''.join(f'<{ord(cc):04X}><{ord(cc):04X}><002E>\n' for cc in cc2u if cc2u[cc] == '.')
        if dotLines != '':
            s += '# The dot\n' + dotLines

        if len(triples) > 0:
            s += '# Special & multibyte chars\n'
            bfRanges = PdfFontCMap.to_bfranges(triples)
            s += PdfFontCMap.bfranges_to_stream(bfRanges, isCid = True)
            s += '\n'
 
        with open(bfrFilePath, 'wt') as f:
            f.write(s)

    # --------------------------------------------------------------------------- read_diff_block_list()

    def read_diff_block_list(self, diff_block_list:list):
        '''Create a CMap from a diff block list.
        An example of diff block list is: [(3,'0x20','0x7e'),(0,'0xa1','0xef'),(1,'0xf1','0xfe')]
        where the first member of the tuple is the padding of the block w/r to prev. block's end
        (or zero if the padded block is the first one),
        and the blocks themselves are given by hex numbers of the start and end of each block
        '''
        self.cc2unicode = {}
        offset = 0

        for n in range(len(diff_block_list)):
            offset += diff_block_list[n][0]
            start = int(diff_block_list[n][1],0)
            end = int(diff_block_list[n][2],0)
            for k in range(start,end+1):
                if offset >=0: self.cc2unicode[chr(offset)] = chr(k)
                offset+=1

        self.reset_unicode2cc()

    # --------------------------------------------------------------------------- reset_unicode2cc()

    def reset_unicode2cc(self):
        '''
        Re-creates self.unicode2cc based on self.cc2unicode. In the case where cc2unicode maps many to one,
        the inverse maps to the first occurrence of the value.
        '''
        self.unicode2cc = {}
        for k,v in self.cc2unicode.items():
            if v not in self.unicode2cc: self.unicode2cc[v] = k

    # --------------------------------------------------------------------------- compose()

    def compose(self, composer:'PdfFontCMap', impose=False):
        '''Returns a tuple (CMap, modified) where CMap is a composite CMap made by applying
        the composer CMap to the values of self.cmap whenever possible.
        If impose == True, composer mappings for CIDs that are not in self.cmap are added to the result.
        The value of modified tells if the returned CMap is actually different from self
        '''
        result = PdfFontCMap(); result.cc2unicode = self.cc2unicode.copy()

        modified = False

        for cid,unicode in result.cc2unicode.items():
            # Skip all multiple-char entries in the map
            if len(unicode) == 1 and unicode in composer.cc2unicode:
                result.cc2unicode[cid] = composer.cc2unicode[unicode]
                modified = True

        if impose:
            for cid,unicode in composer.cc2unicode.items():
                if cid not in result.cc2unicode:
                    result.cc2unicode[cid] = unicode
                    modified = True

        result.reset_unicode2cc()

        return result, modified

    # --------------------------------------------------------------------------- decode()

    def decode(self, encodedString:str):
        '''
        Decodes an encoded string, in which chars represent character codes (cc):
        maps all chars through self.cc2unicode and returns a string of corresponding Unicode values.
        If a char is not in self.cc2unicode it is mapped to the Private Use Area: chr(ord(self.PUP15)+ord(c)).
        The self.cc2unicode.values() may contain ligatures (multi-char strings like 'ff', 'ffi', etc.), in
        which case a single character code will be mapped to multiples chars in the resulting
        Unicode string. For this reason, len(self.decode(s)) >= len(s).
        '''
        return ''.join(self.cc2unicode[c] if c in self.cc2unicode \
                        else chr(ord(self.PUP15)+ord(c)) for c in encodedString)

    # --------------------------------------------------------------------------- encode()

    def encode(self, unicodeString:str):
        '''
        Encodes a unicode string: maps all chars through self.unicode2cc and returns a string of chars
        representing character codes (cc). If ligatures (multi-char keys like 'ff', 'ffi', etc.)
        are present in self.unicode2cc.keys() these ligatures are mapped to a single char character code
        in the resulting encoded string. For this reason, len(self.encode(s)) <= len(s).
        '''
        u = unicodeString
        rmap = self.unicode2cc
        i,encodedString = 0,''
        while i<len(u):
            if u[i:i+3] in rmap: c = rmap[u[i:i+3]]; i+=3 # 3-char ligatures
            elif u[i:i+2] in rmap: c = rmap[u[i:i+2]]; i+=2 # 2-char ligatures
            elif u[i] in rmap: c = rmap[u[i]]; i+=1; # 1-char (single character)
            else:
                d = ord(u[i]) - ord(self.PUP15) # remap to the unmapped original that was stored in private-use area
                if d<0 or d>self.UNICODE_BMP_MAX:
                    return None
                    # warn(f'failed to encode: {[u[i]]}')
                    # d = 0 # If remapping fails return chr(0)
                c = chr(d)
                i+=1
            encodedString += c
        return encodedString
    
    # --------------------------------------------------------------------------- to_bfranges()

    @staticmethod
    def to_bfranges(cc2unicode:dict[str,str]):
        '''Encode CMap as bfranges (list of 3-tuples of str: (start, stop, value_start))
        '''
        m = cc2unicode
        if len(m) == 0: return []
        start,stop,v_start = None,None,None
        result = []

        # Returns s whose last byte has been incremented by i if such an increment does not lead to an overflow,
        # or None otherwise; see PDF Ref. 1.7 p. 474 
        INCREMENT = lambda s,i: (s[:-1] + chr(ord(s[-1]) + i)) if ord(s[-1])//256 == (ord(s[-1]) + i)//256 else None

        for key in sorted(m):

            k = ord(key) # key as int
            v = m[key]
            if v is None: continue
            assert len(v) <= 512 # PDF Ref. 1.7 p. 474 

            if start is None:
                start, stop, v_start = k, k, v
            elif k == stop + 1 and k//256 == start//256 \
                    and v == INCREMENT(v_start, stop - start + 1) \
                    and len(v_start) <= 2:
                # The second condition is necessary but is undocumented in the PDF Ref.!
                # (it's documented in the Adobe Technical Notes on CMaps);
                # The last condition is due to a bug in Adobe Acrobat:
                # it's unable to increment (last bytes of) strings that are >2 bytes long
                stop = k
            else:
                result.append((chr(start), chr(stop), v_start))
                start, stop, v_start = k, k, v

        if start != None:
            result.append((chr(start), chr(stop), v_start))

        return result

    # --------------------------------------------------------------------------- bfranges_to_stream()

    @staticmethod
    def bfranges_to_stream(bfranges:list, isCid:bool):
        '''Convert CMap encoded as bfranges (list of 3-tuples of ints) to stream (str)
        '''
        w = 4 if isCid else 2
        h = lambda i,width: f'{i:0{width}X}' # convert int i to a hex-string of specified width
        toUTF16BE = lambda s: '<' + PdfTextString.Unicode_to_UTF16BE(s) + '>'

        result = []
        for start,stop,value in bfranges:
            start, stop = ord(start), ord(stop)
            if not isCid and (start >= 256 or stop >= 256):
                continue
            v = toUTF16BE(value) if not isinstance(value, list) \
                else '[' + ''.join(toUTF16BE(v) for v in value) + ']'
            result.append(f'<{h(start,w)}><{h(stop,w)}>{v}')
        return '\n'.join(result)


    # --------------------------------------------------------------------------- write_pdf_stream()

    def write_pdf_stream(self, CMapName:PdfName, isCID:bool):
        '''Creates a PDF ToUnicode CMap dictionary stream (see section 9.10.3 ToUnicode CMaps of PDF 1.6 Spec).
        The argument CMapName is the value of the /CMapName entry of the produced stream and should
        be a PdfName itself (i.e., it should start with a slash)
        '''
        if CMapName[0] != '/' or ' ' in CMapName: err(f'invalid CMapName: {CMapName}')
        bfranges = PdfFontCMap.to_bfranges(self.cc2unicode)
        if len(bfranges) == 0:
            raise ValueError(f'empty cmap: {CMapName}')
        bfrStream = PdfFontCMap.bfranges_to_stream(bfranges, isCID)
        bfrStreamLength = len(re.split(r'\n',bfrStream))
        return '\n'.join((
            "/CIDInit /ProcSet findresource begin",
            "12 dict begin",
            "begincmap",
            "/CIDSystemInfo",
            "<</Registry (Adobe)",
            "/Ordering (UCS)",
            "/Supplement 0",
            ">> def",
            "/CMapName " + CMapName + " def",
            "/CMapType 2 def",
            "1 begincodespacerange",
            '<0000><FFFF>' if isCID else '<00><FF>',
            "endcodespacerange",
            f"{bfrStreamLength} beginbfrange",
            bfrStream,
            "endbfrange",
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end"
        ))

    # --------------------------------------------------------------------------- get_unicode_offset()

    @staticmethod
    def get_segment_offset(listOfStrings):
        '''
        For a list of `listOfStrings`, returns an `offset` (a multiple of 256) such that:

        `all(len(s) == 1 and offset <= ord(s) < offset + 256 for s in listOfStrings) == True`.

        If such offset does not exist returns None.
        '''
        if len(listOfStrings) == 0: return None
        offset = (ord(next(iter(listOfStrings))) >> 8) << 8
        return offset if all(len(u) == 1 and offset <= ord(u) <= offset+255 for u in listOfStrings) else None

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
        return (fd is not None) and (fd.FontFile or fd.FontFile2 or fd.FontFile3) is not None

    # -------------------------------------------------------------------------------- is_embedded()

    @staticmethod
    def is_core14(font:PdfDict):
        '''
        Checks if the font is embedded
        '''
        return font.Subtype in ['/Type1', '/TrueType'] \
            and not PdfFontDictFunc.is_embedded(font) \
            and PdfFontCore14.standard_fontname(font.BaseFont) is not None
 
    # -------------------------------------------------------------------------------- is_symbolic()

    @staticmethod
    def is_symbolic(font:PdfDict):
        '''
        Depending on the values of Symbolic/nonSymbolic bits in font descriptor's Flags entry,
        returns True if the Symbolic bit is set, False if nonSymbolic bit is set,
        and None if the font does not have a descriptor or the font descriptor's
        Flags entry is missing or corrupt. If both Symbolic/nonSymbolic bits in the Flags entry
        are either set or unset, an exception is raised.
        '''
        try: Flags = int(PdfFontDictFunc.get_font_descriptor(font).Flags)
        except: return None

        isSymbolic = (Flags & 4 == 4)
        isNonsymbolic = (Flags & 32 == 32)
        if isSymbolic  == isNonsymbolic:
            raise ValueError(f'inconsistent Symbolic/nonSymbolic bits combo in font: {PdfFontDictFunc.get_font_name(font)}')

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

    # -------------------------------------------------------------------------------- get_scaleFactor()

    @staticmethod
    def get_scaleFactor(font:PdfDict):
        '''
        Returns scale factor between font's units and document units; PDF Ref Sec. 5.5.4 Type 3 fonts
        '''
        return abs(float(font.FontMatrix[0])) if font.FontMatrix != None else 0.001

    # -------------------------------------------------------------------------------- get_missingWidth()

    @staticmethod
    def get_missingWidth(font:PdfDict):
        '''
        Returns the MissingWidth (in document units)
        ''' 
        z = PdfFontDictFunc.get_scaleFactor(font)
        if font.Subtype == '/Type3': return 0
        elif font.Subtype == '/Type0':
            return z * float(font.DescendantFonts[0].DW or 1000)
        else:
            fd = font.FontDescriptor
            return z * float(fd.MissingWidth or 0) if fd else 0

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

    # -------------------------------------------------------------------------------- get_encoding_name_list()

    @staticmethod
    def get_encoding_name_list(font:PdfDict):
        '''
        The name of font.Encoding as a list
        '''
        if not font or not font.Encoding: return []
        if not PdfFontDictFunc.is_cid(font):
            enc = font.Encoding
            if not isinstance(enc, PdfDict): return [enc]
            return [enc.BaseEncoding or 'None', '/Differences' if enc.Differences else 'None']
        else:
            dFont = font.DescendantFonts[0]
            s = []
            if dFont.CIDToGIDMap: s.append('/CIDToGIDMap')
            if dFont.FontDescriptor.CIDSet: s.append('/CIDSet')
        return s
    # -------------------------------------------------------------------------------- get_encoding_name_string()

    @staticmethod
    def get_encoding_name_string(font:PdfDict):
        '''
        String representation of the font's encoding
        '''
        s = PdfFontDictFunc.get_encoding_name_list(font)
        if len(s) == 0: return None
        elif len(s) == 1: return s[0]
        else: return '[' + ', '.join(s) +']'

    # -------------------------------------------------------------------------------- differences_to_cc2glyphname()

    @staticmethod
    def differences_to_cc2glyphname(differences:list) -> dict[str,PdfName]:
        '''
        Converts encoding differences list to an encoding map
        — a dictionary that maps from character codes (str) to glyph names (PdfName).
        '''
        i = 0
        encMap = {}
        for d in differences:
            if d == '': err(f'empty string in /Differences: {differences}')
            if isinstance(d,int):
                i = d
            elif all(c in string.digits for c in d):
                i = int(d)
            elif d[:2] == '0x' and all(c in string.hexdigits for c in d[2:]):
                i = int(d,0)
            elif d[0] == '/' and len(d)>1:
                if i > 255: err(f'index out of range in /Differences: {differences}')
                encMap[chr(i)] = PdfName(d[1:])
                i += 1
            else:
                err(f'a token in /Differences is not a glyph or a number: {d}; full /Differences follow:\n{differences}')

        return encMap

    # -------------------------------------------------------------------------------- cc2glyphname_to_differences()

    @staticmethod
    def cc2glyphname_to_differences(cc2gname:dict[str,PdfName]):
        '''
        Encodes a cc2glyphname map as a /Differences list (PdfArray).
        All characters in PdfName-s in differencesList are replaced with #-codes, as per PDF Ref. 3.2.4.
        This should be done on the level of pdfrw's PdfWriter class, but sadly it is not.
        '''
        if len(cc2gname) == 0:
            return PdfArray([])

        i = -1000
        firstChar, lastChar = None, None
        diff = []
        codes = [c for c in cc2gname.keys()]
        codes.sort()
        for code in codes:
            if not 0 <= ord(code) <= 255:
                raise ValueError(f'char code is not in the 0..255 range: {[code]}')
            i += 1
            if ord(code) != i:
                i = ord(code)
                diff.append(i)
                if firstChar == None: firstChar = i
            diff.append(cc2gname[code])
        lastChar = i
        assert firstChar != None
        assert lastChar != None

        # Convert chars outside 33..126 range to #-codes; see PDF Ref. Sec. 3.2.4
        for i in range(len(diff)):
            s = diff[i]
            if isinstance(s, str):
                assert s[0] == '/'
                r = ''.join(c if 33 <= ord(c) <= 126 and ord(c) != 35 else f'#{ord(c):02x}' for c in s[1:])
                diff[i] = PdfName(r)

        return PdfArray(diff), firstChar, lastChar

    # -------------------------------------------------------------------------------- make_font_descriptor()

    @staticmethod
    def make_font_descriptor(info:dict,
                             pfb:bytes = None,
                             cff:bytes = None,
                             ttf:bytes = None,
                             otf:bytes = None,
                             makeCID:bool = False):
        '''
        Creates a font descriptor; consult PDF Ref. 1.7, sec. 5.8: Embedded Font Programs
        '''
        assert bool(pfb) or bool(cff) or bool(ttf) or bool(otf)

        FontFile = FontFile2 = FontFile3 = None

        # Calculate the Flags bit field.

        # This is required by PDF, but one might as well just set flags=32
        # The bits (from the 1st bit up):
        # FixedPitch,Serif,Symbolic,Script,0,Nonsymbolic,Italic,0,0,0,0,0,0,0,0,0,AllCap,SmallCap,ForceBold
        # The XOR of Symbolic & Nonsymbolic bits should always be 1

        Flags = 0
        if info.get('isFixedPitch'): Flags += 1
        if info.get('isSerif'): Flags += 2
        Flags += 4 if info.get('isSymbolic') else 32
        if info.get('ItalicAngle') : Flags += 64

        # Create the font program

        if pfb: # Make FontFile

            lengths = []
            fontProgram = b''
            with BytesIO(pfb) as f:
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

        elif ttf: # Make FontFile2

            FontFile2 = IndirectPdfDict(
                Length1 = len(ttf),
                stream=py23_diffs.convert_load(ttf)
            )

        else: # Make FontFile3

            FontFileSubtype = PdfName('Type1C') if cff and not makeCID \
                        else PdfName('CIDFontType0C') if cff and makeCID \
                        else PdfName('OpenType') if otf \
                        else None

            FontFile3 = IndirectPdfDict(
                Subtype = FontFileSubtype,
                stream=py23_diffs.convert_load(cff or otf)
            )

        # Fix FontBBox if necessary
        bbox = [float(x) for x in info.get('FontBBox',[0,0,1000,1000])]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or bbox[1] > info['Descent'] or bbox[3] < info['Ascent']:
            bbox = [0,info['Descent'],1000, max(1000, info['Ascent'])]
        
        return IndirectPdfDict(
            Type = PdfName('FontDescriptor'),
            FontName = PdfName(re.sub(r'\s+', '', info['FontName'])),
            Flags = Flags,
            FontBBox = PdfArray(bbox),
            ItalicAngle = info.get('ItalicAngle', 0),
            Ascent = info['Ascent'],
            Descent = info['Descent'],
            CapHeight = info['CapHeight'],
            XHeight = info.get('XHeight'), # optional
            StemV = info['StemV'],
            FontFile  = FontFile,
            FontFile2 = FontFile2,
            FontFile3 = FontFile3
        )

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

        info = {'Type':'PFB'}

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
                                'Medium':100, 'NORMAL':100, 'Normal':100, 'Regular':100, 'regular':100,
                                'Italic':100, 'Roman':100, 'Book':100,
                                'Semibold':120, 'Bold':140, 'Bold Italic':140}
            info['StemV'] = weight_to_stemv[weight]

        # This initializes char.width-s
        CharStrings = t1.font["CharStrings"]
        for gname in CharStrings:
            CharStrings[gname].draw(NullPen()) # Initializes glyph.width-s

        # Set the maps
        info['gid2gname'] = {i:gname for i,gname in enumerate(t1.font['Encoding']) 
                             if gname != '.notdef' and gname in CharStrings}
        info['gname2width'] = {gname:CharStrings[gname].width for gname in CharStrings}

        # Return info
        return info

    # -------------------------------------------------------------------------------- distill_glyphSet()

    @staticmethod
    def distill_glyphSet(glyphSet:dict):
        '''
        The TTF/OTF glyphSet dict may contain undefined glyphs - those with no drawing instructions.
        This function returns a sub-dict of glyphSet which contains only the glyphs which are actually defined
        (i.e. those, whose .draw() method actually initializes glyph.width and/or BoundsPen().bounds).
        All glyphs in the returned dict have their .width well-defined (as a by-product of calling the .draw() methods).
        '''
        distilled = {}

        # The .notdef has a .width, but no .bounds are set by .draw(BoundsPen()) for it for some reason
        if notdef := glyphSet.get('.notdef'):
            notdef.draw(NullPen())
            distilled['.notdef'] = notdef

        for gname, glyph in glyphSet.items():
            pen = BoundsPen(glyphSet)
            glyph.draw(pen)
            if pen.bounds is not None or glyph.width is not None:
                distilled[gname] = glyph

        return distilled

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

        info = {'TYPE':'TTF'}

        # Get font family, subfamily (normal/italic/bold/bolditalic) & full name
        # See: https://docs.microsoft.com/en-us/typography/opentype/spec/name
        if name := ttFont.get('name'):

            for record in name.names:

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

        else:
            warn(f'no name table; setting FontName = Unknown')
            info['FontName'] = 'Unknown'

        fontName = info.get('FontName')

        # MAXP
        if maxp := ttFont.get('maxp'):
            info['numGlyphs'] = maxp.numGlyphs
        else:
            warn(f'no maxp table: {fontName}')

        # HEAD
        if head := ttFont.get('head'):

            info['flags'] = head.flags

            info['unitsPerEm'] = head.unitsPerEm
            z = float(1000/info['unitsPerEm'])

            minMax = [head.xMin, head.yMin, head.xMax, head.yMax]
            info['FontBBox'] = [int(z*x) for x in minMax]

        else:
            z = 1.0
            warn(f'no head table: {fontName}')

        # OS2
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

            info['isSerif']         = os2.panose.bFamilyType == 2 and os2.panose.bSerifStyle
            info['isScript']        = os2.panose.bFamilyType == 3

        elif hhea := ttFont.get('hhea'):

            info['Ascent']      = int(z*hhea.ascent)
            info['Descent']     = int(z*hhea.descent)
            info['CapHeight']   = int(z*hhea.ascent)
            info['StemV']       = 100

        else:
            warn(f'no OS/2 or hhea table: {fontName}')

        # Stylistic parameters
        if post := ttFont.get('post'):
            info['ItalicAngle']     = post.italicAngle
            info['isFixedPitch']    = post.isFixedPitch

        # info['Widths'] = PdfArray(ttf.metrics.widths)
        # info['DefaultWidth'] = int(round(ttf.metrics.defaultWidth, 0))

        # GlyphSet
        try: glyphSet = ttFont.getGlyphSet()
        except: glyphSet = None

        if glyphSet is not None:
            gSet = PdfFontFile.distill_glyphSet(glyphSet)
        elif glyf := ttFont.get('glyf'):
            gSet = {g:glyf[g] for g in glyf.keys()}
        else:
            warn(f'failed to obtain glyphSet: {fontName}')
            gSet = {}

        getWidth = lambda glyph: glyph.width if hasattr(glyph, 'width') else 0
        info['gid2gname'] = {ttFont.getGlyphID(gname):gname for gname in gSet}
        info['gname2width'] = {gname:getWidth(gSet[gname]) for gname in gSet}

        # CMAP
        if cmap := ttFont.get('cmap'):

            # Platform IDs: 0:Unicode, 1:Macintosh, 2:ISO(deprecated), 3:Windows, 4:Custom
            # Windows PlatformEncIDs: (3,0):Symbol, (3,1):UnicodeBMP

            for table in cmap.tables:
                info[f'cmap{table.platformID}{table.platEncID}'] \
                    = {chr(code):gname for code,gname in table.cmap.items()}

            info['isSymbolic'] = 'cmap30' in info
            info['unicode2gname'] = ttFont.getBestCmap()

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

        info = {'Type':'CFF'}

        # Parse the CFF data
        cffFont = CFFFontSet()
        cffFont.decompile(file = BytesIO(cff), otFont = TTFont())

        # Access the first font in the CFF font set
        assert len(cffFont.fontNames) == 1
        FontName = cffFont.fontNames[0]
        info['FontName'] = FontName

        font = cffFont[0]

        # Presence of ROS means a CID-encoded CFF font
        info['ROS'] = font.ROS if hasattr(font, 'ROS') else None

        try:
            chars = font.CharStrings    
        except:
            warn(f'CFF font has no CharStrings: {FontName}')
            chars = None

        if chars is not None:

            # This initializes char.width-s
            for gname in chars.keys():
                chars[gname].draw(NullPen())

            # gname2width
            info['gname2width'] = {gname:chars[gname].width for gname in chars.keys()}

            # gid2gname
            if info['ROS']: # CID-keyed CFF (glyph names are: cid1234 )

                assert all(g[:3] == 'cid' or g == '.notdef' for g in chars.keys())
                info['gid2gname'] = {(int(gname[3:]) if gname != '.notdef' else 0):gname for gname in chars.keys()}
                info['CID-keyed'] = True

            elif font.Encoding is not None: # Simple CFF

                if isinstance(font.Encoding, list):

                    info['gid2gname'] = {i:gname for i,gname in enumerate(font.Encoding)
                                        if gname != '.notdef' and (len(chars.keys()) == 0 or gname in chars.keys())}

                elif isinstance(font.Encoding, str):

                    trans = {'ExpertEncoding':'MacExpertEncoding'}
                    enc = trans.get(font.Encoding, font.Encoding)
                    if cc2g := PdfFontEncodingStandards.get_cc2glyphname(PdfName(enc)):
                        info['gid2gname'] = {ord(cc):g[1:] for cc,g in cc2g.items() if g[1:] in chars.keys()}

            if info.get('gid2gname') is None:

                warn(f'bad CFF font encoding: {font.Encoding}')
                info['gid2gname'] = {}

        else:
            info['gid2gname'] = {}
            info['gname2width'] = {}
 
        # Get the FontMatrix
        info['FontMatrix'] = font.FontMatrix

        # Get the FontBBox
        try:
            bbox = font.FontBBox
        except:
            bbox = None

        if bbox is None:
            try:
                font.recalcFontBBox()
                bbox = font.FontBBox
            except:
                warn(f'failed to get FontBBox; setting to [0,0,1000,1000]: {FontName}')
                bbox = [0,0,1000,1000]

        info['FontBBox'] = bbox

        # Get font params
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
    def read_core14_info(fontName:PdfName) -> dict:
        '''
        '''
        info = {'FontName':fontName}

        # gid2gname
        if builtInEncoding := PdfFontCore14.built_in_encoding(fontName):
            cc2g = PdfFontEncodingStandards.get_cc2glyphname(builtInEncoding)
            info['gid2gname'] = {ord(cc):gname[1:] for cc,gname in cc2g.items()}

        # gname2width
        if name2width := PdfFontCore14.make_name2width(fontName):
            info['gname2width'] = {name[1:]:width for name,width in name2width.items()}

        return info

    # -------------------------------------------------------------------------------- print_xml()

    @staticmethod
    def cff_to_xml(cff:bytes, filePath:str):
        '''
        Writes a CFF font program to file.
        '''
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
                 name:PdfName = None,
                 pfb:bytes = None,
                 cff:bytes = None,
                 ttf:bytes = None,
                 otf:bytes = None,
                 font:PdfDict = None,
                 glyphMap:PdfFontGlyphMap = PdfFontGlyphMap(),
                 encoding:ENC_TYPE = None,
                 cc2unicode:dict = None,
                 extractFontProgram:bool = True,
                 makeSyntheticCmap:bool = True):
        '''
        Creates an instance of PdfFont from a PFB(PFA)/CFF/TTF/OTF font file (bytes),
        or from a PDF font dictionary object (PdfDict).
        '''

        assert name or pfb or cff or ttf or otf or font

        self.name = name
        self.pfb = pfb
        self.cff = cff
        self.ttf = ttf
        self.otf = otf
        self.font = font
        self.glyphMap = glyphMap

        self.info = None

        # Extract font program (populates self.pfb|cff|ttf|otf if the font argument is present)
        if extractFontProgram:
            self.extract_font_program()

        # Set self.info
        if self.name:

            self.font = PdfFontCore14.make_core14_font_dict(fontname = name, encoding = encoding)
            if self.font is None:
                raise ValueError(f'invalid core14 font name: {name}')
            self.info = PdfFontFile.read_core14_info(self.name)

        else:
            self.info = PdfFontFile.read_pfb_info(self.pfb) if self.pfb \
                        else PdfFontFile.read_cff_info(self.cff) if self.cff \
                        else PdfFontFile.read_ttf_otf_info(self.ttf or self.otf) if (self.ttf or self.otf) \
                        else PdfFontFile.read_core14_info(font.BaseFont) if (font and PdfFontDictFunc.is_core14(font)) \
                        else {}

        # Make font dict
        if not self.font:
            self.make_font_dict(encoding = encoding, cc2unicode = cc2unicode)

        # Set font name
        if not self.name:
            self.name = self.get_font_name()

        # Get encodings
        self.cc2g, self.cc2g_internal = self.get_cc2g()
        
        # Set chars widths
        self.widthMissing = PdfFontDictFunc.get_missingWidth(self.font)
        self.cc2width = self.get_cc2w()

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
            # try: self.cmap_toUnicode = PdfFontCMap(toUnicodeDict = self.font.ToUnicode) # /ToUnicode may be junk
            # except: pass
            self.cmap_toUnicode = PdfFontCMap(toUnicodeDict = self.font.ToUnicode) # /ToUnicode may be junk
            if self.cmap_toUnicode == None: warn(f'failed to read ToUnicode CMap for font: {self.name}')

        # An internal Unicode map (read from the font program)
        self.cmap_internal = None
        if u2g := self.info.get('unicode2gname'):
            self.cmap_internal = PdfFontCMap()
            g2u = {PdfName(g):u for u,g in u2g.items()}
            self.cmap_internal.cc2unicode = {cc:chr(g2u[g]) for cc,g in self.cc2g.items() if g in g2u}
            self.cmap_internal.reset_unicode2cc()

        # Set cmap_synthetic: infer Unicode points based on glyph names using glyphMap
        self.cmap_synthetic = None
        if makeSyntheticCmap:
            self.cmap_synthetic, _ = self.make_cmap()
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

        # if re.search('Times', self.name):
        #     pprint(self.cmap_internal.cc2unicode)

    # -------------------------------------------------------------------------------- extract_font_program()

    def extract_font_program(self):
        '''
        If self.font is not None, extracts the font program from self.font and populates
        one of: self.pfb, self.ttf, self.cff, self.otf, otherwise does nothing.
        '''
        if not self.font: return

        fd = PdfFontDictFunc.get_font_descriptor(self.font)

        if fd == None: return

        FontFile = fd.FontFile or fd.FontFile2 or fd.FontFile3
        if not FontFile: return

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

    # -------------------------------------------------------------------------------- parse_encoding()

    @staticmethod
    def parse_encoding(Encoding:ENC_TYPE, info:dict):
        '''
        Parses `Encoding`, which is either a `PdfName` or a `PdfDict`, and returns a map from
        char codes (str) to glyphnames (PdfName)
        '''
        gid2gname = info.get('gid2gname') or {}

        def parse_named_encoding(Encoding:PdfName, gid2gname:dict[int,str]) -> dict[str,PdfName]:
            '''
            PDF Ref. v1.7 sec. 5.5.5: Table 5.11:
            For a font program that is embedded in the PDF file, the implicit base
            encoding is the font program's built-in encoding [..]. Otherwise, for a nonsymbolic font,
            it is StandardEncoding, and for a symbolic font, it is the font's built-in encoding.

            We assume that the built-in encoding of embedded fonts and the StandardEncoding/built-in
            encoding of non-embedded (Core14) fonts is already in gid2gname.
            '''
            cc2g = PdfFontEncodingStandards.get_cc2glyphname(Encoding) if Encoding \
                else {chr(gid):PdfName(gname) for gid,gname in gid2gname.items()}
            if cc2g is None:
                raise ValueError(f'invalid Encoding: {Encoding}')
            return cc2g

        if isinstance(Encoding, PdfDict):

            # PDF Ref. v1.7 sec. 5.5.5: Table 5.11:
            # The base encoding [is] the encoding from which the Differences
            # entry (if present) describes differences—specified as the name of a predefined
            # encoding MacRomanEncoding, MacExpertEncoding, or WinAnsiEncoding (see Appendix D).
            # If this entry is absent, the Differences entry describes differences from an implicit
            # base encoding.

            cc2g = parse_named_encoding(Encoding.BaseEncoding, gid2gname)

            if Encoding.Differences != None:
                cc2g |= PdfFontDictFunc.differences_to_cc2glyphname(Encoding.Differences)

        else:

            cc2g = parse_named_encoding(Encoding, gid2gname)
            
        return cc2g

    # -------------------------------------------------------------------------------- get_cc2g()

    def get_cc2g(self) -> tuple[dict[str,PdfName], dict[str,PdfName]]:
        '''
        Based on self.font, returns the tuple (cc2g, cc2g_internal)
        -- the explicit and internal (for TrueType fonts) maps from character codes (str) to glyph names (PdfName).
        '''

        assert self.font is not None
        assert self.info is not None

        # All self-ish variables which we need
        font = self.font
        info = self.info

        cc2g = {}             # A map from character codes (chars) to glyph names
        cc2g_internal = None    # A map from character codes (chars) to glyph names, internal to TrueType fonts

        fontName = PdfFontDictFunc.get_font_name(font)

        # .............................................................. Classify font's Encoding

        if self.is_cid():

            # Get gid2gname
            if gid2gname := info.get('gid2gname'):

                cid2gid = {}

                # Remap CIDs to GIDs if the CIDTOGID map exists
                CIDToGIDMap = font.DescendantFonts[0].CIDToGIDMap
                if CIDToGIDMap and isinstance(CIDToGIDMap, PdfDict):

                    b = py23_diffs.convert_store(PdfFilter.uncompress(CIDToGIDMap).stream)
                    cid2gid = {i:b[2*i]*256 + b[2*i+1] for i in range(len(b) // 2)}

                    # Skip all null-codes (i.e. .notdefs), except for the code at cid=0, where an actual null code might be
                    cid2gid = {cid:gid for cid,gid in cid2gid.items() if gid != 0 or cid == 0}

                # Append cids from CIDSet if it exists
                CIDSet = font.DescendantFonts[0].FontDescriptor.CIDSet
                if CIDSet:
                    byteStream = py23_diffs.convert_store(PdfFilter.uncompress(CIDSet).stream)
                    CIDSet = [(i*8 + j) for i,byte in enumerate(byteStream) for j,bit in enumerate(f'{byte:08b}') if bit == '1']
                    for cid in CIDSet:
                        if cid not in cid2gid: cid2gid[cid] = cid

                # Create cc2g
                if len(cid2gid) == 0:
                    # If cid2gid is empty, use built-in encoding
                    cc2g = {chr(gid):PdfName(gname) for gid,gname in gid2gname.items()}
                else:
                    cc2g = {chr(cid):PdfName(gid2gname.get(gid)) for cid,gid in cid2gid.items() if gid in gid2gname}

        else: # Simple fonts

            cc2g = PdfFont.parse_encoding(font.Encoding, info)
            gname2width = info.get('gname2width')

            if font.Subtype == '/TrueType':

                if font.Encoding is None or PdfFontDictFunc.is_symbolic(font) is True:

                    # TrueType fonts which are either Symbolic or lack Encoding.
                    # When the Symbolic flag is set in a TrueType font the font.Encoding is ignored
                    if cmap30 := info.get('cmap30'):
                        offset = PdfFontCMap.get_segment_offset(cmap30.keys())
                        if offset not in [0x0000, 0xf000, 0xf100, 0xf200]:
                            raise ValueError(f'bad cmap30 table in TrueType font: {fontName}:\n{cmap30}')
                        cc2g_internal= {chr(ord(cc)-offset):PdfName(gname) for cc,gname in cmap30.items()}
                    elif cmap10 := info.get('cmap10'):
                        if PdfFontCMap.get_segment_offset(cmap10.keys()) != 0:
                            raise ValueError(f'bad cmap10 table in TrueType font: {fontName}:\n{cmap10}')
                        cc2g_internal = {cc:PdfName(gname) for cc,gname in cmap10.items()}

                else:

                    if cmap10 := info.get('cmap10'):
                        stdRoman = PdfFontEncodingStandards.get_cc2glyphname('/StandardRomanEncoding')
                        stdRomanInv = {gname:cc for cc,gname in stdRoman.items()}
                    cmap31 = info.get('cmap31')

                    mapGname = lambda g: (cmap31.get(AdobeGlyphMap.get(g)) or g[1:]) if cmap31 \
                                        else (cmap10.get(stdRomanInv.get(g)) or g[1:]) if cmap10 \
                                        else g[1:]

                    cc2g_internal = {cc:PdfName(mapGname(g)) for cc,g in cc2g.items()}

                    # Distill cc2g_internal
                    if gname2width:
                        cc2g_internal = {cc:g for cc,g in cc2g_internal.items() if g[1:] in gname2width}

                # Distill cc2g
                if cc2g_internal:
                    cc2g = {cc:g for cc,g in cc2g.items() if cc in cc2g_internal}
 
            else:
 
                # Distill cc2g
                if gname2width:
                    cc2g = {cc:g for cc,g in cc2g.items() if g[1:] in gname2width}


        return cc2g, cc2g_internal

    # -------------------------------------------------------------------------------- get_cc2w()

    def get_cc2w(self):
        '''
        Based on self.font, Returns a map of character codes (chars) to widths.
        NB: the widths are in document units, not the font units. For example, widths are normally < 1 for Type 1 fonts.
        '''
        assert self.font
        font = self.font

        assert self.widthMissing is not None

        if font.Subtype != '/Type0': # simple fonts

            # font.Widths is required except for Core14 fonts
            if font.Widths != None:
                try:
                    first, last = int(font.FirstChar), int(font.LastChar)
                    assert last - first + 1 == len(font.Widths)
                except:
                    raise ValueError(f'invalid FirstChar, LastChar, Widths combination in font: {font}')
                cc2width = {chr(i + first):float(w) for i,w in enumerate(font.Widths)}
            else:
                if name2width := PdfFontCore14.make_name2width(font.BaseFont):
                    cc2width = {cc:name2width[g] for cc,g in self.cc2g.items()}
                else:
                    raise ValueError(f'missing Widths array in a non-Core14 font: {font}')

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

        # Rescale from font units to document units
        z = PdfFontDictFunc.get_scaleFactor(font)
        cc2width = {cc: w*z for cc,w in sorted(cc2width.items())}

        # Add default widths to cc's whose widths are not explicitly specified
        for cc in self.cc2g:
            if cc not in cc2width:
                cc2width[cc] = self.widthMissing

        # Remove cc2width entries that are not in self.encoding
        cc2width = {cc:w for cc,w in cc2width.items() if cc in self.cc2g}

        return cc2width

    # -------------------------------------------------------------------------------- make_font_dict()

    def make_font_dict(self,
                        encoding:ENC_TYPE = None,
                        cc2unicode:dict[str,str] = None):
        '''
        Sets self.font
        '''
                    
        # ................................................................................ Fix Font Name

        FontName = re.sub(r'\s+', '', self.info['FontName'])

        if self.ttf or self.otf:
            randomPrefix = ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
            FontName = randomPrefix + '+' + FontName

        # ................................................................................ gname2width & unitsPerEm

        gname2width = self.info.get('gname2width')
        unitsPerEm = self.info.get('unitsPerEm', 1000)

        # ................................................................................ Determine CID/Type1

        makeCID = encoding is None

        # Consistency checks
        if self.cff:
            cidEncoded = self.info['ROS'] != None
            if cidEncoded != makeCID:
                raise ValueError(f'cannot make a {"CID" if makeCID else "Type1C"} font from a {"" if cidEncoded else "non-"}CID-keyed CFF font: {FontName}')
        if self.pfb and makeCID:
            raise ValueError(f'cannot make a CID font out of a PFB font: {FontName}')

        # PDF Ref.: "[A FontFile3 with the /OpenType Subtype can appear in] a Type1 font dictionary
        # [..] if the embedded font program contains a "CFF" table without CIDFont operators."
        # Since there's no simple way of checking for the presence of CIDFont operators
        # in a CFF font inside an OpenType font we simply impose a requirement:
        # unless there's a "glyf" table in the OpenType font, it has to be a CID font,
        # and so we simply embed all such OTF fonts in CIDFontType0 fonts.

        otfWithGlyf = bool(self.otf) and 'glyf' in TTFont(BytesIO(self.otf))
        if self.otf and not otfWithGlyf and makeCID == False:
            raise ValueError(f'making of TrueType fonts from OTF fonts without the glyf table not supported: {FontName}')

        # ................................................................................ Font decriptor

        FontDescriptor = PdfFontDictFunc.make_font_descriptor(info = self.info,
                                                                pfb = self.pfb,
                                                                cff = self.cff,
                                                                ttf = self.ttf,
                                                                otf = self.otf,
                                                                makeCID = makeCID)
        
        # ................................................................................ Font Dictionary

        gname2width = self.info.get('gname2width')

        if encoding: # Simple fonts

            cc2g = PdfFont.parse_encoding(encoding, self.info)
            s = sorted(cc2g)
            FirstChar, LastChar = ord(s[0]), ord(s[-1])

            self.font = IndirectPdfDict(
                Type = PdfName('Font'),
                Subtype = PdfName('Type1') if self.pfb or self.cff else PdfName('TrueType'),
                BaseFont = PdfName(FontName),
                Encoding = encoding,
                FirstChar = FirstChar,
                LastChar = LastChar,
                FontDescriptor = FontDescriptor,
            )

            cc2g, cc2g_internal = PdfFont.get_cc2g()
            cc2g = cc2g_internal or cc2g

            DW = gname2width.get('.notdef', 0)
            W = [gname2width.get(cc2g(chr(i))[1:], DW) for i in range(FirstChar, LastChar+1)]

            if unitsPerEm != 1000:
                W = [int(w * 1000 /unitsPerEm ) for w in W]
                DW = int(DW * 1000 / unitsPerEm)

            self.font.Widths = PdfArray(W)
            self.font.FontDescrtor.MissingWidth = DW

        else: # CID fonts

            # CIDToGIDMap
            if not cc2unicode or len(cc2unicode) == 0:
                cid2gname = self.info.get('gid2gname')
                cid2gid = {cid:cid for cid in cid2gname}

                cc2unicode = None
                if u2g := self.info.get('unicode2gname'):
                    g2u = {g:u for u,g in u2g.items()}
                    cc2unicode = {chr(cid):chr(g2u.get(gname)) for cid,gname in cid2gname.items() if gname in g2u}
            else:
                # make cid2gname
                unicode2gname = self.info.get('unicode2gname')
                gname2gid = {gname:gid for gid,gname in self.info['gid2gname'].items()}
                cid2gname = {ord(cc):unicode2gname.get(ord(u)) for cc,u in cc2unicode.items() if ord(u) in unicode2gname}
                cid2gid = {cid:gname2gid.get(gname) for cid,gname in cid2gname.items()}

            # make CIDToGIDMap
            maxCID = sorted(cid2gid)[-1]
            gids = [cid2gid.get(cid,0) for cid in range(maxCID + 1)]
            CIDToGIDMap = IndirectPdfDict(
                stream=py23_diffs.convert_load(b''.join(bytes([gid >> 8, gid & 255]) for gid in gids))
            )

            # Create ToUnicode
            ToUnicode = None
            if cc2unicode:
                cmap = PdfFontCMap()
                cmap.cc2unicode = cc2unicode
                cmap.reset_unicode2cc()
                ToUnicode = IndirectPdfDict(stream = cmap.write_pdf_stream(CMapName = PdfName(FontName), isCID = True))
            
            # Widths
            cid2width = {cid:gname2width.get(gname) for cid,gname in cid2gname.items() if gname in gname2width}
            DW = gname2width.get('.notdef')

            if unitsPerEm != 1000:
                cid2width = {cid:int(w * 1000 /unitsPerEm ) for cid,w in cid2width.items()}
                DW = int(DW * 1000 / unitsPerEm) if DW else None

            W = PdfArray([x for cid,w in cid2width.items() for x in [cid,PdfArray([w])]])

            # CIDFontSubtype; see PDF Ref. Sec. 5.8, Table 5.23: Embedded font organization for various font types
            CIDFontSubtype = PdfName('CIDFontType2') if otfWithGlyf or self.ttf else PdfName('CIDFontType0')

            self.font = IndirectPdfDict(
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
                        DW = DW,
                        CIDToGIDMap = CIDToGIDMap
                    )
                ]),
                ToUnicode = ToUnicode
            )

        return

    # --------------------------------------------------------------------------- make_cmap()

    def make_cmap(self,
                  internal_cc2unicode:dict[str,str] = None,
                  explicitMap:dict[str,str] = {},
                  symbolicOffset:int = 0,
                  rebase:bool = False,
                  mapComposites = True,
                  mapSemicolons:bool = True):
        '''
        Tries to map glyph names in `self.cc2g` to Unicode points  to produce a map from
        character codes to creates to Unicode points. Returns a tuple `(CMap, unrecognized)`,
        where `CMap` is an instance of `PdfFontCMap`, and `uncrecognized` is a list of unrecognized glyph names.

        The mapping a glyph name from `self.cc2g.values()` can proceed in either of the following paths:
        
        1) the glyph name is a `known` glyph name (as per one of the available so-called `glyph lists` likes
        Adobe Glyph List, e.g.), in which case it is mapped directly to a Unicode point;
        2) the glyph name is the so-called `composite glyph name` (e.g. glyph0001), in which it is mapped
        to a symbolic code (`symbollicOffset + 0001` in this case) and this code is then looked up
        in internal_cc2unicode to map it to a Unicode point.
        '''

        encNameList = PdfFontDictFunc.get_encoding_name_list(self.font)
        encName = encNameList[0] if len(encNameList) > 0 else None
        stdGlyphMap = PdfFontGlyphMapStandards.get_glyphName2unicodeMap(encName)

        isType3 = self.font.Subtype == '/Type3'

        cmap = PdfFontCMap()
        unrecognized = {}

        baseInvMap = {}
        if rebase:
            baseEnc = self.font.Encoding.BaseEncoding if isinstance(self.font.Encoding, PdfDict) \
                        else self.font.Encoding
            if baseEnc:
                baseEncMap = PdfFont.parse_encoding(Encoding = baseEnc, info = {})
                baseInvMap = PdfFontEncodingStandards.invert_cc2glyphname(baseEncMap, baseEnc)

        # If the special BnZr gname is used for chr(0) then we shouldn't map semicolons
        if 'BnZr' in self.cc2g.items():
            mapSemicolons = False

        for cc, gname in self.cc2g.items():

            # Do not map .notdef
            if gname == '/.notdef': continue
 
            # This returns either int or str
            code = self.glyphMap.decode_gname(gname,
                                        stdGlyphMap = stdGlyphMap,
                                        isType3 = isType3,
                                        baseInvMap = baseInvMap,
                                        explicitMap = explicitMap,
                                        mapComposites = mapComposites,
                                        mapSemicolons = mapSemicolons)

            if code is None:
                unrecognized[ord(cc)] = gname[1:]
                continue

            u = None
            if isinstance(code, int):
                try: u = internal_cc2unicode.get(chr(code))
                except: u = chr(symbolicOffset + code)
            else:
                u = code

            if u != None:
                cmap.cc2unicode[cc] = u

        cmap.reset_unicode2cc()

        return cmap, unrecognized

    # -------------------------------------------------------------------------------- save()

    def save(self, basePath:str = None):
        '''
        The basePath arguments is is the intended path to the saved font file without the file's extension.
        If it's None then the self.name is chosen as the file name, and the file is saved in the current folder.
        '''
        fontProgram = self.pfb or self.ttf or self.otf or self.cff
        if fontProgram is None:
            warn(f'no font program in font: {self.name}')
            return
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

    # -------------------------------------------------------------------------------- is_cid_cff()

    def is_cid_cff(self):
        '''
        Checks if the font is a CID-keyed CFF (Type1C or CIDFontType0C) font.
        If font program is not extracted (len(self.info) == 0) returns None.
        '''
        if len(self.info) == 0: return None
        return self.info.get('Type') == 'CFF' and self.info.get('isCID') is True

    # -------------------------------------------------------------------------------- is_embedded()

    def is_embedded(self):
        '''
        Checks if the font is embedded
        '''
        return PdfFontDictFunc.is_embedded(self.font)

    # -------------------------------------------------------------------------------- is_embedded()

    def is_trivial(self):
        '''
        Trivial fonts are Times/TimesNewRoman, Arial/Helvetica & Courier/CourierNew fonts which are not embedded,
        and whose `/Encoding` is `/WinAnsiEncoding`. Such fonts do not need a ToUnicode CMap.
        '''
        core14name = PdfFontCore14.standard_fontname(self.name)
        return not self.is_embedded() \
            and (self.font.Encoding == '/WinAnsiEncoding' and core14name not in ['/Symbol', '/ZapfDingbats'] \
                or self.font.Encoding is None and core14name not in [None, '/Symbol', '/ZapfDingbats'])

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


    # -------------------------------------------------------------------------------- __repr__()

    def __repr__(self):
        '''
        A string representation of a font
        '''
        subtype = PdfFontDictFunc.get_subtype_string(self.font)
        encoding = PdfFontDictFunc.get_encoding_name_string(self.font)
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

        cc2g = self.cc2g or {}
        cc2g = {cc:gname[1:] for cc,gname in (self.cc2g or {}).items()}
        cc2g_internal = {cc:gname[1:] for cc,gname in self.cc2g_internal.items()} if self.cc2g_internal else None
        cc2w = {cc:w for cc,w in self.cc2width.items()}

        red, green, blue, gray = '1 0.25 0.25 rg', '0 0.75 0 rg', '0.25 0.25 1 rg', '0.5 g'

        # Check if the font is trivial
        isTrivial = self.is_trivial()

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
        cc2unicodePrinted = {}
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
                
                if s != '':

                    squeeze = 100 * 8 / len(s) if len(s) > 8 else 100 

                    streams[n] += f'1 0 0 1 {x} {y + 11} Tm {color} {squeeze} Tz ({s}) Tj 100 Tz 0.5 g\n'
                    cc2unicodePrinted[cc] = s

        # Print glyph name
        counter = {}
        for cc in cc2w:

            gname = cc2g.get(cc, 'None')
            
            if cc2g_internal:
                gname_internal = cc2g_internal.get(cc, 'None')
                if gname != gname_internal:
                    gname += '/' + gname_internal
            
            # Replace non-printable chars according to name syntax (see PDF Ref. sec. 3.2.4)           
            gname = ''.join(f'#{ord(c):02x}' if ord(c) <= 0x20 or 0x7f <= ord(c) < 0xa0 or c == '#' else c for c in gname)

            # The math
            cid = ord(cc)
            col,row,n = cid % 16, (cid // 16 ) % 16, (cid // 256)
            if n not in counter: streams[n] += f'/A 3 Tf 0.75 g\n' ; counter[n] = 1
            x,y = 10*(2*col), -2*10*row

            if gname != '':
                squeeze = 100 * 12 / len(gname) if len(gname) > 12 else 100
                gnameHex = gname.encode('latin').hex()
                streams[n] += f'1 0 0 1 {x} {y - 4} Tm {squeeze} Tz <{gnameHex}> Tj 100 Tz\n'

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

            if (cc in cc2unicodePrinted) or isTrivial:
                streams[n] += f'{TmString} Tm <{cidHex}> Tj\n'
            else:
                streams[n] += f'{TmString} Tm {red} <{cidHex}> Tj 0 g\n'

        # Switch back from text mode
        for n in streams: streams[n] += 'ET\n'

        def make_page_template(n:int):
            '''
            Makes a page template — a page an empty font table.
            n is the code range page number: n = 0 is 0-255 etc.
            '''

            courier = PdfFont(name = PdfName.Courier)
            arial = PdfFont(name = PdfName.Arial)

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
            encoding = PdfFontDictFunc.get_encoding_name_string(self.font)
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
        for n in sorted(streams):
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
                try: font = PdfFont(name = PdfName(name))
                except: font = None
                if font != None:
                    nameFound = name
                    break
                
            path = PdfFontUtils.findFile(name+'.*', dirList)
            if path != None:
                name,ext = os.path.splitext(path)
                data = open(path, 'rb').read()
                cc2unicode = {chr(i):chr(i) for i in range(0xd800)}

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

# =========================================================================== get_object_fonts()

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
    
    result = {id(obj):obj for name, obj in objectTuples if fontFilter(obj)}

    return result

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

    if font.cmap_internal:
        font.cmap_internal.write_bfr_file(options.filePath + '.internal.bfr')
        gl = {}
        for cc, gname in font.cc2g.items():
            if cc in font.cmap_internal.cc2unicode:
                gl[gname[1:]] = PdfTextString.Unicode_to_UTF16BE(font.cmap_internal.cc2unicode[cc])

        gl = {k:gl[k] for k in sorted(gl)}

        open(options.filePath + '.glyphlist.txt', 'wt').write(''.join(f'{k};{v}\n' for k,v in gl.items()))

    print(f'Output written to {outPath}')

    # pprint(font.info)
