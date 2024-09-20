#!/usr/bin/env python3

import string, sys, re

from pdfrw import PdfDict, PdfName, PdfArray, py23_diffs

from .common import err, msg, warn
from .pdffontcore14 import PdfFontCore14
from .pdffilter import PdfFilter

# =========================================================================== class PdfFontEncoding

class PdfFontEncoding:

    # --------------------------------------------------------------------------- __init__()

    def __init__(self,
                 name:str = None,
                 differences:list = None,
                 font:PdfDict = None,
                 ignoreBaseEncoding = False,
                 gid2gname:dict = None):
        '''Creates an instance of the PdfFontEncoding class from the name of the encoding,
        a /Differences list or from a PDF font dict.
        '''
        self.name = None
        self.isType3 = False
        self.baseEncoding = None

        # self.baseEncoding = '/WinAnsiEncoding'

        self.cc2glyphname = {} # A map from character codes (chars) to glyph names
        self.glyphname2cc = {} # A reverse map from glyph names to character codes (chars)

        if name != None:

            self.name = name
            self.cc2glyphname = PdfFontEncodingStandards.get_cc2glyphname(name)
            if self.cc2glyphname == None: err(f'invalid encoding name: {name}')

        if differences != None:

            self.name = '[/Differences]'
            self.cc2glyphname = PdfFontEncoding.differences_to_cc2glyphname(differences)

        if font != None:

            assert font.Subtype != '/Type0' # SIMPLE font

            self.isType3 = font.Subtype == '/Type3'
            self.cc2glyphname = {}

            fd = font.FontDescriptor
            isEmbedded = fd and (fd.FontFile or fd.FontFile2 or fd.FontFile3)
                        
            if isinstance(font.Encoding, PdfDict):

                # PDF Ref. v1.7 sec. 5.5.5:
                # The base encoding [is] the encoding from which the Differences
                # entry (if present) describes differences—specified as the name of a predefined
                # encoding MacRomanEncoding, MacExpertEncoding, or WinAnsiEncoding (see Appendix D).
                # If this entry is absent, the Differences entry describes differences from an implicit
                # base encoding. For a font program that is embedded in the PDF file, the implicit base
                # encoding is the font program’s built-in encoding, as described above and further
                # elaborated in the sections on specific font types below. Otherwise, for a nonsymbolic font,
                # it is StandardEncoding, and for a symbolic font, it is the font’s built-in encoding.

                if not ignoreBaseEncoding and not self.isType3:


                    self.baseEncoding = font.Encoding.BaseEncoding if font.Encoding.BaseEncoding \
                        else PdfFontCore14.built_in_encoding(font.BaseFont) if not isEmbedded \
                        else None

                    if self.baseEncoding:
                        self.cc2glyphname = PdfFontEncodingStandards.get_cc2glyphname(self.baseEncoding)
                    elif gid2gname:
                        self.cc2glyphname = {gid:PdfName(gname) for gid,gname in gid2gname.items()}
                        

                # PDF Ref. v1.7 sec. 5.5.5:
                # If the Encoding entry is a dictionary, the table is initialized with the entries from the
                # dictionary’s BaseEncoding entry. Any entries in the Differences array are used to update
                # the table. Finally, any undefined entries in the table are filled using StandardEncoding.
                # UPD: this refers to TrueType fonts; it's unclear how to proceed with other types,
                # so do likewise except fot Type3 fonts, for which we know for sure that the only
                # chars that are present in the font are those in the /Encoding entry.

                if font.Encoding.Differences != None:
                    differencesMap = PdfFontEncoding.differences_to_cc2glyphname(font.Encoding.Differences)
                    self.cc2glyphname = self.cc2glyphname | differencesMap
                    
                self.name = [font.Encoding.BaseEncoding, '/Differences' if font.Encoding.Differences != None else None]

            else:

                self.name = font.Encoding if font.Encoding \
                            else PdfFontCore14.built_in_encoding(font.BaseFont) if not isEmbedded \
                            else None

                if self.name:
                    self.cc2glyphname = PdfFontEncodingStandards.get_cc2glyphname(self.name)
                elif gid2gname:
                    self.cc2glyphname = {gid:PdfName(gname) for gid,gname in gid2gname.items()}


            # # Limit the map; ??? DO WE REALLY WANT THIS ???
            # if font.FirstChar != None and font.LastChar != None:
            #     first,last = int(font.FirstChar), int(font.LastChar)
            #     self.cc2glyphname = {cc:g for cc,g in self.cc2glyphname.items() if first <= ord(cc) <= last}


        # reset self.glyphname2cc
        self.reset_glyphname2cc()


    def differences_to_cc2glyphname(differences:list):
        '''
        Converts encoding differences list to an encoding map — a dictionary that maps from cc to glyph names (with slash).
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

    def cc2glyphname_to_differences(cc2gname:dict):
        '''
        Encodes a cc2glyphname map as an encoding differences list.
        Returns a tuple: (differencesList, firstChar, lastChar)
        '''
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
        return diff, firstChar, lastChar

    def conjugate(self, encoding:'PdfFontEncoding'):
        '''
        Using `self` and `encoding`, creates and returns a tuple (`reEncMap`, `diffEncoding`):

        * `reEncMap` is a map that maps character codes `cc --> reEncMap(cc)` in such a way that
        `self.cc2glyphname[cc] == encoding.cc2glyphname(reEncMap(cc))` for any `cc` for which both
        sides of the equality are defined.
        * `diffEncoding` is essentially equal to self with the mappings of all character codes
        such that `self.cc2glyphname` is in `encoding.cc2glyphname.values()` removed from it. So, in other words,
        if `E` is the image of the `encoding`, i.e. a set of all glyph names that the `encoding` maps to, then
        `diffEncoding` is that part of `self` that ony maps to glyph names that are not in `E`.
        '''
        reEncMap = {}
        cc2glyphname = {}

        for cc,gname in self.cc2glyphname.items():
            if gname in encoding.glyphname2cc:
                reEncMap[cc] = encoding.glyphname2cc[gname]
            else:
                cc2glyphname[cc] = gname

        diffEncoding = PdfFontEncoding()
        diffEncoding.cc2glyphname = cc2glyphname
        diffEncoding.reset_glyphname2cc()
        diffEncoding.isType3 = self.isType3
        return reEncMap, diffEncoding

    def reset_glyphname2cc(self):
        '''
        Call this function every time self.cc2glyphname changes
        '''
        self.glyphname2cc = PdfFontEncodingStandards.invert_cc2glyphname(self.cc2glyphname, self.name)

    def __str__(self):
        '''
        String representation of the class
        '''
        return f'{self.name}'


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

    def invert_cc2glyphname(cc2glyphname:dict, encodingName):
        '''
        Returns a glyphname2cc map (with cc as char) given a cc2glyphname map an encodingName.
        The inversion is non-trivial since cc2glyphname map can be many-to-one. In such cases,
        the inverted map points to the first occurrence of the value.
        Also, if encodingName == '/WinAnsiEncoding',
        the '/bullet' glyphName should be mapped to 0x95; see PDF Reference 1.7 Appendix D, page 1000.
        '''
        inverted = {}
        for cc,name in cc2glyphname.items():
            if name not in inverted: inverted[name] = cc
        if encodingName == '/WinAnsiEncoding':
            inverted['/bullet'] = chr(0x95)
        return inverted

    # --------------------------------------------------------------------------- encodingVectors

    encodingVectors = {}

    encodingVectors['/WinAnsiEncoding'] = (
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        'space', 'exclam', 'quotedbl', 'numbersign', 'dollar', 'percent', 'ampersand', 'quotesingle', 'parenleft', 'parenright', 'asterisk', 'plus', 'comma', 'hyphen', 'period', 'slash',
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'colon', 'semicolon', 'less', 'equal', 'greater', 'question',
        'at', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
        'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'bracketleft', 'backslash', 'bracketright', 'asciicircum', 'underscore',
        'grave', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'braceleft', 'bar', 'braceright', 'asciitilde', 'bullet',
        'Euro', 'bullet', 'quotesinglbase', 'florin', 'quotedblbase', 'ellipsis', 'dagger', 'daggerdbl', 'circumflex', 'perthousand', 'Scaron', 'guilsinglleft', 'OE', 'bullet', 'Zcaron', 'bullet',
        'bullet', 'quoteleft', 'quoteright', 'quotedblleft', 'quotedblright', 'bullet', 'endash', 'emdash', 'tilde', 'trademark', 'scaron', 'guilsinglright', 'oe', 'bullet', 'zcaron', 'Ydieresis',
        'space', 'exclamdown', 'cent', 'sterling', 'currency', 'yen', 'brokenbar', 'section', 'dieresis', 'copyright', 'ordfeminine', 'guillemotleft', 'logicalnot', 'hyphen', 'registered', 'macron',
        'degree', 'plusminus', 'twosuperior', 'threesuperior', 'acute', 'mu', 'paragraph', 'periodcentered', 'cedilla', 'onesuperior', 'ordmasculine', 'guillemotright', 'onequarter', 'onehalf', 'threequarters', 'questiondown',
        'Agrave', 'Aacute', 'Acircumflex', 'Atilde', 'Adieresis', 'Aring', 'AE', 'Ccedilla', 'Egrave', 'Eacute', 'Ecircumflex', 'Edieresis', 'Igrave', 'Iacute', 'Icircumflex', 'Idieresis',
        'Eth', 'Ntilde', 'Ograve', 'Oacute', 'Ocircumflex', 'Otilde', 'Odieresis', 'multiply', 'Oslash', 'Ugrave', 'Uacute', 'Ucircumflex', 'Udieresis', 'Yacute', 'Thorn', 'germandbls',
        'agrave', 'aacute', 'acircumflex', 'atilde', 'adieresis', 'aring', 'ae', 'ccedilla', 'egrave', 'eacute', 'ecircumflex', 'edieresis', 'igrave', 'iacute', 'icircumflex', 'idieresis',
        'eth', 'ntilde', 'ograve', 'oacute', 'ocircumflex', 'otilde', 'odieresis', 'divide', 'oslash', 'ugrave', 'uacute', 'ucircumflex', 'udieresis', 'yacute', 'thorn', 'ydieresis'
    )

    encodingVectors['/MacRomanEncoding'] = (
        None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, 'space', 'exclam',
        'quotedbl', 'numbersign', 'dollar', 'percent', 'ampersand',
        'quotesingle', 'parenleft', 'parenright', 'asterisk', 'plus', 'comma',
        'hyphen', 'period', 'slash', 'zero', 'one', 'two', 'three', 'four',
        'five', 'six', 'seven', 'eight', 'nine', 'colon', 'semicolon', 'less',
        'equal', 'greater', 'question', 'at', 'A', 'B', 'C', 'D', 'E', 'F',
        'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'bracketleft', 'backslash', 'bracketright',
        'asciicircum', 'underscore', 'grave', 'a', 'b', 'c', 'd', 'e', 'f',
        'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z', 'braceleft', 'bar', 'braceright',
        'asciitilde', None, 'Adieresis', 'Aring', 'Ccedilla', 'Eacute',
        'Ntilde', 'Odieresis', 'Udieresis', 'aacute', 'agrave', 'acircumflex',
        'adieresis', 'atilde', 'aring', 'ccedilla', 'eacute', 'egrave',
        'ecircumflex', 'edieresis', 'iacute', 'igrave', 'icircumflex',
        'idieresis', 'ntilde', 'oacute', 'ograve', 'ocircumflex', 'odieresis',
        'otilde', 'uacute', 'ugrave', 'ucircumflex', 'udieresis', 'dagger',
        'degree', 'cent', 'sterling', 'section', 'bullet', 'paragraph',
        'germandbls', 'registered', 'copyright', 'trademark', 'acute',
        'dieresis', None, 'AE', 'Oslash', None, 'plusminus', None, None, 'yen',
        'mu', None, None, None, None, None, 'ordfeminine', 'ordmasculine', None,
        'ae', 'oslash', 'questiondown', 'exclamdown', 'logicalnot', None, 'florin',
        None, None, 'guillemotleft', 'guillemotright', 'ellipsis', 'space', 'Agrave',
        'Atilde', 'Otilde', 'OE', 'oe', 'endash', 'emdash', 'quotedblleft',
        'quotedblright', 'quoteleft', 'quoteright', 'divide', None, 'ydieresis',
        'Ydieresis', 'fraction', 'currency', 'guilsinglleft', 'guilsinglright',
        'fi', 'fl', 'daggerdbl', 'periodcentered', 'quotesinglbase',
        'quotedblbase', 'perthousand', 'Acircumflex', 'Ecircumflex', 'Aacute',
        'Edieresis', 'Egrave', 'Iacute', 'Icircumflex', 'Idieresis', 'Igrave',
        'Oacute', 'Ocircumflex', None, 'Ograve', 'Uacute', 'Ucircumflex',
        'Ugrave', 'dotlessi', 'circumflex', 'tilde', 'macron', 'breve',
        'dotaccent', 'ring', 'cedilla', 'hungarumlaut', 'ogonek', 'caron')

    encodingVectors['/SymbolEncoding']=(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 'space',
        'exclam', 'universal', 'numbersign', 'existential', 'percent', 'ampersand', 'suchthat',
        'parenleft', 'parenright', 'asteriskmath', 'plus', 'comma', 'minus', 'period', 'slash', 'zero',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'colon', 'semicolon',
        'less', 'equal', 'greater', 'question', 'congruent', 'Alpha', 'Beta', 'Chi', 'Delta', 'Epsilon',
        'Phi', 'Gamma', 'Eta', 'Iota', 'theta1', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Omicron', 'Pi', 'Theta',
        'Rho', 'Sigma', 'Tau', 'Upsilon', 'sigma1', 'Omega', 'Xi', 'Psi', 'Zeta', 'bracketleft',
        'therefore', 'bracketright', 'perpendicular', 'underscore', 'radicalex', 'alpha', 'beta', 'chi',
        'delta', 'epsilon', 'phi', 'gamma', 'eta', 'iota', 'phi1', 'kappa', 'lambda', 'mu', 'nu',
        'omicron', 'pi', 'theta', 'rho', 'sigma', 'tau', 'upsilon', 'omega1', 'omega', 'xi', 'psi', 'zeta',
        'braceleft', 'bar', 'braceright', 'similar', None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, 'Euro', 'Upsilon1', 'minute', 'lessequal',
        'fraction', 'infinity', 'florin', 'club', 'diamond', 'heart', 'spade', 'arrowboth', 'arrowleft',
        'arrowup', 'arrowright', 'arrowdown', 'degree', 'plusminus', 'second', 'greaterequal', 'multiply',
        'proportional', 'partialdiff', 'bullet', 'divide', 'notequal', 'equivalence', 'approxequal',
        'ellipsis', 'arrowvertex', 'arrowhorizex', 'carriagereturn', 'aleph', 'Ifraktur', 'Rfraktur',
        'weierstrass', 'circlemultiply', 'circleplus', 'emptyset', 'intersection', 'union',
        'propersuperset', 'reflexsuperset', 'notsubset', 'propersubset', 'reflexsubset', 'element',
        'notelement', 'angle', 'gradient', 'registerserif', 'copyrightserif', 'trademarkserif', 'product',
        'radical', 'dotmath', 'logicalnot', 'logicaland', 'logicalor', 'arrowdblboth', 'arrowdblleft',
        'arrowdblup', 'arrowdblright', 'arrowdbldown', 'lozenge', 'angleleft', 'registersans',
        'copyrightsans', 'trademarksans', 'summation', 'parenlefttp', 'parenleftex', 'parenleftbt',
        'bracketlefttp', 'bracketleftex', 'bracketleftbt', 'bracelefttp', 'braceleftmid', 'braceleftbt',
        'braceex', None, 'angleright', 'integral', 'integraltp', 'integralex', 'integralbt',
        'parenrighttp', 'parenrightex', 'parenrightbt', 'bracketrighttp', 'bracketrightex',
        'bracketrightbt', 'bracerighttp', 'bracerightmid', 'bracerightbt', None)

    encodingVectors['/ZapfDingbatsEncoding'] = (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        'space', 'a1', 'a2', 'a202', 'a3', 'a4', 'a5', 'a119', 'a118', 'a117', 'a11', 'a12', 'a13', 'a14',
        'a15', 'a16', 'a105', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27',
        'a28', 'a6', 'a7', 'a8', 'a9', 'a10', 'a29', 'a30', 'a31', 'a32', 'a33', 'a34', 'a35', 'a36',
        'a37', 'a38', 'a39', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45', 'a46', 'a47', 'a48', 'a49', 'a50',
        'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a60', 'a61', 'a62', 'a63', 'a64',
        'a65', 'a66', 'a67', 'a68', 'a69', 'a70', 'a71', 'a72', 'a73', 'a74', 'a203', 'a75', 'a204', 'a76',
        'a77', 'a78', 'a79', 'a81', 'a82', 'a83', 'a84', 'a97', 'a98', 'a99', 'a100', None, 'a89', 'a90',
        'a93', 'a94', 'a91', 'a92', 'a205', 'a85', 'a206', 'a86', 'a87', 'a88', 'a95', 'a96', None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, 'a101', 'a102', 'a103', 'a104', 'a106', 'a107', 'a108', 'a112', 'a111', 'a110', 'a109',
        'a120', 'a121', 'a122', 'a123', 'a124', 'a125', 'a126', 'a127', 'a128', 'a129', 'a130', 'a131',
        'a132', 'a133', 'a134', 'a135', 'a136', 'a137', 'a138', 'a139', 'a140', 'a141', 'a142', 'a143',
        'a144', 'a145', 'a146', 'a147', 'a148', 'a149', 'a150', 'a151', 'a152', 'a153', 'a154', 'a155',
        'a156', 'a157', 'a158', 'a159', 'a160', 'a161', 'a163', 'a164', 'a196', 'a165', 'a192', 'a166',
        'a167', 'a168', 'a169', 'a170', 'a171', 'a172', 'a173', 'a162', 'a174', 'a175', 'a176', 'a177',
        'a178', 'a179', 'a193', 'a180', 'a199', 'a181', 'a200', 'a182', None, 'a201', 'a183', 'a184',
        'a197', 'a185', 'a194', 'a198', 'a186', 'a195', 'a187', 'a188', 'a189', 'a190', 'a191', None)

    encodingVectors['/StandardEncoding']=(
        None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,
        None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,
        "space","exclam","quotedbl","numbersign","dollar","percent","ampersand","quoteright","parenleft","parenright","asterisk","plus","comma","hyphen","period","slash",
        "zero","one","two","three","four","five","six","seven","eight","nine","colon","semicolon","less","equal","greater","question",
        "at","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O",
        "P","Q","R","S","T","U","V","W","X","Y","Z","bracketleft","backslash","bracketright","asciicircum","underscore",
        "quoteleft","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o",
        "p","q","r","s","t","u","v","w","x","y","z","braceleft","bar","braceright","asciitilde",None,
        None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,
        None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,
        None,"exclamdown","cent","sterling","fraction","yen","florin","section","currency","quotesingle","quotedblleft","guillemotleft","guilsinglleft","guilsinglright","fi","fl",
        None,"endash","dagger","daggerdbl","periodcentered",None,"paragraph","bullet","quotesinglbase","quotedblbase","quotedblright","guillemotright","ellipsis","perthousand",None,"questiondown",
        None,"grave","acute","circumflex","tilde","macron","breve","dotaccent","dieresis",None,"ring","cedilla",None,"hungarumlaut","ogonek","caron",
        "emdash",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,
        None,"AE",None,"ordfeminine",None,None,None,None,"Lslash","Oslash","OE","ordmasculine",None,None,None,None,
        None,"ae",None,None,None,"dotlessi",None,None,"lslash","oslash","oe","germandbls",None,None,None,None
    )

    encodingVectors['/PDFDocEncoding']=(None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,
        None,None,None,None,None,"breve","caron","circumflex",
        "dotaccent","hungarumlaut","ogonek","ring","tilde","space","exclam","quotedbl","numbersign","dollar","percent",
        "ampersand","quotesingle","parenleft","parenright","asterisk","plus","comma","hyphen","period","slash","zero",
        "one","two","three","four","five","six","seven","eight","nine","colon","semicolon","less","equal","greater",
        "question","at","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X",
        "Y","Z","bracketleft","backslash","bracketright","asciicircum","underscore","grave","a","b","c","d","e","f","g",
        "h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","braceleft","bar","braceright",
        "asciitilde",None,"bullet","dagger","daggerdbl","ellipsis","emdash","endash","florin","fraction","guilsinglleft",
        "guilsinglright","minus","perthousand","quotedblbase","quotedblleft","quotedblright","quoteleft","quoteright",
        "quotesinglbase","trademark","fi","fl","Lslash","OE","Scaron","Ydieresis","Zcaron","dotlessi","lslash","oe",
        "scaron","zcaron",None,"Euro","exclamdown","cent","sterling","currency","yen","brokenbar","section","dieresis",
        "copyright","ordfeminine","guillemotleft","logicalnot",None,"registered","macron","degree","plusminus","twosuperior",
        "threesuperior","acute","mu","paragraph","periodcentered","cedilla","onesuperior","ordmasculine","guillemotright",
        "onequarter","onehalf","threequarters","questiondown","Agrave","Aacute","Acircumflex","Atilde","Adieresis","Aring",
        "AE","Ccedilla","Egrave","Eacute","Ecircumflex","Edieresis","Igrave","Iacute","Icircumflex","Idieresis","Eth",
        "Ntilde","Ograve","Oacute","Ocircumflex","Otilde","Odieresis","multiply","Oslash","Ugrave","Uacute","Ucircumflex",
        "Udieresis","Yacute","Thorn","germandbls","agrave","aacute","acircumflex","atilde","adieresis","aring","ae",
        "ccedilla","egrave","eacute","ecircumflex","edieresis","igrave","iacute","icircumflex","idieresis","eth","ntilde",
        "ograve","oacute","ocircumflex","otilde","odieresis","divide","oslash","ugrave","uacute","ucircumflex","udieresis",
        "yacute","thorn","ydieresis")

    encodingVectors['/MacExpertEncoding'] =  (None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        'space', 'exclamsmall', 'Hungarumlautsmall', 'centoldstyle', 'dollaroldstyle', 'dollarsuperior', 'ampersandsmall',
        'Acutesmall', 'parenleftsuperior', 'parenrightsuperior', 'twodotenleader', 'onedotenleader', 'comma', 'hyphen',
        'period', 'fraction', 'zerooldstyle', 'oneoldstyle', 'twooldstyle', 'threeoldstyle', 'fouroldstyle',
        'fiveoldstyle', 'sixoldstyle', 'sevenoldstyle', 'eightoldstyle', 'nineoldstyle', 'colon', 'semicolon', None,
        'threequartersemdash', None, 'questionsmall', None, None, None, None, 'Ethsmall', None, None, 'onequarter',
        'onehalf', 'threequarters', 'oneeighth', 'threeeighths', 'fiveeighths', 'seveneighths', 'onethird', 'twothirds',
        None, None, None, None, None, None, 'ff', 'fi', 'fl', 'ffi', 'ffl', 'parenleftinferior', None,
        'parenrightinferior', 'Circumflexsmall', 'hypheninferior', 'Gravesmall', 'Asmall', 'Bsmall', 'Csmall', 'Dsmall',
        'Esmall', 'Fsmall', 'Gsmall', 'Hsmall', 'Ismall', 'Jsmall', 'Ksmall', 'Lsmall', 'Msmall', 'Nsmall', 'Osmall',
        'Psmall', 'Qsmall', 'Rsmall', 'Ssmall', 'Tsmall', 'Usmall', 'Vsmall', 'Wsmall', 'Xsmall', 'Ysmall', 'Zsmall',
        'colonmonetary', 'onefitted', 'rupiah', 'Tildesmall', None, None, 'asuperior', 'centsuperior', None, None, None,
        None, 'Aacutesmall', 'Agravesmall', 'Acircumflexsmall', 'Adieresissmall', 'Atildesmall', 'Aringsmall',
        'Ccedillasmall', 'Eacutesmall', 'Egravesmall', 'Ecircumflexsmall', 'Edieresissmall', 'Iacutesmall', 'Igravesmall',
        'Icircumflexsmall', 'Idieresissmall', 'Ntildesmall', 'Oacutesmall', 'Ogravesmall', 'Ocircumflexsmall',
        'Odieresissmall', 'Otildesmall', 'Uacutesmall', 'Ugravesmall', 'Ucircumflexsmall', 'Udieresissmall', None,
        'eightsuperior', 'fourinferior', 'threeinferior', 'sixinferior', 'eightinferior', 'seveninferior', 'Scaronsmall',
        None, 'centinferior', 'twoinferior', None, 'Dieresissmall', None, 'Caronsmall', 'osuperior', 'fiveinferior', None,
        'commainferior', 'periodinferior', 'Yacutesmall', None, 'dollarinferior', None, None, 'Thornsmall', None,
        'nineinferior', 'zeroinferior', 'Zcaronsmall', 'AEsmall', 'Oslashsmall', 'questiondownsmall', 'oneinferior',
        'Lslashsmall', None, None, None, None, None, None, 'Cedillasmall', None, None, None, None, None, 'OEsmall',
        'figuredash', 'hyphensuperior', None, None, None, None, 'exclamdownsmall', None, 'Ydieresissmall', None,
        'onesuperior', 'twosuperior', 'threesuperior', 'foursuperior', 'fivesuperior', 'sixsuperior', 'sevensuperior',
        'ninesuperior', 'zerosuperior', None, 'esuperior', 'rsuperior', 'tsuperior', None, None, 'isuperior', 'ssuperior',
        'dsuperior', None, None, None, None, None, 'lsuperior', 'Ogoneksmall', 'Brevesmall', 'Macronsmall', 'bsuperior',
        'nsuperior', 'msuperior', 'commasuperior', 'periodsuperior', 'Dotaccentsmall', 'Ringsmall', None, None, None, None)
