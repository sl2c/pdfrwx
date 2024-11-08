#!/usr/bin/env python3

import string, re
from pprint import pprint

from pdfrw import PdfDict, PdfName, PdfArray

from .common import err, msg, warn, chain
from .pdffontencoding import PdfFontEncoding
from .pdffontcmap import PdfFontCMap
from .pdfobjects import PdfObjects

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

    def __init__(self, glyphListPaths:list[str] = [], fonts:dict = {}, knownPrefixes = {}):
        '''Initializes PdfFontGlyphMap by doing the following:
        
        * loads glyph map lists from the list of glyph list paths (1st argument)
        and sets self.glyphMap (the map from glyph names to Unicode);
        the glyph map lists should be in the Adobe Glyph List (AGL) format: https://github.com/adobe-type-tools/agl-aglfn;
        * trains the composite_glyphname_to_cc() algo on glyph names from the fonts list;
        * self.knownPrefixes is updated based on knownPrefixes, which is a map from prefixes (strings)
        to the corresponding suffix type (see class CompositeNames), for example: {'uni':'hex', 'c':'dex', ...}.

        For more info on composite names, please see the help for PdfFontGlyphMap.composite_glyphname_to_cc()

        All three arguments may be safely omitted.
        '''
        self.glyphMap = {} # a map from the (Adobe) standard glyph names to Unicode points
        self.prefix_types = {} # a map from prefixes to suffix types for composite glyph names

        # print(f"[GLYPH MAPS]: {glyphListPaths}")

        # Initialize self.glyphMap
        for path in glyphListPaths:
            with open(path, 'r') as file:
                for k,line in enumerate(file):
                    s = line.strip(' \t\r\n')
                    if s == '' or s[0] == '#': continue
                    try: glyphName, codePointHex = re.split(';',s)
                    except: err(f'malformed line # {k} in a glyph list: {path}')
                    codePointHex = re.sub(r'[^0-9a-fA-F].*$','',codePointHex) # remove non-hex chars from 1st occurrence to end
                    if len(codePointHex) < 4: err(f'malformed line # {k} in a glyph list: {path}')
                    self.glyphMap[PdfName(glyphName)] = PdfFontCMap.UTF16BE_to_Unicode_NEW(codePointHex)

        # train the composite_glyphname_to_cc() algo on glyph names from the fonts list
        for font in [f for f in fonts.values() if f.font.Subtype != '/Type0']:
            for gname in [g for g in font.encoding.glyphname2cc 
                            if PdfFontGlyphMap.strip_dot_endings(g) not in self.glyphMap]:
                self.composite_glyphname_to_unicode(gname)

        # add prefix to prefix type mapping from the knownPrefixes argument
        for prefix,t in knownPrefixes.items():
            self.prefix_types[prefix] = PdfFontGlyphMap.HEX if t.lower() == 'hex' else PdfFontGlyphMap.DEX

    # --------------------------------------------------------------------------- composite_glyphname_to_unicode()

    def composite_glyphname_to_unicode(self, gname:str):
        ''' For a glyph name of the composite 'prefix + number' form, e.g. 'c13', 'glyph10H', etc.,
        where prefix is of the form: '[a-zA-Z]|#|FLW|uni|Char|glyph|MT|.*\.g' and suffix is a DEX (decimal/hex)
        number: '[0-9a-fA-F]+' returns the number part as int by interpreting the corresponding string
        part as a hex or dec number based on the statistics of previous encounters with the glyph names
        of this form. The usage scenario is to first run this function through all available glyph names
        to train the algorithm, and then call it on any particular glyph name to get results.
        '''
        suffix_type = lambda suffix: self.DEX if all(c in string.digits for c in suffix) else self.HEX

        gname_marked = re.sub(r'^([a-zA-Z]|#|FLW|uni|cid|Char|char|glyph|MT|.*\.g)([0-9a-fA-F]+)$',r'\1|||\2',gname)
        gname_split = re.split(r'\|\|\|',gname_marked)
        prefix,suffix = gname_split if len(gname_split) == 2 else (None,None)
        if prefix == None: return None

        suffix_t = suffix_type(suffix) if prefix != 'uni' else self.HEX
        if prefix not in self.prefix_types or suffix_t > self.prefix_types[prefix]:
            self.prefix_types[prefix] = suffix_t

        i = int(suffix,16) if self.prefix_types[prefix] == self.HEX else int(suffix)
        try: return chr(i)
        except: return None

    # --------------------------------------------------------------------------- make_cmap()

    def make_cmap(self,
                  encoding:PdfFontEncoding,
                  mapComposites = True,
                  quiet:bool = False,
                  explicitMap = {}):
        '''
        Create an instance of PdfFontCMap from an instance of PdfFontEncoding by attempting to map
        glyph names to Unicode points.
        '''
        if encoding == None: return None
        stdGlyphMap = PdfFontGlyphMapStandards.get_glyphName2unicodeMap(encoding.name)

        cmap = PdfFontCMap()
        unrecognized = {}

        # Count the number of 'weird' glyphs - glyphs whose names start with '/;'
        # weird_count = len([v for v in encoding.cc2glyphname.values() if v[:2] == '/;'])

        for cc, gname in encoding.cc2glyphname.items():

            if gname == '/.notdef': continue
 
            unicode = self.gname_to_unicode(gname,
                                                stdGlyphMap = stdGlyphMap,
                                                isType3 = encoding.isType3,
                                                mapComposites = mapComposites,
                                                explicitMap = explicitMap)
            if unicode != None:
                cmap.cc2unicode[cc] = unicode
            else:
                unrecognized[ord(cc)] = gname

        cmap.reset_unicode2cc()
        if len(unrecognized) > 0 and not quiet:
            u = ' '.join(unrecognized.values())
            warn(f'unrecognized glyph names: {u}')
        return cmap

    def gname_to_unicode(self,
                         gname,
                         stdGlyphMap:dict = {},
                         isType3 = False,
                         mapComposites = True,
                         explicitMap = {}):
        '''
        Convert glyph name to Unicode
        '''

        unicode = None
            
        if isType3:

            t3map1 = {c:i for i,c in enumerate('ABCDEFGH')}
            t3map2 = {c:i for i,c in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
            # semicolons = {';':';', ';;':chr(0), ';;;;':chr(0), ';;;;;;;;':';'}
            semicolons = {';':chr(0), ';;':';', ';;;;':chr(0), ';;;;;;;;':';'}

            if gname in explicitMap:
                unicode = explicitMap[gname]
                warn(f'explicit mapping {gname} --> {[unicode]}')
            elif gname == '/BnZr':
                unicode = chr(0) # This is how chr(0) is sometimes denoted in Type3 fonts
            elif gname[1:] in semicolons:
                unicode = semicolons[gname[1:]]
                warn(f'mapping {gname} --> {[unicode]}')
            elif len(gname) == 3:
                try: unicode = chr(t3map1[gname[1]] * 36 + t3map2[gname[2]])
                except: unicode = None
            elif len(gname) == 4 and gname[1] == '#':
                try: unicode = chr(int(gname[2:],16))
                except: unicode = None
                    
        if unicode == None:

            g = PdfFontGlyphMap.strip_dot_endings(gname)
            g = PdfFontGlyphMap.strip_prefixes(g)
            unicode = self.glyphMap.get(g) or stdGlyphMap.get(g) # Check external glyph map first

            if unicode == None and mapComposites:

                composite = g[1:]
                unicode = chr(int(composite)) if all(c in string.digits for c in composite) \
                    else composite if len(composite) == 1 \
                    else self.composite_glyphname_to_unicode(composite)

        return unicode


    # --------------------------------------------------------------------------- reencode_cmap()

    def reencode_cmap(self, cmap:PdfFontCMap, encoding:PdfFontEncoding, direct=False):
        '''
        Returns a reencoded version of cmap based on the difference between encoding and baseEncoding.
        '''
        cmapRe = PdfFontCMap()

        baseEncoding = PdfFontEncoding(encoding.baseEncoding or '/WinAnsiEncoding')

        if encoding.isType3:
            cmapRe.cc2unicode = chain(self.make_cmap(encoding).cc2unicode, cmap.cc2unicode)
        else:
            reEnc, diffEncoding = encoding.conjugate(baseEncoding)
            if not direct: reEnc = chain(reEnc, cmap.cc2unicode)
            self.make_cmap(diffEncoding).cc2unicode
            diffEncMap = self.make_cmap(diffEncoding).cc2unicode
            if not direct: diffEncMap = chain(diffEncMap, cmap.cc2unicode)
            cmapRe.cc2unicode = reEnc | diffEncMap

        cmapRe.reset_unicode2cc()
        return cmapRe

    # --------------------------------------------------------------------------- strip_dot_endings()

    def strip_dot_endings(s:str):
        '''Strips ._, .sc & .cap endings from a string; useful to match variants (small caps etc) of glyph names in glyph lists
        '''
        s1 = re.sub(r'(\.(_|sc|cap|alt[0-9]*|disp|big|small|ts1|lf|swash))+$','', s)
        s1 = re.sub(r'\\rm', '', s1)
        return s1 if len(s1)>1 else s

    # --------------------------------------------------------------------------- strip_prefixes()

    def strip_prefixes(s:str):
        '''Strips \\rm and the like from the beginning of the glyph name
        '''
        s1 = re.sub(r'(\\rm|\\mathchardef)', '', s)
        return s1 if len(s1)>1 else s


# =========================================================================== class PdfFontGlyphMapStandards

class PdfFontGlyphMapStandards:

    def get_glyphName2unicodeMap(encodingName):
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
        encodingName = encodingName \
            if not isinstance(encodingName, PdfArray) and not isinstance(encodingName, list) \
            else encodingName[0]
        glyphName2unicodeMapUnion = PdfFontGlyphMapStandards.glyphName2unicodeMaps['LatinSet'] \
                    | PdfFontGlyphMapStandards.glyphName2unicodeMaps['MacExpertSet'] \
                    | PdfFontGlyphMapStandards.glyphName2unicodeMaps['SymbolSet']
        glyphname2unicode =  PdfFontGlyphMapStandards.glyphName2unicodeMaps[encNameToEncSet[encodingName]] \
                                if isinstance(encodingName,str) and encodingName in encNameToEncSet \
                                else glyphName2unicodeMapUnion
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
