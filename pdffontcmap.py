#!/usr/bin/env python3

import re

from pdfrw import IndirectPdfDict, py23_diffs

from pdfrwx.common import err, warn, msg
from pdfrwx.pdffilter import PdfFilter



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

        stream = ToUnicode.stream
        if stream == None: warn(f'no stream in font\'s ToUnicode object: {ToUnicode}') ; self.set_to_identity_map() ; return

        if ToUnicode.Filter != None:
            # stream = PdfFilter.uncompress(ToUnicode).stream
            try:
                stream = PdfFilter.uncompress(ToUnicode).stream
            except Exception as e:
                s = ToUnicode.stream
                ss = s[:10] + '..' + s[-10:]
                ee = f"failed to decompress the font's ToUnicode CMap: {ToUnicode}, stream: {repr(ss)}"
                warn(e); warn(ee); raise ValueError(ee)
                
        self.cc2unicode = {}

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
                    u = PdfFontCMap.UTF16BE_to_Unicode_NEW(t)
                    # if u == None: u = chr(0) # This is a quick and dirty fix for errors in existing ToUnicode CMaps
                    if u == None: warn('bad token in ToUnicode CMap: ' + t)
                    if isValueList: valueList.append(u if u != None else chr(0))
                    else: rangeList.append(u if u != None else chr(0))

            if len(rangeList) % 3 != 0: err(f'bfrange: not a whole number of triplets in a bfrange block: {block}')
 
            for i in range(0,len(rangeList),3):
                start,end,u = rangeList[i:i+3]
                if isinstance(u,list) and len(u) == 1: u = u[0] # This seems weird, but this is how it is
                if isinstance(u,str) and len(u) != 1 and start != end:
                    err(f'bfrange: an incremental bfrange cannot start with a ligature: {block} @ {i}')
                start,end = ord(start),ord(end)
                for k in range(start,end+1):
                    self.cc2unicode[chr(k)] = u[k - start] if isinstance(u,list) \
                                        else u if start == end \
                                        else chr(ord(u) + k - start)
    
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
                    u = PdfFontCMap.UTF16BE_to_Unicode_NEW(t)
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
                else: u = chr(0)

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
        self.cc2unicode = {}
        start,start2, stop = -1,-1,-1
        with open(bfrFilePath, 'r') as f:
            for line in f:
                line = re.sub(r'#[ ]+.*', '', line) # strip comments
                line = line.rstrip('\r\n ') # strip end-of-line chars
                if line == '': continue
                if line[0] == '<' and line[-1]== '>':
                    try:
                        lineSplit = re.split(r'>\s*<',line.strip('<>'))
                        if len(lineSplit) == 3:
                            start,stop,hex = lineSplit
                            start,stop = int(start,16),int(stop,16)
                            u = PdfFontCMap.UTF16BE_to_Unicode_NEW(hex)
                            if start == stop: self.cc2unicode[chr(start)] = u
                            elif len(u) != 1: err('starting hex value in a range is multi-char')
                            else:
                                for cc in range(start,stop+1):
                                    self.cc2unicode[chr(cc)] = chr(ord(u)+cc-start)
                            start2 = ((stop // 16 ) + 1) * 16
                        elif len(lineSplit) == 1:
                            start2 = int(lineSplit[0],16)
                            if start2 % 16 != 0: err(f'start position is not a multiple of 16: {lineSplit[0]}')
                        else:
                            err(f'bad line in a bfr file: {line}')
                    except:
                        err(f'bad line in a bfr file: {line}')
                elif len(line) <= 16 and start2 != -1:
                    if len(line) < 16: line = line + ' '*(16-len(line))
                    for i in range(len(line)):
                        if line[i] != '.': self.cc2unicode[chr(start2 + i)] = line[i]
                    start2 += 16
                else:
                    err(f'bad line in a bfr file: {line}')
        self.reset_unicode2cc()

    # --------------------------------------------------------------------------- write_bfr_file()

    def write_bfr_file(self, bfrFilePath:str):
        '''
        Write CMap to a bfr file using the compact format
        '''
        cc2u = self.cc2unicode
        s = "# dots '.' mean '.notdef'; to encode the dot per se, use e.g.: <002E><002E><002E>\n"
        skip = False
        keys = sorted(cc2u.keys())
        if len(keys) == 0: return s

        special = [i for i in range(32)] + [i for i in range(0x7f,0xa0)] + [0xad]
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
            if skip or row % 16 == 0: s += f'<{row*16:04X}>\n'
            s += line + '\n'
            skip = False
 
        dotLines = ''.join(f'<{ord(cc):04X}><{ord(cc):04X}><002E>\n' for cc in cc2u if cc2u[cc] == '.')
        if dotLines != '':
            s += '# The dot\n' + dotLines

        if len(triples) > 0:
            s += '# Special & multibyte chars\n'
            for cc,u in triples.items():
                s += f'<{ord(cc):04X}><{ord(cc):04X}><{PdfFontCMap.Unicode_to_UTF16BE(u)}>\n'

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
        If impose == True, composer mappings for cids that are not in self.cmap are added to the result.
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

    # --------------------------------------------------------------------------- UTF16BE_to_Unicode_NEW()

    def UTF16BE_to_Unicode_NEW(hexStr:str):
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

    def Unicode_to_UTF16BE(unicodeStr:str):
        '''Converts a Unicode string to a utf-16be-encoded hex string
        '''
        # return unicodeStr.encode('utf-16-be','surrogatepass').hex()
        return unicodeStr.encode('utf-16-be').hex().upper()

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

    def to_bfranges(self):
        '''Encode CMap as bfranges (list of 3-tuples of ints)
        '''
        m = self.cc2unicode
        if len(m) == 0: return []
        start,stop,result = None,None,[]
        for k in sorted(m):
            if m[k] == None: continue
            if len(m[k]) > 1:
                if start != None:
                    result.append((ord(start),ord(stop),ord(m[start])))
                    start = None
                result.append((ord(k),ord(k),[ord(x) for x in m[k]]))
                continue
            if start == None: start = k; stop = k
            elif ord(k)//256 == ord(stop)//256 and ord(k)-ord(stop) == 1 and ord(m[k])-ord(m[stop]) == 1: stop = k
            else: result.append((ord(start),ord(stop),ord(m[start]))); start = k; stop = k
        if start != None: result.append((ord(start),ord(stop),ord(m[start])))
        return result

    # --------------------------------------------------------------------------- bfranges_to_stream()

    def bfranges_to_stream(bfranges:list, isCid:bool):
        '''Convert CMap encoded as bfranges (list of 3-tuples of ints) to stream (str)
        '''
        w = 4 if isCid else 2
        h = lambda i,width: f'{i:0{width}X}' # convert int i to a hex-string of specified width

        result = []
        for start,stop,value in bfranges:
            if not isCid and (start >= 256 or stop >= 256): continue
            value_unicode = ''.join(chr(x) for x in value) if isinstance(value,list) else chr(value)
            value_utf16 = PdfFontCMap.Unicode_to_UTF16BE(value_unicode)
            v = '<' + value_utf16 + '>' if len(value_unicode) == 1 else '[<' + value_utf16 + '>]'
            result.append(f'<{h(start,w)}><{h(stop,w)}>{v}')
        return '\n'.join(result)


    # --------------------------------------------------------------------------- write_pdf_stream()

    def write_pdf_stream(self, CMapName:str, isCID:bool):
        '''Creates a PDF ToUnicode CMap dictionary stream (see section 9.10.3 ToUnicode CMaps of PDF 1.6 Spec).
        The argument CMapName is the value of the /CMapName entry of the produced stream and should
        be a PdfName itself (i.e., it should start with a slash)
        '''
        if CMapName[0] != '/' or ' ' in CMapName: err(f'invalid CMapName: {CMapName}')
        bfranges = self.to_bfranges()
        if len(bfranges) == 0: return None
        bfrStream = PdfFontCMap.bfranges_to_stream(bfranges,isCID)
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

