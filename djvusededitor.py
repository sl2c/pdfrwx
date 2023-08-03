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

# ========================================================================== class DjVuSedEditor

class DjVuSedEditor:

    def __init__(self, djvuSedStream:str):
        '''Parses the djvuSedStream and stores the resulting djvused page tree in self.tree'''
        self.stream = djvuSedStream
        lexer,parser = DjVuSedLexer(),DjVuSedParser()
        tokens = lexer.tokenize(self.stream)
        # for tok in tokens: print(tok)
        # sys.exit()
        self.tree = parser.parse(tokens)
        # pprint.pprint(self.tree)
        # sys.exit()

    def djvuSedStringToUnicode(self, djvuSedString:str):
        '''Convert an ASCII-encoded string from a djvused stream to a Unicode string
        '''
        s = djvuSedString
        if len(s) < 2 or s[0] != '"' or s[-1] != '"': err(f'invalid stirng: {s}')
        return s[1:-1].encode('latin1').decode('unicode-escape').encode('latin1').decode('utf-8')

    def unicodeToDjvuSedString(self, unicodeStr:str):
        '''Convert a Unicode string to am ASCII-encoded string (non-ASCII chars are octal-escaped as \\123)
        to be used in a djvused stream
        '''
        # return ''.join(chr(b) if b<128 and b>=32 and b!=134 and b!=42 else f'\\{b:o}' for b in unicodeStr.encode('utf-8'))
        # No idea why the "and b!=134 and b!=42" part was inserted
        return ''.join(chr(b) if b<128 and b>=32 else f'\\{b:o}' for b in unicodeStr.encode('utf-8'))

    # def djvusedPageTreeToPDFStream(self, djvusedPageTree:list, font:PdfFont, fontsize = None, baseline_y = None, debug = False):
    def djvusedPageTreeToPDFStream(self, djvusedPageTree:list, font:PdfFont, baseline_y = None, scale_y = None):

        '''Convert annotations in the parsed djvusedPageTree to a PDF stream.
        The djvusedPageTree can be an arbitrary set of annotation commands that relates to a single page, that is
        anything between two consecutive "select pageNo" commands in the original djvused stream.'''

        stream = ''
        for chunk in djvusedPageTree:
            if len(chunk) != 6: err(f'set-txt chunk has #elements != 6: {chunk}')
            type,xmin,ymin,xmax,ymax,text = chunk # note that always ymax < ymin, this is just bad naming
            if type == 'line':
                baseline_y = int(round(0.2*(4*int(ymax) + 1*int(ymin))))
                scale_y = int(ymin)- int(ymax)
            if isinstance(text,str) and len(text)>0:

                textNormalized = text
                textNormalized = self.unicodeToDjvuSedString(text) # This extends djvused format to allow Unicode strings
                textUnicode = self.djvuSedStringToUnicode(textNormalized)
                if len(textUnicode) == 0: continue
                hexString = font.encode(textUnicode)
                stringWidth = font.width(textUnicode)

                scale_x = int(round(font.scaleFactor*(float(xmax)-float(xmin))/stringWidth))

                # if fontsize != None and fontsize_new > fontsize and len(textUnicode)<3:
                #     fontsize_new = fontsize
                if baseline_y == None:  baseline_y = int(round(0.2*(4*int(ymax) + 1*int(ymin))))
                if scale_y == None: scale_y = int(ymin)-int(ymax)

                # stream += f'BT /OCR {fontsize_new} Tf {xmin} {baseline_y:.1f} Td {invisible}{hexString} Tj ET\n'
                stream += f'{scale_x} 0 0 {scale_y} {xmin} {baseline_y} Tm {hexString} Tj\n'

                # if len(textUnicode)>=5: fontsize = fontsize_new

            elif isinstance(text,list):
                # if type == 'line': # test run to determine fontsize
                #     streamDelta,fontsize = self.djvusedPageTreeToPDFStream(chunk[5],font,None,baseline_y,debug)
                # streamDelta,fontsize = self.djvusedPageTreeToPDFStream(chunk[5],font,fontsize,baseline_y,debug)
                # stream += streamDelta
                stream += self.djvusedPageTreeToPDFStream(chunk[5], font, baseline_y, scale_y)
            elif text is None:
                continue
            else:
                err('set-txt: chunk\'s last element is neither of: string/list/None: {chunk}')
        # return stream, fontsize
        return stream

    def insert_ocr(self, pdf:PdfReader, firstPage = 1, debug = False):

        # One font, many pages
        utils = PdfFontUtils()
        font = utils.loadFont([defaultUnicodeFont], ['.',defaultFontDir])

        # Run over pages in the djvusedTree
        for djvusedPage in self.tree:

            if djvusedPage[0] == 'save': continue
            if djvusedPage[0] != 'select': err(f'expected select, got: {djvusedPage[0]}')
            try: pageNo = int(djvusedPage[1]) + firstPage - 1
            except: pageNo = firstPage

            if pageNo < 1 or pageNo > len(pdf.pages):
                err(f'pageNo ({pageNo}) is outside pdf page range (1-{len(pdf.pages)})')
            eprint(f'inserting OCR in page {pageNo}')
            pdfPage = pdf.pages[pageNo-1]

            # Remove old OCR
            stream = ''.join(PdfFilter.uncompress(c).stream for c in encapsulate(pdfPage.contents))
            resources = pdfPage.inheritable.Resources
            if resources == None: resources = PdfDict(); pdfPage.Resources = resources
            pdfEditor = PdfStreamEditor(stream, resources, PdfFontGlyphMap())
            pdfEditor.tree = pdfEditor.parse_stream()
            isModified, pdfEditor.tree = pdfEditor.remove_text(pdfEditor.tree, '', {}, True, debug)
            if isModified: pdfPage.Contents = IndirectPdfDict(stream = PdfStream.tree_to_stream(pdfEditor.tree))

            djvusedPageCommands = djvusedPage[2:]
            ocrStream = None
            for djvusedCmd in djvusedPageCommands:
                # assert syntax
                if djvusedCmd[0] != 'set-txt': continue
                djvusedPageTree = djvusedCmd[1]
                if djvusedPageTree[0] != 'page': err('argument of set-txt should be a single page chunk')

                # Get the stream
                # ocrStream,fontsize = self.djvusedPageTreeToPDFStream([djvusedPageTree],font,None,None,debug)

                invisible = '' if debug else ' 3 Tr'
                ocrStream = f'BT /OCR 1 Tf{invisible}\n' + self.djvusedPageTreeToPDFStream([djvusedPageTree],font,None,None) + 'ET\n'

                # Get the scale factors
                xmin,ymin,xmax,ymax = [float(a) for a in djvusedPageTree[1:5]]
                width,height = xmax - xmin, ymax - ymin
                bbox = pdfPage.inheritable.CropBox if pdfPage.inheritable.CropBox != None else pdfPage.inheritable.MediaBox
                if bbox == None: err(f'No page bbox on page {pageNo}')
                x1,y1,x2,y2 = [float(b) for b in bbox]

                if pdfPage.inheritable.Rotate in ['90','270']:
                    sx,sy = (x2-x1)/height,(y2-y1)/width
                else:
                    sx,sy = (x2-x1)/width,(y2-y1)/height

                break

            if ocrStream == None: warn('no set-text command after select') ; continue           

            # Set up a font to be used for the OCR layer
            font.install(pdfPage,'OCR', overwrite=True)

            # Write the geometry part of the OCR PDF stream header; account for pdf page rotations
            # TEST THIS MORE !!!!
            ocrHeader = f'q 1 0 0 1 {x1} {y1} cm\n{sx} 0 0 {sy} 0 0 cm\n'
            if pdfPage.inheritable.Rotate == '270': ocrHeader += f'0 -1 1 0 0 {width} cm\n'
            if pdfPage.inheritable.Rotate == '180': ocrHeader += f'-1 0 0 -1 {width} {height} cm\n'
            if pdfPage.inheritable.Rotate == '90': ocrHeader += f'0 1 -1 0 {height} 0 cm\n'
            ocrHeader += f'1 0 0 1 {-xmin} {-ymin} cm\n'
            ocrStream = ocrHeader + ocrStream + 'Q\n'

            ocrContents = IndirectPdfDict(stream = ocrStream)
            if not isinstance(pdfPage.Contents,PdfArray): pdfPage.Contents = [pdfPage.Contents]
            pdfPage.Contents = [ocrContents] if debug else [ocrContents] + pdfPage.Contents

        return pdf

