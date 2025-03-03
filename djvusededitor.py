#!/usr/bin/env python3

import re
from sly import Lexer, Parser

import sys, os, re
from math import sqrt
from typing import Callable

# Try using: github.com/sarnold/pdfrw as it contains many fixes compared to pmaupin's version
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfDict, IndirectPdfDict, PdfName

from .common import err,msg,warn,eprint, encapsulate
from .pdffont import PdfFont, PdfTextString, PdfFontUtils, PdfFontGlyphMap
from .pdfstreamparser import PdfStream
from .pdfstreameditor import PdfStreamEditor
from .djvusedparser import DjVuSedSymbol, DjVuSed
from .pdfstate import PdfState
from .pdffilter import PdfFilter

import xml.etree.ElementTree as ET # for parsing hOCR files



# ========================================================================== class DjVuSedEditor

class DjVuSedEditor:

    def __init__(self, djvuSedStream:str):
        '''Parses the djvuSedStream and stores the resulting djvused page tree in self.tree'''
        self.stream = djvuSedStream
        self.tree = DjVuSed.stream_to_tree(self.stream)

    def djvusedPageTreeToPDFStream(self, djvusedPageTree:list, font:PdfFont, baseline_y = None, scale_y = None):
        '''
        Convert annotations in the parsed djvusedPageTree to a PDF stream.
        The djvusedPageTree can be an arbitrary set of annotation commands that relates to a single page, that is
        anything between two consecutive "select pageNo" commands in the original djvused stream.
        '''

        RATIO = 4 # assumed ascent/descent ratio
        BASELINE = lambda ymin, ymax: int(round((RATIO*ymax + ymin)/(RATIO + 1)))

        stream = ''

        for chunk in djvusedPageTree:
            
            symbol,xmin,ymin,xmax,ymax = chunk[:5] # note that always ymax < ymin, this is just bad naming

            assert isinstance(symbol, DjVuSedSymbol)

            if symbol.name == 'line':
                baseline_y = BASELINE(ymin, ymax)
                scale_y = int(ymin - ymax)

            text = chunk[5] if len(chunk) == 6 and isinstance(chunk[5],str) else chunk[5:]

            if isinstance(text,str) and len(text)>0:

                if len(text) == 0: continue
                pdfString = font.encodePdfTextString(text)
                stringWidth = font.width(text)
                if stringWidth == 0: continue

                scale_x = int(round(float(xmax)-float(xmin))/stringWidth)

                if baseline_y == None:  baseline_y = BASELINE(ymin, ymax)
                if scale_y == None: scale_y = int(ymin - ymax)

                stream += f'{scale_x} 0 0 {scale_y} {xmin} {baseline_y} Tm {pdfString} Tj\n'

            if isinstance(text, list):

                stream += self.djvusedPageTreeToPDFStream(text, font, baseline_y, scale_y)

        return stream

    def insert_ocr(self,
                   pdf:PdfReader,
                   defaultUnicodeFont:str,
                   defaultFontDir:str,
                   firstPage = 1,
                   removeOCR:bool = False,
                   debug = False):

        # One font, many pages
        utils = PdfFontUtils()
        font = utils.loadFont([defaultUnicodeFont], ['.',defaultFontDir])
        if not font: raise ValueError(f'cannot proceed with inserting an OCR layer: no font loaded')
        xobjCache = {}

        pdfPage = None
        ocrStream = None

        for cmd in self.tree:

            if cmd[0].name == 'select':
    
                try: pageNo = int(cmd[1]) + firstPage - 1
                except: pageNo = firstPage
                if pageNo < 1 or pageNo > len(pdf.pages):
                    err(f'pageNo ({pageNo}) is outside pdf page range (1-{len(pdf.pages)})')
    
                if pdfPage != None and ocrStream == None:
                    warn('select command not followed by set-txt')

                eprint(f'inserting OCR in page {pageNo}')
                pdfPage = pdf.pages[pageNo-1]
                ocrStream = None

            if cmd[0].name == 'set-txt':

                if ocrStream != None:
                    err('multiple set-txt commands on the same page')

                if pdfPage == None:
                    err(f'set-text before select')

                # Fix missing Contents
                if pdfPage.Contents == None:
                    pdfPage.Contents = IndirectPdfDict(stream = '')

                # Remove old OCR
                if removeOCR:
                    resources = pdfPage.inheritable.Resources
                    if resources == None: resources = PdfDict(); pdfPage.Resources = resources
                    glyphMap = PdfFontGlyphMap(loadAdobeGlyphList = True)
                    pdfEditor = PdfStreamEditor(pdfPage, glyphMap = glyphMap, makeSyntheticCmap = True)
                    pdfEditor.processText(xobjCache=xobjCache, options={'removeOCR':True})

                if len(cmd) != 2 or cmd[1][0].name != 'page':
                    err('argument of set-txt should be a single page chunk')

                pageChunk = cmd[1]

                invisible = '' if debug else ' 3 Tr'
                ocrStream = f'BT /OCR 1 Tf{invisible}\n' + self.djvusedPageTreeToPDFStream([pageChunk],font,None,None) + 'ET\n'

                # Get the scale factors
                xmin,ymin,xmax,ymax = [float(a) for a in pageChunk[1:5]]
                width,height = xmax - xmin, ymax - ymin
                bbox = pdfPage.inheritable.CropBox or pdfPage.inheritable.MediaBox
                if bbox == None: err(f'No page bbox on page {pageNo}')
                x1,y1,x2,y2 = [float(b) for b in bbox]

                if pdfPage.inheritable.Rotate in ['90','270']:
                    sx,sy = (x2-x1)/height,(y2-y1)/width
                else:
                    sx,sy = (x2-x1)/width,(y2-y1)/height

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

