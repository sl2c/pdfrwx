#!/usr/bin/env python3

import re
from sly import Lexer, Parser

import sys, os, re
from math import sqrt
from typing import Callable

# Try using: github.com/sarnold/pdfrw as it contains many fixes compared to pmaupin's version
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfDict, IndirectPdfDict, PdfName

from .common import err,msg,warn,eprint, encapsulate
from .pdffont import PdfFont, PdfTextString, PdfFontGlyphMap
from .pdfstreamparser import PdfStream
from .djvusedparser import DjVuSedLexer, DjVuSedParser
from .pdfstate import PdfState
from .pdffilter import PdfFilter

import xml.etree.ElementTree as ET # for parsing hOCR files



# ========================================================================== class hocrEditor

class hocrEditor:

    def __init__(self, pdf:PdfReader, defaultUnicodeFont:str, defaultFontDir:str, debug=False):
        self.pdf = pdf
        self.pdfPage = None
        self.stream = ''
        self.invisible = '' if debug else '3 Tr '
        self.isline = False
        self.debug = debug
        self.font = PdfFontUtils.loadFont(defaultUnicodeFont, ['.',defaultFontDir])

    def insert_hocr(self, hocrPath:str, firstPage=0):
        self.pdfPageNo = firstPage
        import xml.etree.ElementTree as ET
        self.root = ET.parse(hocrPath).getroot()
        self.insert_node_in_pdf(self.root)

    def insert_node_in_pdf(self, node:ET.ElementTree):
        '''Inserts a parsed hocr tree node into self.pdf. Use node = self.tree to insert entire hocr stream'''
        cls,id,title,text = self.get_attr(node,'class'),self.get_attr(node,'id'),self.get_attr(node,'title'),node.text
        if text != None: text = text.strip(' \t\n')
        props = {} if title == None else dict([re.split(r'\s+',item,1) for item in re.split(r'\s*;\s*',title)])
        dump = f'class={cls}, id={id}, title={props}, text="{text}"'

        # Get the bbox if it's required
        if text not in [None,''] or cls in ['ocr_page','ocr_line']:
            bbox = self.get_prop_value_list(props,'bbox')
            if len(bbox) != 4: err(f'failed to get bbox: {dump}')

        # Start ocr_page
        if cls == 'ocr_page':
            # Set up pdf page
            if self.pdfPageNo < 0 or self.pdfPageNo >= len(self.pdf.pages):
                err(f'pageNo={self.pdfPageNo} outside pdf page range: {dump}')
            self.pdfPage = self.pdf.pages[self.pdfPageNo]
            self.font.install(self.pdfPage,'OCR')

            # Determine pdf & ocr page bboxes and calculate scaling factors accounting for page rotation
            self.pagebbox = bbox
            xmin,ymin,xmax,ymax = bbox
            width,height = xmax - xmin,ymax-ymin
            i = self.pdfPage.inheritable
            pdfbbox = i.CropBox if i.CropBox != None else i.MediaBox
            if pdfbbox == None: err(f'No pdf page bbox on page {pageNo}')
            x1,y1,x2,y2 = [float(b) for b in pdfbbox]
            if i.Rotate in ['90','270']: sx,sy = (x2-x1)/self.height,(y2-y1)/self.width
            else: sx,sy = (x2-x1)/width,(y2-y1)/height

            # Tranform the coordinate system to the [0 0 width height] rectangle
            self.stream = f'q 1 0 0 1 {x1} {y1} cm\n{sx} 0 0 {sy} 0 0 cm\n'
            if i.Rotate == '90': self.stream += f'0 -1 1 0 0 {width} cm\n'
            if i.Rotate == '180': self.stream += f'-1 0 0 -1 {width} {height} cm\n'
            if i.Rotate == '270': self.stream += f'0 1 -1 0 {height} 0 cm\n'
 
        # Start ocr_line
        if cls == 'ocr_line':
            self.isline = True
            self.linebbox = bbox
            _linebase = self.get_prop_value_list(props,'baseline')
            self.linebase = _linebase if len(_linebase) > 0 else [-0.2*(bbox[3]-bbox[1])]
            # self.linefontsize = None
        
        # Write text to self.stream
        if text not in [None,'']:
            if self.pdfPage == None: err(f'cannot insert text outside ocr_page: {dump}')
            xmin,ymin,xmax,ymax = bbox
            y = ymin*0.2 + ymax*0.8 if not self.isline \
                else self.linebbox[3] + self.polynomial(self.linebase,xmin-self.linebbox[0])            
            hexString = self.font.makePdfString(text)
            fontsize = int(round(1000*self.font.scaleFactor*(xmax-xmin)/self.font.getStringWidth(text)))
            x,y = xmin - self.pagebbox[0],self.pagebbox[3] - y
            self.stream += f'BT /OCR {fontsize} Tf {x:.1f} {y:.1f} Td {self.invisible}{hexString} Tj ET\n'

        # Process kids
        for kid in node: self.insert_node_in_pdf(kid)

        # End ocr_line
        if cls == 'ocr_line': self.isline = False

        # End ocr_page and prepend self.stream to the pdf page stream
        if cls == 'ocr_page':
            self.stream += 'Q\n'
            if self.debug: self.pdfPage.Contents = IndirectPdfDict(stream = '')
            c = self.pdfPage.Contents
            if isinstance(c,PdfArray): c[0].stream = self.stream + c[0].stream
            else: c.stream = self.stream + c.stream
            self.stream = ''
            self.pdfPage = None
            self.pdfPageNo += 1

    def get_attr(self, node:ET.ElementTree,attrName:str):
        return node.attrib[attrName] if attrName in node.attrib else None

    def polynomial(self, coefficients:list,x:float):
        r = 0
        for c in coefficients: r = r*x + c
        return r

    def get_prop_value_list(self, props:dict, propName:str):
        '''Returns a list of floats if props[propName] exists and is a string of int/floats separated
        by whitespace chars, otherwise returns [].'''
        try: return [float(v) for v in re.split(r'\s+', props[propName])]
        except: return []
