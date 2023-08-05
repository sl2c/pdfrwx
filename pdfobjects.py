#!/usr/bin/env python3

from pdfrw import PdfDict, PdfReader, PdfName

# ============================================================================= PdfXObjects

class PdfObjects(dict):
    '''A utility class to store collections of objects. It is a dict that maps id(xobject) --> xobject.
    
    The class is a map from id(fontDict) to fontDict that can be used to store font collections in a PDF.
    You can add fonts to the collection by calling either of the following:
    
    * low-level function ```add(fontDict)```;
    * medium-level function ```read(obj, filter)```, which will add objects from xobj's resources, as well as
    in the resources of objects from the object's resources, recursively;
    * high-level function ```read_all(pdf, filter)```, which will read all fonts from the PDF

    The several pre-defined filter are provided by the class.

    An xobject can be also added to the collection directly using self.add(xobject).
    '''

    objFilter           = (PdfName.XObject, lambda name,xobj: True)
    imageFilter         = (PdfName.XObject, lambda name,xobj: xobj.Subtype == PdfName.Image)
    formFilter          = (PdfName.XObject, lambda name,xobj: xobj.Subtype == PdfName.Form)

    fontFilter          = (PdfName.Font,    lambda name,xobj: True)
    fontType1Filter     = (PdfName.Font,    lambda name,xobj: xobj.Subtype == PdfName.Type1)
    fontType3Filter     = (PdfName.Font,    lambda name,xobj: xobj.Subtype == PdfName.Type3)

    def add(self, obj:PdfDict):
        '''
        Does this: self[id(xobject)] = xobject
        '''
        self[id(obj)] = obj

    def read(self, obj:PdfDict, filter = 'PdfObjects'.objFilter):
        '''
        Recursively appends images contained in PDF dictionary's Resources to the collection
        '''
        res = obj.inheritable.Resources
        if res == None: return

        type, func = filter
        if type not in ['/XObject', '/Font']: raise ValueError(f'invalid filter: {filter}')

        res = res[type]
        for name, xobj in res.items():
            if func(name,xobj):
                self[id(xobj)] = xobj
            if type == '/XObject': self.read(xobj,filter)

    def read_all(self, pdf:PdfReader, filter = 'PdfObjects'.objFilter):
        '''
        Read all fronts from the pdf.
        '''
        for page in pdf.pages:
            self.read(page, filter)
    
