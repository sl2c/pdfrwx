#!/usr/bin/env python3

from pdfrw import PdfDict, PdfReader, PdfName

# ============================================================================= PdfXObjects

class PdfObjects(dict):
    '''A utility class to effectively create collections of objects such as images, fonts etc.
    It is a dict that maps id(obj) --> obj.
    You can add objects to the collection by calling either of the following:
    
    * ```add(obj)``` -- ads an obj to the collection;
    * ```read(obj, filter)```, add objects from obj's resources, as well as
    objects from the objects' resources from the obj's resources, recursively;
    * ```read_all(pdf, filter)```, which will read all fonts from the PDF

    The type of objects you gather is defined by the filter.
    Several pre-defined filter are provided by the class.

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

    def read(self, obj:PdfDict, filter = 'PdfObjects.objFilter'):
        '''
        Add objects from obj's resources, as well as objects from the objects' resources
        from the obj's resources, recursively.
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

    def read_all(self, pdf:PdfReader, filter = 'PdfObjects.objFilter'):
        '''
        Read all objects from the pdf, recursively; this just calls self.read(page, filter) on every
        page from pdf.pages.
        '''
        for page in pdf.pages:
            self.read(page, filter)
    
