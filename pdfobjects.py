#!/usr/bin/env python3

from pdfrw import PdfDict, PdfReader, PdfName

# ============================================================================= PdfXObjects

class PdfObjects(dict):
    '''A utility class to effectively create collections of objects such as images, fonts etc.
    It is a dict that maps id(obj) --> obj.
    You can add objects to the collection by calling:
    
    * ```read(obj, filter)```, add objects from obj's resources, as well as
    objects from the objects' resources from the obj's resources, recursively;
    * ```read_all(pdf, filter)```, which will read all fonts from the PDF

    The type of objects you gather is defined by the filter.
    Several pre-defined filter are provided by the class.

    An xobject can be also added to the collection directly using self.add(xobject).
    '''
    objFilter           = lambda name,xobj: xobj.Type == PdfName.XObject
    imageFilter         = lambda name,xobj: xobj.Subtype == PdfName.Image
    formFilter          = lambda name,xobj: xobj.Subtype == PdfName.Form

    fontFilter          = lambda name,xobj: xobj.Type == PdfName.Font
    fontType1Filter     = lambda name,xobj: xobj.Subtype == PdfName.Type1
    fontType3Filter     = lambda name,xobj: xobj.Subtype == PdfName.Type3

    def read(self, obj:PdfDict, filter = 'PdfObjects.objFilter'):
        '''
        Add objects from obj's resources, as well as objects from the objects' resources
        from the obj's resources, recursively.
        '''
        Resources = obj.inheritable.Resources
        if Resources == None: return

        # The resource will run over /XObject, /Font, /ExtGState & other dictionaries
        for resource in Resources.values():
            if not isinstance(resource, PdfDict): continue
            for name, xobj in resource.items():
                if isinstance(xobj, PdfDict):
                    if filter(name, xobj):
                        self[id(xobj)] = xobj
                    self.read(xobj,filter)

    def read_all(self, pdf:PdfReader, filter = 'PdfObjects.objFilter'):
        '''
        Read all objects from the pdf, recursively; this just calls self.read(page, filter) on every
        page from pdf.pages.
        '''
        for page in pdf.pages:
            self.read(page, filter)
    
