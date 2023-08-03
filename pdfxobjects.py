#!/usr/bin/env python3

from pdfrw import PdfDict

# ============================================================================= PdfXObjects

class PdfXObjects(dict):
    """A utility class to store collections of xobjects. It is a dict that maps id(xobject) --> xobject.
    
    The class has only one function read(dictionary,filter) that reads xobjects contained in a dictionary's Resources
    (as well as xobjects contained in each of these xobjects' own Resources, recursively). The optional filter
    argument is a boolean function filter(name,xobject) that takes two arguments: the name by which the xobject is
    referred to in Resources, and the xobject itself.

    An xobject can be also added to the collection directly using self.add(xobject).

    Here's example code that uses the PdfXObjects class to process images in a PDF:
    ```
    imageFilter = lambda name,xobj: xobj.Subtype == PdfName.Image
    images = PdfXObjects()
    for page in pdf.pages: images.read(page, imageFilter)
    for image in images.values(): process(image)
    ```
    """

    def read(self, xobject:PdfDict, filter=None):
        '''
        Recursively appends images contained in PDF dictionary's Resources to the collection
        '''
        res = xobject.inheritable.Resources
        if res == None or res.XObject == None: return
        for name, xobj in res.XObject.items():
            if filter == None or filter(name,xobj):
                self[id(xobj)] = xobj
            self.read(xobj,filter)

    def add(self, xobject:PdfDict):
        '''
        Does this: self[id(xobject)] = xobject
        '''
        self[id(xobject)] = xobject
    
