#!/usr/bin/env python3


from pdfrw import PdfReader, PdfDict

from .common import err, warn, msg

# ========================================================================== class PdfReaderX

class PdfReaderX(PdfReader):

    '''
    A class derived from PdfReader extended with the following page manipulation functions:

    * removePage()
    * insertPage()
    '''

    def __shift_kids_count(leaf:PdfDict, shift:int):
        '''
        An auxiliary private function used by removePage() and insertPage()
        '''
        if leaf == None: return
        leaf.Count = int(leaf.Count) + shift
        PdfReaderX.__shift_kids_count(leaf.Parent, shift)

    def removePage(self, page:PdfDict):
        '''
        Remove the page from the page tree (the page should be part of the page tree).
        To remove page by pageNo call:

        self.removePage(self.pages[pageNo])
        '''
        page.Parent.Kids.remove(page)
        PdfReaderX.__shift_kids_count(page.Parent, -1)
        if page.Parent.Count == 0 and page.Parent.Parent != None:
            page.Parent.Parent.Kids.remove(page.Parent) # delete empty parent from the grandparent
        page.Parent = None
        self.private.pages = self.readpages(self.Root)
        self.Root.PageLabels = None

    def insertPage(self, page:PdfDict, pageNo:int):
        '''
        Insert page into the page tree before pageNo.
        Values n >= len(self.pages) append the page to the tree.
        Values n <= 0 prepend the page to the tree.
        '''
        if pageNo >= len(self.pages):
            page.Parent = self.Root.Pages
            page.Parent.Kids.append(page)
        else:
            p = self.pages[max(pageNo,0)]
            page.Parent = p.Parent
            page.Parent.Kids.insert(page.Parent.Kids.index(p),page)
        PdfReaderX.__shift_kids_count(page.Parent, +1)
        self.private.pages = self.readpages(self.Root)
        self.Root.PageLabels = None

    def copyPage(pageFrom:PdfDict, pageTo:PdfDict):
        '''
        Copies pageFrom to pageTo while keeping pageTo itself in the page tree.
        This is done by copying all attributes from pageFrom to pageTo,
        except the attributes related to pageTo's position in the page tree.
        This preserves all links to pageTo that may exist in the PDF while
        effectively replacing the page.

        The pageFrom object should not be accessed directly after the call to this function.
        '''

        p, q = pageTo, pageFrom

        p.indirect = q.indirect
        p._stream = q.stream

        p.Contents = q.Contents
        p.Resources = q.inheritable.Resources

        p.MediaBox = q.inheritable.MediaBox
        p.CropBox = q.inheritable.CropBox
        p.ArtBox = q.inheritable.ArtBox
        p.BleedBox = q.inheritable.BleedBox
        p.TrimBox = q.inheritable.TrimBox
        
        p.Rotate = q.inheritable.Rotate
