#!/usr/bin/env python3

from pdfrw import PdfReader, PdfWriter, PdfArray
from pdfrwx.pdffilter import PdfFilter
from pdfrwx.pdfstreamparser import PdfStream

toArray = lambda obj: obj if isinstance(obj,PdfArray) \
    else PdfArray([obj]) if obj != None else PdfArray()

pdfIn = PdfReader('example.pdf')
pdfOut = PdfWriter('example-out.pdf')
for page in pdfIn.pages:
    contentsArray = toArray(page.Contents)
    for contents in contentsArray:
        stream = PdfFilter.uncompress(contents).stream
        treeIn = PdfStream.stream_to_tree(stream)
        treeOut = []
        for leaf in treeIn:
            cmd, args = leaf[:2]
            if cmd != 'BT': treeOut.append(leaf)
        contents.stream = PdfStream.tree_to_stream(treeOut)
        contents.Filter = None
    pdfOut.addPage(page)
pdfOut.write()
