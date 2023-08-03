# pdfrwx

An extension of the wonderful [pdfrw](https://github.com/pmaupin/pdfrw) library that adds handling of content streams (and all the objects referenced therein, e.g. images, fonts etc.) while keeping it as simple as possible.

So, why extend _pdfrw_ when there are full-fledged PDF libraries like [pypdf](https://pypi.org/project/pypdf/)? Think of it this way: most PDF libraries try to provide functions that simplify common PDF processing tasks. These libraries are very good at what they do, but once in a while a task comes up that requires a new function. Now, if you are a developer who is in such a situation you have to start digging into large amounts of source code of these libraries.

The approach of _pdfrw_ is different: let's parse PDF as much as possible, while keeping the result as simple as possible. Not a bad idea, especially given the fact that the PDF object model (which is just a bunch of dictionaries with some of their values being references to other dictionaries) is very well suited for mapping to standard Python dictionaries. On top of that, _pdfrw_ implements an idea behind another wonderful package [attrdict](https://pypi.org/project/attrdict/) to make traversing PDF objects even simpler: accessing page fonts, for example, can now be done like this:
```python
for fontName, fontDict in page.Resources.Font.items():
  (do something)
```
The next step is to try to parse the dictionary streams — these are special entries in the dictionaries which contain the interesting stuff: text, images, vector graphics etc. _pdfrw_ doesn't do any of that, and it doesn't do it on purpose: in accordance with its Unix-like philosophy, it does what it does very well, and doesn't do a bit extra. The data structure it produces is complete: it contains all of PDF, enough for any possible processing task. And it is simple enough so the developer could start coding right away, spending more time enjoying the passages in the [Adobe's magnum opus](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.7old.pdf), and none — learning another complicated library. And, in fact, _pdfrw_ is an ideal tool to help learn PDF syntax by playing with it!

Ok, if you got to this point then chances are you are wondering: so, where do I go from here? Specifically: how do I parse the dictionary streams? This is where _pdfrwx_ can help: it can parse dictionary streams and do other useful things. But first things first:

# Design choices

_pdfrwx_ first and foremost tries to keep with the philosophy of _pdfrw_ outlined above. To this it adds an observation that in many tasks to process PDF most time is spent on developing a software solution, and not on running it. This leads to the design choices:

* **pure Python implementation**: easier to check and debug the source code of the library if needed;
* **output should be simple**: the output should be mostly standard Python / _pdfrw_ classes;
* **few lines of code, many lines of documentation**: _pdfrwx_ tries to make PDF programming fun.

Now we're ready to see what pdfrwx does and how it achieves it:

# Parsing streams

Here's **examples/example-pdfstream.py** which reads example.pdf, removes all text content from every page and writes the result to example-out.pdf:

```python
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
```

As you can see, we the code runs over pages, then over contents of every page, then uncompresses the contents, as it may be compressed, then parses the contents stream into a tree, removes the BT/ET text blocks, and then parses the resulting tree back to stream. The meaning of each line of code above should be clear apart from, possibly, the toArray lambda function: it's there because a page in PDF can have more the one Contents dictionary, in which case page.Contents is a PdfArray, and each of its elements has to be processed separately. And so, the toArray lambda function makes the situation with page contents more uniform by turning page.Contents that are not arrays into PdfArray with one element.

A few other things have to be noted as well. First, note that in order to acomplish the task the code uses just two new classes from _pdfrwx_: PdfFilter and PdfStream, which one/two function calls from each. Second, the parsed tree i just a nested standard Python list of the following trivial format:
```
[
  ['q', []],
  ['cm', ['1','0','0','1','0','0'],
  ...
  ['BT', [], [ /a tree list of text operators/ ]],
  ...
  ['Q', []]
]
```
So, each leaf (element) of the tree list is itself a list of 2 or 3 elements: the first two elements are the command and the list of arguments (an empty list if the command has no arguments), while the third optional argument is present in the case where the leaf is a block of commands, in which case it is the tree list of the commands that make up the block. By design, there's just one type of block: the BT/ET text block, which uses the name 'BT' in the parsed tree. Note, that in the original PDF stream there's a sequence of commands; it is the PdfStream parse that creates these blocks its output for convenience. To further familiarize themselves with the structure of the output of the stream parser, try inserting a command like pprint(treeIn) right after the call to the parser.

Note also that the _toArray_ function, however useful, has _not_ been implemented in the module, and so you have to code it every time you do the parsing. This may sound strange, but it's the result of the same design principles: the module just parses the stream; its up to the developer to code everything else.

The PdfStream parser class is implemented in pure Python using just about 300 lines of code with the help of the popular (and pure Python!) [SLY](https://github.com/dabeaz/sly) parser generator library by [David Beazley](https://github.com/dabeaz/sly) — check out his lectures on YT, he is terrific! For the curious: the parser uses two parser states (i.e. two different parsers with different grammars that switch between themselves as they operate), one for parsing PDF literal strings (i.e. strings that are in parentheses), and the other — for everything else. Yep, in order for the literal strings to support encoding parentheses as part of the string the format of the PDF literal strings has been made so intricate that it required a separate parser just to parse those. So, if for some reason (speed?) you will ever want to implement the stream parser using another parser generator library make sure it supports a stack for parser states.

# More text stuff: PdfFont

Solves the fonts hell. Docs coming soon, stay tuned.

# PdfFilter

Supported filters:

* /FlateDecode (ZIP, full set of PNG predictors supported)
* /ASCIIHexDecode & /ASCII85Decode
* /LZWDecode
* /RunLengthDecode

# PdfImage

Tired of Adobe's own products bugs when it comes to color accuracy in image exporting? Then you've come to the right place! Aiming to be the most accurate PDF image manipulation class of all. Docs coming soon, stay tuned.

Supported codecs (encode/decode):

* /FlateDecode (PNG, full set of PNG predictors supported) -- handled by the PdfFilter class
* /DCTDecode (JPEG, encode/decode)
* /JPXDecode (JPEG2000, decode only)
* /CCITTFaxDecode (G3/G4 for bitonal images, with no prediction/TIFF Predictor 2, encode/decode)

Supported color spaces:

* /DeviceGray, /CalGray
* /DeviceRGB, /CalRGB
* /DeviceCMYK
* /Lab
* /Indexed (palette-based)
* /ICCBased (embedded color profiles)
* /Separation & /DeviceN (partial support — no support for function-based CS is implemented and none is planned for now)

Plus:

* Full support for rendering intents (image- and page-based)
* Full support for default (page-specific) color spaces
* Full support hard and soft image masks, including mask Matte color (pre-alpha-blended masks)
* 


# Current status

The module is perfectly usable at this time, and has been run through a multitude of tests to make sure it does what it promises to do correctly. However it is in no way near alpha: the interfaces are not yet fully finalized. Moreover, error handling is broken (will probably be fixed soon and made similar to how errors are handled in _pdfrw_). So, ~~play at your own risk~~ at no risk, just don't expect it to be of production quality yet.
