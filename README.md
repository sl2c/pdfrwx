# pdfrwx

An extension of the wonderful [pdfrw](https://github.com/pmaupin/pdfrw) library that adds handling of content streams (and all the objects referenced therein, e.g. images, fonts etc.) while keeping it as simple as possible.

So, why extend _pdfrw_ when there are full-fledged PDF libraries like [pypdf](https://pypi.org/project/pypdf/)? Think of it this way: most PDF libraries try to provide functions that simplify common PDF processing tasks. These libraries are very good at what they do, but once in a while a task comes up that requires a new function. Now, if you are a developer who is in such a situation you have to start digging into large amounts of source code of these libraries.

The approach of _pdfrw_ is different: let's parse PDF as much as possible, while keeping the result as simple as possible. Not a bad idea, especially given the fact that the PDF object model (which is just a bunch of dictionaries with some of their values being references to other dictionaries) is very well suited for mapping to standard Python dictionaries. On top of that, _pdfrw_ implements an idea behind another wonderful package [attrdict](https://pypi.org/project/attrdict/) to make traversing PDF objects even simpler: accessing page fonts, for example, can now be done like this:
```python
for fontName, fontDict in page.Resources.Font.items():
  (do something)
```
The next step is to try to parse the dictionary streams — these are special entries in the dictionaries which contain the interesting stuff: text, images, vector graphics etc. _pdfrw_ doesn't do any of that, and it doesn't do it on puprpse: in accordance with its Unix-like philosophy, it does what it does very well, and doesn't do a bit extra. The data structure it produces is complete: it contains all of PDF, enough for any possible processing task. And it is simple enough so the developer could start coding right away, spending more time enjoying the passages in the [Adobe's magnum opus](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.7old.pdf), and none — learning another complicated library. And, in fact, _pdfrw_ is an ideal tool to help learn PDF syntax by playing with it!

Ok, if you got to this point then chances are you are wondering: so, where do I go from here? Specifically: how do I parse the dictionary streams? Turns out, this is not trivial.
