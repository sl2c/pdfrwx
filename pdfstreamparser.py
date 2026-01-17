#!/usr/bin/env python3

import re, sys
from sly import Lexer, Parser
from pdfrw import PdfDict, PdfTokens

# ============================================================== timeit

from time import time
tStart = time()
def timeit(s): print(f'TIME: {s}: {time()-tStart:0.3f} sec')

# ========================================================================== class PdfStreamSyntax

class PdfStreamSyntax:

    '''see PDF Ref. sec. 4.1; order is important: BT should precede B, for example'''

    white_spaces_raw = r"\x00\t\n\x0c\r\x20"
    white_spaces = "\x00\t\n\x0c\r\x20"
    delimiters_raw = r'\(\)<>\[\]{}/%'

    marked_content = r'MP|DP|BMC|BDC|EMC'
    compatibility_section = r'BX|EX'
    xobjects = r'Do'

    type3_fonts = r'd0|d1'
    shading_patterns = r'sh'

    general_graphics_state = r'w|J|j|M|d|ri|i|gs'
    special_graphics_state = r'q|Q|cm'
    color = r'CS|cs|SCN|SC|scn|sc|G|g|RG|rg|K|k'

    path_construction = r'm|l|c|v|y|h|re'
    path_painting = r'S|s|F|f\*|f|B\*|B|b\*|b|n'
    clipping_paths = r'W\*|W'

    text_state = r'Tc|Tw|Tz|TL|Tf|Tr|Ts'
    text_positioning = r'Td|TD|Tm|T\*'
    text_showing = r'Tj|TJ|\'|\"'

    # indirect_reference = r'R'

    operator = '|'.join([marked_content,compatibility_section,xobjects,
                            type3_fonts,shading_patterns,
                            general_graphics_state,special_graphics_state,color,
                            path_construction,path_painting,clipping_paths,
                            text_state,text_positioning,text_showing])

# ========================================================================== class PdfStreamLexer

class PdfStreamLexer(Lexer):
    '''A lexer for pdf content streams'''

    # Set of token names. This is always required
    tokens = {INLINE_IMAGE_STREAM,
                HSTRING, FLOAT, INTEGER,
                BOOLEAN, NULL, NAME_OBJ,
                LBRACKET, RBRACKET, LANGLE, RANGLE, LPAREN,
                BT, ET, BI, OPERATOR, ERROR}

    # ignore (single) chars
    ignore = PdfStreamSyntax.white_spaces

    # ignored patterns
    ignore_comment = r'%.*'

    # Text object
    BT,ET = r'BT',r'ET'

    # Inline image start
    BI = r'BI'
    # Inline image stream: PDF Ref 1.7 sec. 4.8; '(.|\n)' -- match across multiple lines, '*?' -- non-greedy matching
    INLINE_IMAGE_STREAM=r'\bID\s(.|\n)*?EI\s'

    # LSTRING = r"(?<!\\)\(.*?(?<!\\)\)"
    HSTRING = r"<[0-9a-fA-F\n]*>"
    FLOAT = r"[+\-]?(\d+\.\d*|\d*\.\d+)"
    INTEGER = r"[+\-]?\d+"

    BOOLEAN = r'true|false'
    NULL = r'null'
    NAME_OBJ = r"/[^" + PdfStreamSyntax.delimiters_raw + PdfStreamSyntax.white_spaces_raw + r"]+"
    LBRACKET,RBRACKET = r"\[",r"\]"
    LANGLE,RANGLE = r"<<",r">>"

    # If '(' is found switch to a special lexer state
    @_(r"\(")
    def LPAREN(self,t):
        self.push_state(PdfLiteralStringLexer)
        return t

    def error(self, t):
        print(f'lexing error: illegal character {t.value[0]} at index {self.index}', file=sys.stderr)
        self.index += 1
        return t

    OPERATOR = PdfStreamSyntax.operator

# ========================================================================== class PdfLiteralStringLexer

class PdfLiteralStringLexer(Lexer):
    '''A special lexer for pdf literal strings'''

    # Set of token names. This is always required
    tokens = {LS_ESCAPED, LS_LPAREN, LS_RPAREN, LS_CHAR}

    LS_ESCAPED = r"\\([nrtbf\(\)\\]|[0-7]{3})"
    LS_LPAREN, LS_RPAREN = r"\(", r"\)" # This will only match unescaped parens; escaped are matched by LS_ESCAPED
    LS_CHAR = r"(.|\n)" # Everything else, including linefeeds
    # LS_CHAR = r"([^\\\(\)]|\n)+" # Everything else, including linefeeds

    def LS_LPAREN(self,t):
        self.push_state(PdfLiteralStringLexer)
        return t

    def LS_RPAREN(self,t):
        self.pop_state()
        return t

# ========================================================================== class PdfStreamParser

class PdfStreamParser(Parser):
    tokens = PdfStreamLexer.tokens.union(PdfLiteralStringLexer.tokens)
    start = "stream"

    # Stream
    @_('[ chunks ]')
    def stream(self, p): return p.chunks if p.chunks else []
    @_('chunk { chunk }')
    def chunks(self, p): return [p.chunk0] + p.chunk1

    # Chunk
    @_('error_token','command','text_object','inline_image')
    def chunk(self, p): return p[0]

    # ERROR token (from lexer)
    # this is different from error() func below which is called when illegal parsing state occurs
    @_('ERROR')
    def error_token(self, p): raise ValueError(f'ERROR token: {[p[0]]}')

    # Command
    @_('[ operands ] OPERATOR')
    def command(self, p): return [p.OPERATOR, p.operands if p.operands else []]

    # Text object
    @_('BT stream ET')
    def text_object(self, p): return ['BT',[], p.stream]

    # Inline image
    @_('BI [ pairs ] INLINE_IMAGE_STREAM')
    def inline_image(self, p): return ['BI',[dict(p.pairs) if p.pairs else {}, p.INLINE_IMAGE_STREAM]]

    # Operands
    @_('operand { operand }')
    def operands(self, p): return [p.operand0] + p.operand1
    @_('scalar', 'array', 'dict', 'lstring')
    def operand(self, p): return p[0]

    # Scalars
    @_('BOOLEAN','NULL','NAME_OBJ','FLOAT','INTEGER')
    def scalar(self, p): return p[0]
    @_('HSTRING')
    def scalar(self, p): return re.sub('\n','',p[0])

    # Arrays & dicts
    @_('LBRACKET [ operands ] RBRACKET')
    def array(self, p): return p.operands if p.operands else []
    @_('LANGLE [ pairs ] RANGLE')
    def dict(self, p): return dict(p.pairs) if p.pairs else {}

    # Literal strings
    @_('left_paren [ ls_internals ] LS_RPAREN')
    def lstring(self, p): return '(' + (''.join(p.ls_internals) if p.ls_internals else '') + ')'
    @_('LPAREN','LS_LPAREN')
    def left_paren(self, p): return p[0]
    @_('ls_internal { ls_internal }')
    def ls_internals(self, p): return [p.ls_internal0] + p.ls_internal1
    @_('LS_ESCAPED', 'LS_CHAR', 'lstring')
    def ls_internal(self, p): return p[0]

    # Pairs
    @_('pair { pair }')
    def pairs(self, p): return [p.pair0] + p.pair1
    @_('operand operand')
    def pair (self, p): return [p.operand0, p.operand1]

    # Error
    def error(self, p):
        raise ValueError("parsing error at token %s" % str(p))


# ========================================================================== class PdfStream

class PdfStream:
    '''The class provides (static) functions:
    
    stream_to_tree()
    tree_to_stream()
    pass_only_text()
    pass_only_vector_graphics_and_images()

    for translation between the pdf stream and pdf stream tree representations.
    '''

    # ------------------------------------------------------------------------------ stream_to_tree()

    def stream_to_tree(stream:str,
                       filterText:bool=False,
                       filterImages:bool=False,
                       filterVector:bool=False,
                       compactify:bool=False,
                       XObject=None):
        '''
        Parse stream and return the resulting parsed PDF stream tree.
        If textOnly==True only the text and state operators are parsed, i.e. the bare minimum that allows to
        to extract full text–related information from the stream.
        '''
        assert isinstance(stream, str)

        lexer, parser = PdfStreamLexer(), PdfStreamParser()

        timeit('DEBUG: tokenize started')

        # tokens = lexer.tokenize(stream)
        # n = len(list(tokens))
        for t in PdfTokens(stream):
            print(t)
        timeit('DEBUG: tokenize ended')
        sys.exit()

        if any((filterText, filterImages, filterVector)):
            tokens = PdfStream.filter_tokens(tokens = tokens,
                                                filterText = filterText,
                                                filterImages = filterImages,
                                                filterVector = filterVector,
                                                XObject = XObject)
            if compactify:
                tokens = PdfStream.compact_tokens(tokens = tokens)


        tree = parser.parse(tokens)

        if compactify and any((filterText, filterImages, filterVector)):
            tree = PdfStream.compact_tree(tree)

        return tree

    # ------------------------------------------------------------------------------ filter_tokens()

    def filter_tokens(tokens,
                        filterText:bool = False,
                        filterImages:bool = False,
                        filterVector:bool = False,
                        XObject:PdfDict = None):
        '''
        Pass all tokens except the specified classes of operators/objects
        '''

        text_state = ['Tc','Tw','Tz','TL','Tf','Tr','Ts']
        text_positioning = ['Td','TD','Tm','T*']
        text_showing = ['Tj','TJ',"'",'"']
        text_operators = set(text_state + text_positioning + text_showing)

        path_construction = ['m','l','c','v','y','h','re']
        path_painting = ['S','s','f','F','f*','B','B*','b','b*','n']
        clipping_paths = ['W','W*']
        shading_operator = ['sh']
        vector_operators = set(path_construction + path_painting + clipping_paths + shading_operator)

        isInlineImage = False
        args = []
        for tok in tokens:
            if tok.type in ['BT', 'ET']: # Start/end of text blocks
                if not filterText:
                    yield tok
            elif tok.type == 'BI': # Start of an inline image
                isInlineImage = True
                if not filterImages:
                    yield tok
            elif tok.type == 'INLINE_IMAGE_STREAM': # End of an inline image
                isInlineImage = False
                if not filterImages:
                    yield tok
            elif isInlineImage: # The dictionary of the inline image
                if not filterImages:
                    yield tok
            elif tok.type == 'OPERATOR':
                if tok.value == 'Do':
                    assert len(args) == 1 # For debug purposes
                    name = args[-1].value
                    isImage =  XObject is not None and name in XObject and XObject[name].Subtype == '/Image'
                    if isImage and filterImages:
                        yield from args[:-1]
                    else:
                        yield from args; yield tok
                elif tok.value in text_operators:
                    if not filterText:
                        yield from args; yield tok
                elif tok.value in vector_operators:
                    if not filterVector:
                        yield from args; yield tok
                else:
                    yield from args; yield tok
                args = []
            else:
                args.append(tok)

        # Flush
        yield from args

    # ------------------------------------------------------------------------------ compact_tokens()

    def compact_tokens(tokens):
        '''
        Reads `tokens` up to and including the next non-nested `Q` token, or up to the end.
        Yields all read tokens up to and including the last painting operator,
        followed by a non-nested 'Q' token if it was encountered. Whenever
        nested `q/Q` blocks are encountered, this function calls itself recursively on those.
        '''

        compatibility_section = r'BX|EX'
        xobjects = r'Do'
        shading_patterns = r'sh'
        path_painting = r'S|s|F|f\*|f|B\*|B|b\*|b|n'
        clipping_paths = r'W\*|W'
        text_showing = r'Tj|TJ|\'|\"'

        painting_operators = r'|'.join([compatibility_section, xobjects, shading_patterns,
                                        path_painting, clipping_paths, text_showing])
        painting_operators = set(painting_operators.split('|'))
        
        def isPainting(tok):
            return tok.type in ['INLINE_IMAGE_STREAM', 'BT', 'ET', 'BI'] \
                or tok.type == 'OPERATOR' and tok.value in painting_operators

        buffer = []

        for tok in tokens:

            # All tokens are added to the buffer first
            buffer.append(tok)

            # Painting tokens flush the buffer
            if isPainting(tok):
                yield from buffer
                buffer = []

            if tok.type == 'OPERATOR' and tok.value == 'q':

                block = list(PdfStream.compact_tokens(tokens)) # recursion

                if not (len(block) > 0 and block[-1].type == 'OPERATOR' and block[-1].value == 'Q'):
                    raise ValueError('the q-block doesn\'t end with Q')

                if len(block) > 1:
                    # Flush undecided
                    yield from buffer
                    buffer = []
                    # Flush block
                    yield from block
                else:
                    buffer.pop() # remove q

            if tok.type == 'OPERATOR' and tok.value == 'Q':
                yield tok
                break

    # ------------------------------------------------------------------------------ compact_tree()

    def compact_tree(tree:list):
        '''
        Compacts a tree by substituting same-kind consecutive state variables changing
        operators with the last instance
        '''

        general_graphics_state_without_gs = r'w|J|j|M|d|ri|i'
        color = r'CS|cs|SCN|SC|scn|sc|G|g|RG|rg|K|k'
        text_state = r'Tc|Tw|Tz|TL|Tf|Tr|Ts'

        state_changing_operators = r'|'.join((general_graphics_state_without_gs, color, text_state))
        state_changing_operators = set(state_changing_operators.split('|'))

        r = []
        prev_cmd = None
        for leaf in tree:
            cmd, _ = leaf[:2]
            if cmd in state_changing_operators and cmd == prev_cmd:
                r[-1] = leaf # replace
            else:
                r.append(leaf)
            prev_cmd = cmd
        return r

    # ------------------------------------------------------------------------------ tree_to_stream()

    def tree_to_stream(tree:list):
        '''
        Turn the parsed PDF stream tree back to PDF stream.
        '''
        pair = {'BT':'ET'}
        s = ''
        for leaf in tree:
            # print(f'leaf: {leaf}')
            cmd,args = leaf[0],leaf[1]
            if cmd == 'BI': # inline image streams; note: BI's dict should be output without the outermost brackets
                s += 'BI ' + \
                    ' '.join(f'{PdfStream.obj_to_stream(k)} {PdfStream.obj_to_stream(args[0][k])}' for k in args[0]) \
                    + '\n' + args[1] 
            else:
                s += ' '.join(f'{PdfStream.obj_to_stream(arg)}' for arg in args)+' '+cmd+'\n' if len(args) != 0 else cmd+'\n'
            kids = leaf[2] if len(leaf) == 3 else None
            if kids != None:
                s += PdfStream.tree_to_stream(kids)
                s += pair[cmd]+'\n'
        return s

    # ------------------------------------------------------------------------------ obj_to_stream()

    def obj_to_stream(obj):
        '''
        Converts objects to string representation in accordance with the PDF syntax.
        The floats are printed in a fixed point format with 10 decimal points accuracy with no trailing zeros.
        '''
        if isinstance(obj,dict):
            return '<<' + ' '.join([PdfStream.obj_to_stream(key) + ' ' + PdfStream.obj_to_stream(obj[key]) for key in obj]) + '>>'
        elif isinstance(obj,list):
            return '[' + ' '.join([PdfStream.obj_to_stream(element) for element in obj]) + ']'
        elif isinstance(obj,float):
            return f'{obj:.10f}'.rstrip('0').rstrip('.')
        else:
            return f'{obj}'

