#!/usr/bin/env python3

import re
from sly import Lexer, Parser

from pdfrwx.common import err,warn,eprint


# ========================================================================== class PdfStreamSyntax

class PdfStreamSyntax:

    '''see PDF Ref. sec. 4.1; order is important: BT should preceed B, for example'''

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

    indirect_reference = r'R'

    operator = '|'.join([marked_content,compatibility_section,xobjects,
                            type3_fonts,shading_patterns,
                            general_graphics_state,special_graphics_state,color,
                            path_construction,path_painting,clipping_paths,
                            text_state,text_positioning,text_showing,indirect_reference])

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
        eprint(f'lexing error: illegal character {t.value[0]} at index {self.index}')
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
    def error_token(self, p): err(f'ERROR token: {p[0]}')

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
        eprint('\a\a\a\a\a')
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

    def stream_to_tree(stream:str, textOnly:bool=False, vectorGraphicsAndImagesOnly=False, XObject=None):
        '''
        Parse stream and return the resulting parsed PDF stream tree.
        If textOnly==True only the text and state operators are parsed, i.e. the bare minimum that allows to
        to extract full text–related information from the stream.
        '''
        lexer, parser = PdfStreamLexer(), PdfStreamParser()
        tokens = lexer.tokenize(stream)
        if textOnly:
            tokens = PdfStream.pass_only_text(tokens,XObject)
        if vectorGraphicsAndImagesOnly:
            tokens = PdfStream.pass_only_vector_graphics_and_images(tokens,XObject)
        tree = parser.parse(tokens)
        return tree

    def pass_only_text(tokens, XObject=None):
        '''
        Pass the tokens in the entire BT/ET blocks and everything else except images & vector graphics,
        '''
        # N.B. sometimes Tf is present outside the BT/ET block!

        path_construction = ['m', 'l', 'c', 'v', 'y', 'h', 're']
        path_painting = ['S', 's', 'f', 'F', 'f*', 'B', 'B*', 'b', 'b*', 'n']
        clipping_paths = ['W','W*']
        vector_graphics = path_construction + path_painting + clipping_paths

        isText,isInlineImage = False,False
        args = []

        for tok in tokens:
            if tok.type == 'BT': isText = True; yield tok
            elif tok.type == 'ET': isText = False; yield tok
            elif tok.type == 'BI': isInlineImage = True
            elif tok.type == 'INLINE_IMAGE_STREAM': isInlineImage = False
            elif isText: yield tok
            elif isInlineImage: pass
            elif tok.type == 'OPERATOR':
                if tok.value not in vector_graphics and tok.value != 'Do' or tok.value == 'Do' and \
                        not (XObject != None and args[0].value in XObject and XObject[args[0].value].Subtype == '/Image'):
                    for arg in args: yield arg
                    yield tok
                args = []
            else:
                args.append(tok)

    def pass_only_vector_graphics_and_images(tokens,XObject=None):
        '''
        Pass all tokens except the BT/ET blocks
        '''
        isText = False
        args = []

        for tok in tokens:
            if tok.type == 'BT': isText = True
            elif tok.type == 'ET': isText = False
            elif not isText:
                if tok.type == 'OPERATOR':
                    for arg in args: yield arg
                    yield tok
                    args = []
                else:
                    args.append(tok)

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

    def obj_to_stream(obj):
        if isinstance(obj,dict):
            return '<<' + ' '.join([PdfStream.obj_to_stream(key) + ' ' + PdfStream.obj_to_stream(obj[key]) for key in obj]) + '>>'
        elif isinstance(obj,list):
            return '[' + ' '.join([PdfStream.obj_to_stream(element) for element in obj]) + ']'
        else:
            return f'{obj}'


