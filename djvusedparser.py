#!/usr/bin/env python3

from sly import Lexer, Parser

# ========================================================================== class DjVuSedSymbol

class DjVuSedSymbol:

    def __init__(self, name:str):
        self.name = name

    def __repr__(self):
        return self.name

# ========================================================================== class DjVuSedLexer

class DjVuSedLexer(Lexer):
    '''A lexer for djvused files'''

    # Set of token names. This is always required
    tokens = {SYMBOL, INTEGER, STRING, LPAREN, RPAREN, PERIOD}

    # ignore (single) chars
    ignore = "\x00\t\n\x0c\r\x20" + ';'

    # ignored patterns; avoid removing SYMBOLs which can also start
    ignore_comment = r'#(?![0-9A-F]{6}\b).*'

    SYMBOL = r"[_#a-zA-Z][_#\-a-zA-Z0-9]*"
    INTEGER = r"\d+"
    # STRING = r'"([a-zA-Z0-9а-яА-ЯёЁ«»№ !#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~ª-힣]|\\[0-9]{1,3}|\\[abtnvfr\\"])*"'
    STRING = r'"([^\\"]|\\[0-9]{1,3}|\\[abtnvfr\\"])*"'
    LPAREN = r'\('
    RPAREN = r'\)'
    PERIOD = r'\.'

    def error(self, t):
        raise ValueError(f'lexing error: illegal character {t.value[0]} at index {self.index}')

# ========================================================================== class DjVuSedParser

class DjVuSedParser(Parser):
    tokens = DjVuSedLexer.tokens
    start = "stream"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ stream & chunks

    # Stream
    @_('[ chunks ]')
    def stream(self, p):
        return p.chunks if p.chunks else []

    # chunks
    @_('chunk { chunk }')
    def chunks(self, p):
        return [p.chunk0] + p.chunk1

    # chunk
    @_('command', 'period')
    def chunk(self, p):
        return p[0]

    # command
    @_('symbol [ args ]')
    def command(self, p):
        return [p.symbol] + (p.args if p.args else [])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ command arguments: entities except symbols

    # args
    @_('arg { arg }')
    def args(self, p):
        return [p.arg0] + p.arg1

    # arg
    @_('integer', 'string', 'list')
    def arg(self, p):
        return p[0]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lists

    # list
    @_('LPAREN [ entities ] RPAREN')
    def list(self, p):
        return p.entities if p.entities else []

    # entities
    @_('entity { entity }')
    def entities(self, p):
        return [p.entity0] + p.entity1

    # entity
    @_('symbol', 'integer', 'string', 'list')
    def entity(self, p):
        return p[0]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ basic types

    #     def symbol(self, p):

    @_('SYMBOL')
    def symbol(self, p):
        return DjVuSedSymbol(p[0])

    # integer
    @_('INTEGER')
    def integer(self, p):
        return int(p[0])

    # string
    @_('STRING')
    def string(self, p):
        '''
        Converts djvused strings to Unicode strings; see DjVuSed.d2u() for more info.
        '''
        s = p[0][1:-1]
        # u2d() takes care of the Unicode djvused files (those produced by djvused -u)
        s = DjVuSed.d2u(DjVuSed.u2d(s, escape = False))
        return s

    # period
    @_('PERIOD')
    def period(self, p):
        return [DjVuSedSymbol('.')]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ error

    # error
    def error(self, p):
        raise ValueError(f'parsing error at token {p}')

# ========================================================================== class PdfStream

class DjVuSed:
    '''The class provides (static) functions:
    
    stream_to_tree()
    tree_to_stream()

    for translation between the djvused stream and djvused tree representations.
    '''

    @staticmethod
    def stream_to_tree(stream:str):
        '''
        '''
        assert isinstance(stream, str)
        lexer, parser = DjVuSedLexer(), DjVuSedParser()
        tokens = lexer.tokenize(stream)
        tree = parser.parse(tokens)
        return tree

    @staticmethod
    def tree_to_stream(tree, unicodeStrings:bool = False):
        '''
        Convert djvused tree to stream. Setting `unicodeStrings = True` will produce
        Unicode strings in the stream, as with `djvused -u`.
        '''
        result = []
        for leaf in tree:
            if not isinstance(leaf, list):
                raise ValueError(f'leaf is not a list: {leaf}')
            result.append(DjVuSed._obj_to_string(leaf,
                                                 parenthesizeLists = False,
                                                 unicodeStrings = unicodeStrings))
        return '\n'.join(result)

    @staticmethod
    def _obj_to_string(obj,
                       indent:int = 0,
                       parenthesizeLists:bool = True,
                       unicodeStrings:bool = False):
        '''
        '''
        prefix = lambda e: ('\n' + ' '*indent) if isinstance(e, list) else ''
        s = ' '*indent
        if isinstance(obj, DjVuSedSymbol):
            return obj.name
        if isinstance(obj, str):
            s = DjVuSed.u2d(obj)
            if unicodeStrings:
                s = DjVuSed.d2u(s, escape=True)
            return '"' + s + '"'
        if isinstance(obj, int):
            return f'{obj}'
        if isinstance(obj, list):
            if not isinstance(obj[0], DjVuSedSymbol):
                raise TypeError(f'first list element not a symbol: {obj}')
            obj2str = lambda e: DjVuSed._obj_to_string(e,
                                                        indent = indent+1,
                                                        parenthesizeLists = True,
                                                        unicodeStrings = unicodeStrings)
            r = ' '.join(prefix(e) + obj2str(e) for e in obj)
            if parenthesizeLists:
                r = '(' + r + ')'
            return r
        raise ValueError(f'unrecognized object type: {obj}')

    @staticmethod
    def d2u(djvuSedString:str, escape:bool = False):
        '''
        Convert a djvused string (backslash & double quotes chars are escaped, non-ASCII chars are
        written as octals: `\\123`) to a Unicode string.

        If you want to keep backslash and double quotes still escaped in the result, set `escape = True`.
        '''
        s = djvuSedString.encode('latin1').decode('unicode-escape').encode('latin1').decode('utf-8', errors='replace')
        if escape:
            s = ''.join(DjVuSed._escapeChar(c) for c in s)
        return s

    @staticmethod
    def u2d(unicodeStr:str, escape:bool = True):
        '''
        Convert a Unicode string to a djvused string
        (backslash & double quotes chars are escaped, non-ASCII chars are written as octals: `\\123`).

        If `unicodeStr` has all backslashes & double quotes already escaped, set `escape = False`
        to avoid escaping them again.
        '''
        toOct = lambda u: ''.join(f'\\{n:03o}' for n in u.encode('utf-8'))
        return ''.join(toOct(u) if not (32 <= ord(u) < 127) else DjVuSed._escapeChar(u) if escape else u
                        for u in unicodeStr)
    
    @staticmethod
    def _escapeChar(c:str):
        '''
        Escapes backslashes and double quotes
        '''
        special = {'"':'\\"', '\\':'\\\\'}
        return special.get(c) or c

