#!/usr/bin/env python3

from sly import Lexer, Parser

# ========================================================================== class DjVuSedLexer

class DjVuSedLexer(Lexer):
    '''A lexer for djvused files'''

    # Set of token names. This is always required
    tokens = {BOOKMARKS, SELECT, REMOVE_TXT, SET_TXT, SAVE, PERIOD, TYPE, INTEGER, STRING, LPAREN, RPAREN}

    # ignore (single) chars
    ignore = "\x00\t\n\x0c\r\x20" + ';'

    # ignored patterns
    ignore_comment = r'#.*'

    # commands
    BOOKMARKS = r'bookmarks'
    SELECT = r'select'
    REMOVE_TXT = r'remove-txt'
    SET_TXT = r'set-txt'
    SAVE = r'save'

    PERIOD = r'\.'

    TYPE = r'page|column|region|para|line|word|char'
    INTEGER = r"\d+"
    # SYMBOL = r"[a-zA-Z_#][a-zA-Z0-9_#\-]*"
    STRING = r'"([a-zA-Z0-9а-яА-ЯёЁ«»№ !#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~ª-힣]|\\[0-9]{1,3}|\\[abtnvfr\\"])*"'
    LPAREN = r'\('
    RPAREN = r'\)'

    def error(self, t):
        raise ValueError(f'lexing error: illegal character {t.value[0]} at index {self.index}')

# ========================================================================== class DjVuSedParser

class DjVuSedParser(Parser):
    tokens = DjVuSedLexer.tokens
    start = "stream"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ stream & chunks

    # Stream
    @_('[ chunks ] [ SAVE ]')
    def stream(self, p):
        result = p.chunks if p.chunks else []
        if p.SAVE: result.append([p.SAVE])
        return result

    # chunks
    @_('chunk { chunk }')
    def chunks(self, p): return [p.chunk0] + p.chunk1

    # chunk
    @_('bookmarks_chunk', 'ocr_chunk')
    def chunk(self, p): return p[0]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bookmarks chunk

    # bookmarks_cmd
    @_('LPAREN BOOKMARKS [ bm_lines ] RPAREN')
    def bookmarks_chunk(self, p): return [p.BOOKMARKS, p.bm_lines]

    # bm_lines
    @_('bm_line { bm_line }')
    def bm_lines(self, p): return [p.bm_line0] + p.bm_line1

    # bm_lines
    @_('LPAREN STRING STRING [ bm_lines ] RPAREN')
    def bm_line(self, p):return [p.STRING0, p.STRING1, p.bm_lines] if p.bm_lines else [p.STRING0, p.STRING1]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ocr chunk

    # ocr_chunk
    @_('select_page_cmd [ commands ]')
    def ocr_chunk(self, p): return p.select_page_cmd + (p.commands if p.commands else [])

    # select_page_cmd
    @_('SELECT [ INTEGER ]')
    def select_page_cmd(self, p): return [p.SELECT, p.INTEGER]

    # commands
    @_('command { command }')
    def commands(self, p): return [p.command0] + p.command1

    # command
    @_('remove_txt_cmd','set_txt_cmd')
    def command(self, p): return p[0]

    # remove_txt_cmd
    @_('REMOVE_TXT')
    def remove_txt_cmd(self, p): return [p.REMOVE_TXT]

    # set_txt_cmd
    @_('SET_TXT ocr_text_block PERIOD')
    def set_txt_cmd(self, p): return [p.SET_TXT, p.ocr_text_block]

    # --------------- ocr text blocks

    # ocr_text_block
    @_('LPAREN TYPE INTEGER INTEGER INTEGER INTEGER [ operand ] RPAREN')
    def ocr_text_block(self, p): return [p.TYPE,p.INTEGER0,p.INTEGER1,p.INTEGER2,p.INTEGER3,p.operand]

    # operand
    @_('STRING', 'ocr_text_blocks')
    def operand(self, p): return p[0]

    # text_blocks
    @_('ocr_text_block { ocr_text_block }')
    def ocr_text_blocks(self, p): return [p.ocr_text_block0] + p.ocr_text_block1

    # --------------- errors

    # ERROR token (from lexer)
    # this is different from error() func below which is called when illegal parsing state occurs
    # @_('ERROR')
    # def error_token(self, p): err(f'ERROR token: {p[0]}')

    # Error
    def error(self, p):
        raise ValueError(f'parsing error at token {p}')


