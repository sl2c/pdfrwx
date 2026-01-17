#!/usr/bin/env python3

import sys
from sly import Lexer, Parser

from pdfrw import IndirectPdfDict, PdfArray, py23_diffs

from pdfrwx.pdffilter import PdfFilter

# !!!!! Try to remove the sci.py dependency later !!!!!
from scipy.interpolate import interpn

import numpy as np

# ============================================================== timeit

from time import time
class MyTime:
    def __init__(self):
        self.tStart = time()
    def get(self):
        t = time() - self.tStart
        self.tStart = time()
        return t

mtime = MyTime()

def timeit(s):
    print(f'[ {int(mtime.get() * 1000):6d} ms]\t{s}')

# ========================================================================== class PdfFunctionType4Syntax

class PdfFunctionType4Syntax:

    '''
    PDF type 4 function syntax
    '''

    white_spaces_raw = r"\x00\t\n\x0c\r\x20"
    white_spaces = "\x00\t\n\x0c\r\x20"
    delimiters_raw = r'\(\)<>\[\]{}/%'

    arithmetic = r'abs|add|atan|ceiling|cos|cvi|cvr|div|exp|floor|idiv|ln|log|mod|mul|neg|round|sin|sqrt|sub|truncate'
    binary = 'and|bitshift|eq|false|ge|gt|le|lt|ne|not|or|true|xor'
    conditional = 'ifelse|if'
    stack = 'copy|dup|exch|index|pop|roll'

    operator = '|'.join([arithmetic,binary,conditional,stack])

# ========================================================================== class PdfFunctionType4Lexer

class PdfFunctionType4StreamLexer(Lexer):
    '''
    A lexer for pdf type 4 function streams
    '''

    # Set of token names. This is always required
    tokens = {FLOAT, INTEGER, LBRACKET, RBRACKET, IFELSE, IF, OPERATOR, ERROR}

    # ignore (single) chars
    ignore = PdfFunctionType4Syntax.white_spaces

    # ignored patterns
    ignore_comment = r'%.*'

    FLOAT = r"[+\-]?(\d+\.\d*|\d*\.\d+)"
    INTEGER = r"[+\-]?\d+"

    LBRACKET,RBRACKET = r"{",r"}"

    IFELSE = r'ifelse'
    IF = r'if'

    OPERATOR = PdfFunctionType4Syntax.operator

    def error(self, t):
        print(f'lexing error: illegal character {t.value[0]} at index {self.index}', file=sys.stderr)
        self.index += 1
        return t

# ========================================================================== class PdfFunctionType4Parser

class PdfFunctionType4StreamParser(Parser):
    '''
    A parser for PDF type 4 function streams
    '''

    tokens = PdfFunctionType4StreamLexer.tokens
    start = "function"

    # Stream
    @_('LBRACKET [ chunks ] RBRACKET')
    def function(self, p): return p.chunks or []
    @_('chunk { chunk }')
    def chunks(self, p): return [p.chunk0] + p.chunk1

    # Chunk
    @_('error_token', 'literal', 'operator', 'if_cond', 'ifelse_cond')
    def chunk(self, p): return p[0]

    # ERROR token (from lexer)
    # this is different from error() func below which is called when illegal parsing state occurs
    @_('ERROR')
    def error_token(self, p): raise ValueError(f'ERROR token: {[p[0]]}')

    # literal
    @_('integer', 'floating')
    def literal(self, p): return p[0]

    # operator
    @_('OPERATOR')
    def operator(self, p): return p[0]

    # if_cond
    @_('function IF')
    def if_cond(self, p): return {'if':p.function}

    # ifelse_cond
    @_('function function IFELSE')
    def ifelse_cond(self, p): return {'ifelse':(p.function0, p.function1)}

    # integer
    @_('INTEGER')
    def integer(self, p): return int(p[0])

    # floating
    @_('FLOAT')
    def floating(self, p): return float(p[0])

    # Error
    def error(self, p):
        raise ValueError("parsing error at token %s" % str(p))


# ========================================================================== class PdfFunctionType4Stream

class PdfFunctionType4Stream:

    def stream_to_list(stream:str):
        '''
        Parses a PDF function type 4 stream into a list of operators. An example stream is:

        {127 lt {0 mul} if }

        which defines a function acting on a single argument in such a way that all values of the argument
        less than 127 result in the function returning 0, and all other values result in the function returning
        the value of the argument unmodified.
        
        The stream_to_list() function returns this stream as a list which consists of literals as int/float,
        commands as string and if/ifelse conditionals that are represented as {'if':proc} & {'ifelse':(proc1,proc2)}
        dictionaries, where procedures are, in turn, represented as lists.
        '''
        lexer, parser = PdfFunctionType4StreamLexer(), PdfFunctionType4StreamParser()
        tokens = lexer.tokenize(stream)
        return parser.parse(tokens)

    # def list_to_stream(tree:list):
    #     '''
    #     Turn the parsed PDF stream tree back to PDF stream.
    #     '''
    #     pair = {'BT':'ET'}
    #     s = ''
    #     for leaf in tree:
    #         # print(f'leaf: {leaf}')
    #         cmd,args = leaf[0],leaf[1]
    #         if cmd == 'BI': # inline image streams; note: BI's dict should be output without the outermost brackets
    #             s += 'BI ' + \
    #                 ' '.join(f'{PdfStream.obj_to_stream(k)} {PdfStream.obj_to_stream(args[0][k])}' for k in args[0]) \
    #                 + '\n' + args[1] 
    #         else:
    #             s += ' '.join(f'{PdfStream.obj_to_stream(arg)}' for arg in args)+' '+cmd+'\n' if len(args) != 0 else cmd+'\n'
    #         kids = leaf[2] if len(leaf) == 3 else None
    #         if kids != None:
    #             s += PdfStream.tree_to_stream(kids)
    #             s += pair[cmd]+'\n'
    #     return s


# ========================================================================== class PdfFunction

class PdfFunction:
    
    '''
    A PDF function class. 
    '''

    def __init__(self, func:IndirectPdfDict):
        '''
        Creates an instance of PdfFunction from a PDF function object (a PdfDict):

        f = PdfFunction(obj)

        The function maps N = len(func.Domain)/2 arguments into M = len(func.Range)/2 results.
        The actual mapping is obtained by calling self.process(stack).
        '''
        FLOAT = lambda a: [float(x) for x in a]

        self.Domain = FLOAT(func.Domain)
        assert len(func.Domain) % 2 == 0
        self.M = len(self.Domain) // 2 # number of function argument components

        if func.Range != None:
            self.Range = FLOAT(func.Range)
            assert len(self.Range) % 2 == 0
            self.N = len(self.Range) // 2 # number of function result components
        else:
            self.Range, self.N = None,None

        fType = int(func.FunctionType)
        if fType == 0:

            # Type 0: Sampled functions

            assert len(func.Size) == self.M
            self.Size = [int(s) for s in func.Size]

            nSamples = np.prod(np.array(self.Size,dtype=int)) # number of sampled points

            self.BitsPerSample = int(func.BitsPerSample)
            self.Order = int(func.Order or 1)

            self.Encode = FLOAT(func.Encode or [x for s in self.Size for x in [0,s-1]])
            self.Decode = FLOAT(func.Decode or func.Range)

            # Read the samples from a bitstream containing specified samples (w/ BitsPerSample)
            bitstream = py23_diffs.convert_store(PdfFilter.uncompress(func).stream)
            # samples = PdfFilter.unpack_bitstream(bitstream, self.BitsPerSample, nSamples * self.N)
            samples = PdfFilter.unpack_pixels(bitstream, nSamples, self.N, self.BitsPerSample)
            samples = samples.reshape(self.Size[::-1] + [self.N]).astype(float)
            samples = np.transpose(samples)
            self.samples = np.moveaxis(samples,0,-1)

            self.execute = self.execute0

        elif fType == 2:

            # Type 2: Exponential interpolation functions
    
            self.M = 1
            assert len(self.Domain) == 2

            self.C0 = FLOAT(func.C0) if func.C0 != None else [0.0]
            self.C1 = FLOAT(func.C1) if func.C1 != None else [1.0]
            self.N = len(self.C0)
            assert len(self.C1) == self.N
            if self.Range: assert len(self.Range) == 2 * self.N

            self.Exponent = float(func.N)

            self.execute = self.execute2

        elif fType == 3:
    
            # Type 3: Stitching functions
    
            self.M = 1
            assert len(self.Domain) == 2

            self.Functions = [PdfFunction(f) for f in func.Functions]
            self.K = len(self.Functions)
            self.N = self.Functions[0].N
            if self.Range: assert len(self.Range) == 2 * self.N
            assert all(f.M == 1 for f in self.Functions)
            assert all(f.N == self.N for f in self.Functions)

            if self.K > 1: assert self.Domain[0] < self.Domain[1]

            self.Bounds = FLOAT(func.Bounds)
            self.Encode = FLOAT(func.Encode)
            assert len(self.Bounds) == self.K - 1
            assert len(self.Encode) == self.K * 2

            if self.K > 1: assert self.Domains[0] < self.Bounds[0] and self.Bounds[-1] < self.Domains[1]
            if self.K > 2: assert all(self.Bounds[i] < self.Bounds[i+1] for i in range(self.K - 2))

            self.execute = self.execute3

        elif fType == 4:

            # Type 4: PostScript calculator functions
    
            self.tree = PdfFunctionType4Stream.stream_to_list(PdfFilter.uncompress(func).stream)
            self.execute = self.execute4

        else:
            raise ValueError(f'bad FunctionType: {func.FunctionType}')

    def process(self, stack:np.ndarray):
        '''
        Maps M = len(self.Domain)/2 components of the arguments into N = len(self.Range)/2 components
        of the result. Both arguments and results are given as a stack - a numpy array in which
        components are stacked along the 0-th axis, each component being itself a numpy array
        of arbitrary shape (the shapes of the arguments' and results' components should be the same, however).
        '''
        assert stack.shape[0] == self.M

        # Clip according to Domain
        s = stack.astype(float)
        s = [np.clip(s[i], self.Domain[2*i], self.Domain[2*i+1]) for i in range(self.M)]

        # Execute the tree
        s = self.execute(np.stack(s,axis=0))
        if s.shape[0] < self.N: raise ValueError(f'expected at least {self.N} results, got {s.shape[0]}')
        if s.shape[0] > self.N: s = s[:self.N]

        # Clip according to Range
        if self.Range != None:
            s = [np.clip(s[i], self.Range[2*i], self.Range[2*i+1]) for i in range(self.N)]

        return np.stack(s,axis=0)

    def execute0(self, stack:np.ndarray):
        '''
        Execute the Type 0 (sampled) function.
        The argument stack represents a self.M-component argument of a function
        and should thus have shape (self.M, image_shape).
        The returned stack has shape (self.N, image_shape).
        '''
        assert stack.shape[0] == self.M

        INTERPOLATE = lambda x,xmin,xmax,ymin,ymax: ymin + ((x-xmin)*(ymax-ymin)/(xmax-xmin))

        # Encoding
        e = [INTERPOLATE(stack[i],self.Domain[2*i], self.Domain[2*i+1], self.Encode[2*i], self.Encode[2*i+1])
                for i in range(self.M)]
        e = [np.clip(e[i],0,self.Size[i]-1) for i in range(self.M)]

        # Interpolation 
        points = [np.arange(self.Size[i]) for i in range(self.M)]
        values = self.samples.reshape(-1, self.N)
        xi = np.stack(e, axis=-1).reshape(-1, self.M)
        method = 'linear' if self.Order == 1 else 'cubic'

        r = interpn(points, values, xi, method)

        r = r.reshape(stack.shape[1:] + (self.N,))

        # Decoding
        r = [INTERPOLATE(r[...,i], 0, 1 << self.BitsPerSample, self.Decode[2*i], self.Decode[2*i+1],)
                for i in range(self.N)]

        return np.stack(r, axis=0)

    def execute2(self, stack:np.ndarray):
        '''
        Execute the Type 2 (exponential) function.
        The argument stack represents a single argument of the function and has shape (1, image_shape).
        The returned stack will have shape (self.N, image_shape).
        '''
        r = [self.C0[i] + (stack[0] ** self.Exponent) * (self.C1[i] - self.C0[i]) for i in range(self.N)]
        return np.stack(r, axis=0)

    def execute3(self, stack:np.ndarray):
        '''
        Execute the Type 3 (stitching) function.
        The argument stack represents a single argument of the function and has shape (1, image_shape).
        The returned stack will have shape (self.N, image_shape).
        Note: by having to do this with numpy we have a self.K-fold computational overhead.
        '''
        INTERPOLATE = lambda x,xmin,xmax,ymin,ymax: ymin + ((x-xmin)*(ymax-ymin)/(xmax-xmin))

        r = np.empty((self.Functions[0].N,) + stack.shape[1:]) ; r[:] = np.nan
        for i in range(self.K):
            xmin = self.Domain[0] if i == 0 else self.Bounds[i-1]
            xmax = self.Domain[1] if i == self.K - 1 else self.Bounds[i]
            mask = stack >= xmin & ((stack < xmax) if i != self.K - 1 else (stack <= xmax))
            e = INTERPOLATE(stack[0], xmin, xmax, self.Encode[2*i], self.Encode(2*i+1))
            f = self.Functions[i].process(e[np.newaxis])
            r[mask] = f[mask]

        return r

    def execute4(self, stack:np.ndarray):
        '''
        Execute the Type 4 (calculator) function.
        The argument stack represents a self.M-component argument of a function
        and should thus have shape (self.M, image_shape).
        The returned stack has shape (self.N, image_shape).
        '''
        # s = stack
        M,H,W = stack.shape
        s = []
        for i in range(M):
            s.append(stack[i])
 
        tree = self.tree

        FALSE = np.zeros_like(s[0],dtype=int)
        TRUE = -np.ones_like(s[0],dtype=int)

        ONES = np.ones_like(s[0], dtype=int)

        BOOL = lambda a: np.logical_not(np.isclose(a,0))
        INT = lambda a: np.rint(a).astype(int)
        # FLOAT = lambda a: a.astype(float)

        def MOD(s,func):
            s[-1] = func(s[-1])

        print(f'Executing function type 4 [stack.shape={[M,H,W]}]:', end=' ')
        sys.stdout.flush()

        for leaf in tree:

            print(leaf, end=' ')
            sys.stdout.flush()

            # Executing conditional operators using numpy is akin to creating parallel Universes.
            # In particular, each pixel's stack length can be different after a conditional since
            # procedures can change the size of the stack. To keep all of this in a single
            # Multiverse array one needs to do some padding (up to the largest stack size)
            if isinstance(leaf,dict):
                s, cond = s[:-1], s[-1]

                # Calculate the Universes
                if 'if' in leaf:
                    raise ValueError(f'PDF type4 function command not implemented: {leaf}')
                    proc = leaf['if']
                    s1 = PdfFunction.execute(proc, s)
                    s2 = s
                elif 'ifelse' in leaf:
                    raise ValueError(f'PDF type4 function command not implemented: {leaf}')
                    proc1, proc2 = leaf['ifelse']
                    s1 = PdfFunction.execute(proc1, s)
                    s2 = PdfFunction.execute(proc2, s)
                else:
                    raise ValueError(f'bad dict in tree: {tree}')

                # Create the Multiverse
                # Do padding (at the bottom of stack); use NaN to make sure the code never exhausts the stack
                N1,N2 = s1.shape[0],s2.shape[0]
                N = max(N1,N2)
                s = np.empty((N,) + s.shape[1:]) ; s[:] = np.nan
                mask = BOOL(cond)
                not_mask = np.logical_not(mask)
                s[N-N1:,mask] = s1[:,mask]
                s[N-N2:,not_mask] = s2[:,not_mask]

            # Ints and floats
            elif isinstance(leaf,float) or isinstance(leaf,int):
                s.append(ONES * leaf)

            # Boolean literals
            elif leaf == 'false': s.append(FALSE)
            elif leaf == 'true': s.append(TRUE)

            # Boolean 1-argument operators
            elif leaf == 'not': s[-1] = np.bitwise_xor(INT(s[-1]), TRUE)

            # Stack manipulation
            elif leaf == 'pop': s.pop()
            elif leaf == 'exch': tmp = s[-1]; s[-1] = s[-2]; s[-2] = tmp
            elif leaf == 'dup': s.append(s[-1])
            elif leaf in ['copy']:
                raise ValueError(f'PDF type4 function command not implemented: {leaf}')

            elif leaf == 'index':

                idx = np.rint(s.pop()).astype(np.int8)

                # Short-cut for uniform idx
                idx0 = idx.flat[0]
                if np.all(idx == idx0):

                    s.append(s[len(s) - 1 - idx0])

                else:

                    out = np.full(s[0].shape, np.nan, dtype=s[0].dtype)

                    abs_idx = len(s) - 1 - idx
                    # Can be optimized further by limiting range of i
                    for i, arr in enumerate(s):
                        np.copyto(out, arr, where=(abs_idx == i))

                    s.append(out)

            elif leaf == 'roll':

                j = np.rint(s.pop()).astype(np.int8)
                n = np.rint(s.pop()).astype(np.int8)

                depth = len(s)
                H, W = n.shape

                # Max number of elements possibly rolled
                n_max = min(depth, int(np.max(n)))

                if n_max > 1:

                    # Short-cut for uniform j and n arrays
                    n0, j0 = n.flat[0], j.flat[0]
                    if np.all(n == n0) and np.all(j == j0):

                        k = j0 % n0
                        s[-n0:] = s[-k:] + s[-n0:-k]

                    else:

                        # Slice of stack involved in roll
                        base = depth - n_max
                        slice_arrays = s[base:]

                        # Preallocate result buffers
                        result = [arr.copy() for arr in slice_arrays]

                        # For each slot in the rolled region
                        for dst in range(n_max):

                            # Only pixels where dst < n participate
                            active = dst < n

                            # For this destination slot, compute source slot per pixel
                            src = np.zeros_like(j)
                            np.add(dst, j, out=src, where=active)
                            np.remainder(src, n, out=src, where=active) # note that pixels with n == 0 are not active

                            # Fill from appropriate source arrays
                            for x in range(1):
                                for i in range(n_max):
                                    sel = active & (src == i)
                                    np.copyto(result[n_max - dst - 1], slice_arrays[n_max - i - 1], where=sel)

                        # Write back
                        s[base:] = result

                # s, j = s[:-1], np.rint(s[-1]).astype('int8') # roll shift
                # s, n = s[:-1], np.rint(s[-1]).astype('int8') # roll region size
                # n_max = np.max(n)
                # s_slice = s[-n_max:,...] # work with the slice to speed things up a bit
                # idx0 = np.arange(s_slice.shape[0])[:, np.newaxis, np.newaxis]
                # idx1 = np.arange(s_slice.shape[1])[np.newaxis, :, np.newaxis]
                # idx2 = np.arange(s_slice.shape[2])[np.newaxis, np.newaxis, :]
                # start = n_max - n

                # # # Ver. 1
                # # idx0_rolled = np.where(idx0 < start, idx0, start + (idx0 - start - j) % n)
                # # s_slice[...] = s_slice[idx0_rolled, idx1, idx2]

                # # Ver. 2 (slightly faster)
                # idx0_rolled = np.where(idx0 < start, idx0, start + (idx0 - start + j) % n)
                # # need a copy: https://github.com/numpy/numpy/issues/26542
                # s_slice[idx0_rolled, idx1, idx2] = np.copy(s_slice)

            # Math: 1-argument operators
            elif leaf == 'abs': MOD(s,np.absolute)
            elif leaf == 'neg': MOD(s,np.negative)
            elif leaf == 'ceiling': MOD(s,np.ceil)
            elif leaf == 'floor': MOD(s,np.floor)
            elif leaf == 'round': MOD(s,np.round)
            elif leaf in ['truncate', 'cvi']: MOD(s,np.trunc)
            elif leaf == 'cvr': pass

            elif leaf == 'sqrt': MOD(s,np.sqrt)
            elif leaf == 'sin': MOD(s,np.sin)
            elif leaf == 'cos': MOD(s,np.cos)
            elif leaf == 'ln': MOD(s,np.log)
            elif leaf == 'log': MOD(s,np.log10)

            # 2-argument operators
            elif leaf in ['add','sub','mul','div','exp','atan',
                        'idiv','mod',
                        'and','or','xor',
                        'eq','ne','lt','gt','le','ge']:
                y = s.pop()
                x = s.pop()

                # Math: 2-argument operators

                if leaf == 'add': r = x + y
                elif leaf == 'sub': r = x - y
                elif leaf == 'mul': r = x * y
                elif leaf == 'div': r = x / y
                elif leaf == 'exp': r = x ** y
                elif leaf == 'atan': r = np.atan2(x,y)

                elif leaf == 'idiv': r = np.trunc(INT(x)/INT(y))
                elif leaf == 'mod': r = np.sign(x) * (np.abs(INT(x)) % np.abs(INT(y)))

                # Boolean: 2-argument operators

                elif leaf == 'and': r = np.bitwise_and(INT(x),INT(y))
                elif leaf == 'or': r = np.bitwise_or(INT(x),INT(y))
                elif leaf == 'xor': r = np.bitwise_xor(INT(x),INT(y))

                elif leaf in ['eq','ne','lt','gt','le','ge']:
                    # Using CLOSE/NOT_CLOSE in place of equal/not equal b/c we represent ints as floats
                    CLOSE = np.isclose(x,y)
                    NOT_CLOSE = np.logical_not(CLOSE)
                    if leaf == 'eq': mask = CLOSE
                    elif leaf == 'ne': mask = NOT_CLOSE
                    elif leaf == 'lt': mask = np.logical_and(x < y, NOT_CLOSE)
                    elif leaf == 'gt': mask = np.logical_and(x > y, NOT_CLOSE)
                    elif leaf == 'le': mask = np.logical_or(x < y, CLOSE)
                    elif leaf == 'ge': mask = np.logical_or(x > y, CLOSE)
                    r = np.where(mask, TRUE, FALSE)
                else:
                    raise ValueError('internal error')

                # Final pop
                s.append(r)

            else:
                raise ValueError(f'bad leaf: {leaf} in tree: {tree}')

            # timeit(f'{leaf}\t--> {print_stack(s)}')
            # timeit(f'{leaf} --> stack size: {s.shape[0]}')

        # Turn list back to numpy array
        s = np.stack(s, axis = 0)

        print()
        print(f'Function type 4 successfully executed [stack.shape={s.shape}]')
        return s


def main():

    function = IndirectPdfDict(
        stream = '{dup 0 mul exch dup 0 mul exch dup 0 mul exch 1 mul }',
        FunctionType = 4,
        Domain = PdfArray([0, 1]),
        Range = PdfArray([0, 1, 0, 1, 0, 1, 0, 1])
    )

    f = PdfFunction(function)
    print(f.tree)
    stack = np.arange(256).reshape(1,16,16)
    out = f.process(stack)
    print(stack.shape, '->', out.shape)

if __name__ == '__main__':
    main()