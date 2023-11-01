#!/usr/bin/env python3

# Write this man about the bug in PIL Image.paste()!
# https://sudonull.com/post/129265-Alpha_composite-optimization-history-in-Pillow-20
# https://habr.com/ru/post/98743/
# https://github.com/homm/
# https://twitter.com/wouldntfix

from multiprocessing.sharedctypes import Value
import struct, zlib, base64, sys, re
from io import BytesIO
import numpy as np

from PIL import Image, TiffImagePlugin, ImageChops, ImageCms
Image.MAX_IMAGE_PIXELS = None

from pdfrw import PdfObject, PdfName, PdfArray, PdfDict, IndirectPdfDict, py23_diffs
from pdfrwx.common import err, msg, warn, eprint, get_key, encapsulate

# ========================================================================== PdfFilter

class PdfFilter:

    # -------------------------------------------------------------------- uncompress()

    def uncompress(obj:IndirectPdfDict):
        '''
        Returns an uncompressed version of obj. Supports objects compressed with one or more of the following filters:

        * /ASCIIHexDecode and /ASCII85Decode
        * /FlateDecode and /LZWDecode (all values of /Predictor are supported)
        * /RunLengthDecode
        '''

        if obj.stream == None or len(obj.stream) == 0: return obj
        stream = py23_diffs.convert_store(obj.stream) if isinstance(obj.stream,str) else obj.stream

        filters = encapsulate(obj.Filter)
        parms = encapsulate(obj.DecodeParms)

        decapsulate = lambda array: array[0] if len(array) == 1 else array

        if obj.Subtype == '/Image':
            width, height = int(obj.Width), int(obj.Height)

        f = 0
        while f < len(filters):

            filter = filters[f]
            if filter == 'null': filter = None

            parm = parms[f] if f < len(parms) else PdfDict()
            if parm == 'null': parm = PdfDict()

            if filter == None: # No filter

                pass

            elif filter in ['/ASCIIHexDecode', '/AHx']:

                # PDF Ref. 1.7 Sec. 3.3.1: '>' is the EOD marker
                stream = stream.decode('latin1').rstrip()
                assert stream[-1] == '>'
                stream = re.sub(r'\s','', stream[:-1])
                if len(stream) % 2 != 0: stream += '0'
                stream = bytes.fromhex(stream)

            elif filter in ['/ASCII85Decode', '/A85']:

                # PDF Ref. 1.7 Sec. 3.3.2: '~>' is the EOD marker
                stream = stream.rstrip()
                assert stream[-2:] == b'~>'
                stream = base64.a85decode(stream[:-2])

            elif filter in ['/FlateDecode', '/Fl', '/LZWDecode', '/LZW']:

                # PDF Ref. 1.7 Sec. 3.3.3
                if filter in ['/FlateDecode','/Fl']:
                    stream = zlib.decompress(stream)
                else:
                    earlyChange = int(get_key(parm, '/EarlyChange', '1'))
                    stream = PdfFilter.lzw_decode(stream, earlyChange)

                predictor = int(get_key(parm, '/Predictor', '1'))
                colors = int(get_key(parm, '/Colors', '1'))
                bpc = int(get_key(parm, '/BitsPerComponent', '8'))
                columns = int(get_key(parm, '/Columns', '1'))

                if predictor == 1: # No predictor

                    pass

                elif 10 <= predictor <= 15: # PNG filters

                    width_bytes = len(stream) // height
                    array = np.frombuffer(stream,dtype='uint8').reshape(height, width_bytes)

                    # Fix incorrect /Predictor value, if necessary
                    if f < len(parms):
                        predValues = set(np.transpose(array)[0])
                        if len(predValues) == 1:
                            predValue = next(iter(predValues)) + 10
                            if predValue != predictor:
                                parms[f].Predictor = predValue
                                obj.DecodeParms = decapsulate(parms)
                                warn(f'fixed PNG predictor: {predictor} --> {parms[f].Predictor}')
                        else:
                            if predictor != 15:
                                parms[f].Predictor = 15
                                obj.DecodeParms = decapsulate(parms)
                                warn(f'fixed PNG predictor: {predictor} --> 15')

                    if colors in [1,3]:

                        mode = 'L' if colors == 1 else 'RGB'
                        stream = PdfFilter.make_png_image(width,height,bpc,mode,zlib.compress(stream),None).tobytes()

                    else:
                        # If # of colors is not 1 or 3 we transform each color component into a
                        # grayscale PNG image and decode it using PIL; the components are then reassembled

                        predVector = array[:,0].reshape(height,1)
                        array = array[:,1:].reshape(height,width,colors)
                        array = np.transpose(array,(2,0,1))

                        result = np.zeros(height*width*colors, dtype='uint8').reshape(colors,height,width)
                        for c in range(colors):
                            encoded = np.hstack((predVector,array[c])).tobytes()
                            decoded = PdfFilter.make_png_image(width,height,bpc,'L',zlib.compress(encoded),None).tobytes()
                            component = np.frombuffer(decoded,dtype='uint8').reshape(height, width)
                            result[c] = component

                        stream = np.transpose(result,(1,2,0)).tobytes()
                 
                elif predictor == 2: # TIFF Predictor 2 filter

                    array = PdfFilter.unpack_pixels(stream, width, height, colors, bpc)

                    array = np.cumsum(array, axis=1, dtype = 'uint16' if bpc == 16 else 'uint8')

                    stream = PdfFilter.pack_pixels(stream, width, height, colors, bpc)

                else:
                    raise ValueError(f'unknown predictor: {predictor}')

            elif filter in ['/RunLengthDecode', '/RL']:

                # PDF Ref. 1.7 Sec. 3.3.4
                s = b''
                i = 0
                while i < len(stream):
                    runLength = stream[i]
                    if runLength == 128: break
                    elif 0 <= runLength < 128:
                        j = (i + 1) + (runLength + 1)
                        s += stream[i+1, j]
                        i = j
                    else:
                        s += stream[i+1] * (257 - runLength)
                        i += 2
                stream = s

            else:
                break

            # move to the next filter
            f += 1

        # there should be at most 1 (image-specific) filter left
        assert len(filters) - f <= 1

        uncapsulate = lambda array: array[0] if len(array) == 1 else None

        result = obj.copy()
        result.Filter = uncapsulate(filters[f:])
        result.DecodeParms = uncapsulate(parms[f:])
        result.stream = py23_diffs.convert_load(stream)
        result.Length = len(result.stream)

        return result

    # -------------------------------------------------------------------- unpack_pixels()

    def unpack_pixels(stream:bytes, width:int, height:int, colors:int, bpc:int):
        '''
        Converts a bytes stream of pixels, packed with bpc bits per pixel, into a numpy array
        of shape (height, width, colors). The elements of the array take values of the form N * M,
        where N = 2^{8-bpc} and M = 1..2^{bpc}-1 if bpc <= 8, or 1..2^16-1 if bpc == 16.
        '''
        assert 1<= bpc <=8 or bpc == 16
        array = np.frombuffer(stream, dtype='uint16' if bpc == 16 else 'uint8')
        if bpc not in [8,16]:
            assert len(stream) % height == 0
            width_bytes = len(stream)// height
            array = array.reshape(height, width_bytes)
            array = np.unpackbits(array, axis=1)
            array = array[:,:width * colors * bpc].reshape(height, width, colors, bpc)
            array = np.packbits(array, axis=3) # bits in the upper part of byte
        return array.reshape(height, width, colors)

    # -------------------------------------------------------------------- pack_pixels()

    def pack_pixels(array:np.ndarray, width:int, height:int, colors:int, bpc:int):
        '''
        Converts a numpy array of shape (height, width, colors) into a bytes stream of pixels,
        with bpc bits per pixel. The elements of the array should have values of the form N * M,
        where N = 2^{8-bpc} and M = 1..2^{bpc}-1.
        '''
        assert 1<= bpc <=8 or bpc == 16
        if bpc not in [8,16]:
            array = array.reshape(height, width, colors, 1)
            array = np.unpackbits(array,axis=3)[:,:,:,:bpc].reshape(height, width * colors * bpc)
            array = np.packbits(array, axis=1)
        return array.tobytes()

    # -------------------------------------------------------------------- make_png_image()

    def make_png_image(width:int, height:int, bpc:int, mode:str, stream:bytes, palette:bytes):
        '''
        Creates a PNG PIL Image from the zlib (/Deflate) compressed bytes stream and, for mode == 'P', the palette bytes.
        PNG supports the following modes: 'L', 'RGB', 'P', 'LA', 'RGBA' (see 'color types' in the PNG spec.)
        '''
        png_color_types = {'1':0, 'L':0, 'RGB':2, 'P':3, 'LA':4, 'RGBA':6} # LA & RGBA are not part of the PDF standard (yet)
        if mode not in png_color_types: err(f'unsupported mode: {mode}')
        color_type = png_color_types[mode]

        PNG_header = bytes([137,80,78,71,13,10,26,10])

        print([width, height, bpc, color_type])

        IHDR = PdfFilter.make_png_chunk(b'IHDR', struct.pack(b'!LLBBBBB', width, height, bpc, color_type, 0, 0, 0))
        PLTE = PdfFilter.make_png_chunk(b'PLTE', palette) if mode == 'P' else b''
        IDAT = PdfFilter.make_png_chunk(b'IDAT', stream)
        IEND = PdfFilter.make_png_chunk(b'IEND', b'')

        msg(f'Length of PLTE: {len(PLTE)}')
        return Image.open(BytesIO(PNG_header + IHDR + PLTE + IDAT + IEND))

    def make_png_chunk(chunk_type:bytes, chunk_data:bytes):
        chunk_len = struct.pack('!L',len(chunk_data))
        chunk_crc = struct.pack('!L',zlib.crc32(chunk_type + chunk_data))
        return chunk_len + chunk_type + chunk_data + chunk_crc

    # -------------------------------------------------------------------- lzw_decode()

    def lzw_decode(byteString:bytes, earlyChange = 1):
        '''
        Decodes an LZW-encoded byteString

        !!! Add processing of the earlyChange == 0 case !!!
        '''
        if earlyChange == 0:
            warn(f'/LZWDecode with /EarlyChange == 0 not implemented')
            return None

        # The initial state and the error
        INIT = (None, [bytes([i]) for i in range(256)] + [None, None]) 
        CODE_ERROR = lambda: ValueError(f'bad code: {code}, table length: {L:X}')

        bits = ''.join(f'{byte:08b}' for byte in byteString) # The byteString as a string of 0s and 1s
        prev_string, table = INIT
        result, n = b'', 0
    
        while n < len(bits):
    
            # Determine the length of the variable-length encoding
            L = len(table)
            codeBits = 12 if L >= 2047 else 11 if L >= 1023 else 10 if L >= 511 else 9

            # Read the code
            code = int(bits[n:n+codeBits],2)
            n += codeBits

            # Interpret the code
            if code == 256:
                prev_string, table = INIT
            elif code == 257:
                pass
            else:
                if code > L or code == L and prev_string == None:
                    raise CODE_ERROR
                if code == L:
                    table.append(prev_string + prev_string[:1])
                string = table[code]
                if code < L and prev_string != None:
                    table.append(prev_string + string[:1])
                result += string
                prev_string = string

        return result

