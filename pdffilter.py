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
from .common import err, msg, warn, eprint, get_key, encapsulate, decapsulate

# ========================================================================== PdfFilter

class PdfFilter:

    # -------------------------------------------------------------------- uncompress()

    @staticmethod
    def uncompress(obj:IndirectPdfDict):
        '''
        Returns an uncompressed version of obj. Supports objects compressed with one or more of the following filters:

        * /ASCIIHexDecode and /ASCII85Decode
        * /FlateDecode and /LZWDecode (all values of /Predictor are supported)
        * /RunLengthDecode

        Also, checks to see if PNG predictor values (obj.DecodeParms[i].Predictor == 10..15)
        coincide with the predictor bytes in the data stream, and if not fixes obj.DecodeParms[i].Predictor
        values (i.e. modifies the obj!).
        '''
        assert isinstance(obj, PdfDict)
        stream = obj.stream
        if stream == None or len(stream) == 0: return obj
        if isinstance(stream, str): stream = py23_diffs.convert_store(stream)

        filters = encapsulate(obj.Filter)
        parms = encapsulate(obj.DecodeParms)

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
                if stream[-1] != '>': raise ValueError(f'/ASCIIHexDecode stream does not end with [>]: {[stream[-10:]]}')
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
                    try: stream1 = zlib.decompress(stream)
                    except: raise ValueError(f'failed to unzip stream: {stream[:10]}...{stream[-10:]}')
                    stream = stream1
                else:
                    earlyChange = int(get_key(parm, '/EarlyChange', '1'))
                    stream = b''.join(x for x in PdfFilter.lzw_decode(stream, earlyChange))

                predictor = int(get_key(parm, '/Predictor', '1'))
                colors = int(get_key(parm, '/Colors', '1'))
                bpc = int(get_key(parm, '/BitsPerComponent', '8'))
                columns = int(get_key(parm, '/Columns', '1'))

                if predictor == 1: # No predictor

                    pass

                elif 10 <= predictor <= 15: # PNG filters

                    # Predictors operate on rows whose size (width_bytes) is determined by
                    # the entries in obj.DecodeParm and not by the image parameters.
                    # So, in general rows != height, width_bytes != width + 1
                    width_bytes = (columns * colors * bpc + 7) // 8 + 1
                    rows = (len(stream) + width_bytes - 1 ) // width_bytes

                    # Pad with null bytes if needed
                    res = rows * width_bytes - len(stream)
                    if res: stream = stream + bytes([0]*res)

                    # For fastest decoding, make a PNG image from the stream and decode it using PIL
                    stream = PdfFilter.make_png_image(width_bytes-1,rows,8,'L',zlib.compress(stream),None).tobytes()

                    # Remove padding
                    if res: stream = stream[:-res]

                    # Fix PNG Predictor value: set it to 15 (fixes Multivalent bug)
                    # its actual value doesn't matter: PDF Ref. Sec. 3.3.3
                    if f < len(parms):
                        parms[f].Predictor = 15
                        obj.DecodeParms = decapsulate(parms)
                        warn(f'fixed PNG predictor: {predictor} --> 15')

                    # # For fastest decoding, make a PNG image from the stream and decode it using PIL
                    # if colors in [1,3]:

                    #     print('DEBUG', len(stream), width_bytes, height)
                    #     # print(f'DEBUG: {bpc}')
                    #     # if bpc != 1:
                    #     bpc=1
                    #     mode = 'L' if colors == 1 else 'RGB'
                    #     stream = PdfFilter.make_png_image(width,height,bpc,mode,zlib.compress(stream),None).tobytes()
                    #     # else:
                    #         # stream = PdfFilter.make_png_image(width_bytes-1,height,8,'L',zlib.compress(stream),None).tobytes()

                    # else:
                    #     # If # of colors is not 1 or 3 we transform each color component into a
                    #     # grayscale PNG image and decode it using PIL; the components are then reassembled
    
                    #     array = np.frombuffer(stream,dtype='uint8').reshape(height, width_bytes)

                    #     predVector = array[:,0].reshape(height,1)
                    #     array = array[:,1:].reshape(height,width,colors)
                    #     array = np.transpose(array,(2,0,1))

                    #     result = np.zeros(height*width*colors, dtype='uint8').reshape(colors,height,width)
                    #     for c in range(colors):
                    #         encoded = np.hstack((predVector,array[c])).tobytes()
                    #         decoded = PdfFilter.make_png_image(width,height,bpc,'L',zlib.compress(encoded),None).tobytes()
                    #         component = np.frombuffer(decoded,dtype='uint8').reshape(height, width)
                    #         result[c] = component

                    #     stream = np.transpose(result,(1,2,0)).tobytes()
                 
                elif predictor == 2: # TIFF Predictor 2 filter

                    # !!! THIS IS INCORRECTLY USING IMAGE WIDTH/HEIGHT !!!

                    array = PdfFilter.unpack_pixels(stream=stream, width=width, cpp=colors, bpc=bpc)
                    array = np.cumsum(array, axis=1, dtype = 'uint16' if bpc == 16 else 'uint8')
                    stream = PdfFilter.pack_pixels(array, bpc)

                else:
                    raise ValueError(f'unknown predictor: {predictor}')

            elif filter in ['/RunLengthDecode', '/RL']:

                stream = PdfFilter.rle_decode(stream)

            else:
                break

            # move to the next filter
            f += 1

        # there should be at most 1 (image-specific) filter left
        assert len(filters) - f <= 1

        result = obj.copy()
        result.Filter = decapsulate(filters[f:])
        result.DecodeParms = decapsulate(parms[f:])
        result.stream = py23_diffs.convert_load(stream)
        result.Length = len(result.stream)

        return result

    # -------------------------------------------------------------------- unpack_pixels()

    @staticmethod
    def unpack_pixels(stream:bytes, width:int, cpp:int, bpc:int, truncate = False):
        '''
        Converts a stream of pixel values into a numpy array.
        
        The stream is a sequence of lines, each line is a sequence of `width` pixels,
        each pixel is a sequence of `cpp` components (colors),
        each component is a sequence of `bpc` bits. All components are contiguous within each line
        and are padded at the end of the line to the nearest full byte.

        If `truncate` is True and there's a single extra byte at the end of the stream if that
        byte is the newline char. If `truncate` is False, the presence of any extra bytes bytes
        at the end of the stream raises an exception.

        The `dtype` of the returned numpy array is the either `uint8`, `uint16` or `uint32`, whichever is
        the smallest one that is sufficient to hold `bpc` bits.
        '''
        assert bpc <= 32
        samplesPerLine = width * cpp
        bytesPerLine = (samplesPerLine * bpc + 7) // 8
        height = len(stream) // bytesPerLine

        # Truncate
        extraBits = lambda: len(stream) % bytesPerLine if bytesPerLine > 1 else 0
        if truncate and extraBits() == 1 and stream[-1] == 10:
                warn(f"truncating stream of length {len(stream)}")
                stream = stream[:-1]
        if extraBits() != 0:
                raise ValueError(f'{extraBits()} extra bits in stream: length = {len(stream)}, bytesPerLine = {bytesPerLine}')

        # Determine dtype
        dtypes = {1:np.dtype('>u1'), 2:np.dtype('>u2'), 4:np.dtype('>u4')}
        if bpc in [8, 16, 32]:
            return np.frombuffer(stream, dtype=dtypes.get(bpc//8)).reshape(height, width, cpp)
        bytesPerSample = (bpc + 7) // 8
        if bytesPerSample == 3: bytesPerSample = 4 # no uint24 in numpy
        dtype = dtypes.get(bytesPerSample)

        # Determine padBitsPerSample
        padBitsPerSample = bytesPerSample * 8 - bpc

        # Create numpy array
        byte_array = np.frombuffer(stream, dtype='uint8')
        bit_array = np.unpackbits(byte_array).reshape(height,-1)[:, :samplesPerLine * bpc]
        bit_array = bit_array.reshape(height, samplesPerLine, bpc)
        bit_array_padded = np.pad(bit_array,
                                  pad_width=((0,0),(0,0),(padBitsPerSample, 0)),
                                  constant_values=((0,0),(0,0),(0,0)))
        byte_array = np.packbits(bit_array_padded.flatten())
        samples = np.frombuffer(byte_array.tobytes(), dtype = dtype)

        return samples.reshape(height, width, cpp)

    # -------------------------------------------------------------------- pack_pixels()

    @staticmethod
    def pack_pixels(array:np.ndarray, bpc:int):
        '''
        Converts a numpy array of shape (height, width, cpp) into a bytes stream of pixels,
        with bpc bits per pixel. The elements of the array should have values of the form N * M,
        where N = 2^{8-bpc} and M = 1..2^{bpc}-1. Rows of packed pixel values are padded at the end.
        The pixel values are contiguous within each row.
        '''
        assert bpc <= 32
        height, width = array.shape[:2]
        cpp = 1 if array.ndim == 2 else array.shape[2]

        if bpc % 8 != 0:
            array = array.reshape(height, width, cpp, 1)
            array = np.unpackbits(array, axis=3)[:,:,:,:bpc].reshape(height, width * cpp * bpc)
            array = np.packbits(array, axis=1)
        return array.tobytes()

    # -------------------------------------------------------------------- make_png_image()

    @staticmethod
    def make_png_image(width:int, height:int, bpc:int, mode:str, stream:bytes, palette:bytes):
        '''
        Creates a PNG PIL Image from the zlib (/Deflate) compressed bytes stream and, for mode == 'P', the palette bytes.
        PNG supports the following modes: 'L', 'RGB', 'P', 'LA', 'RGBA' (see 'color types' in the PNG spec.)
        '''
        png_color_types = {'1':0, 'L':0, 'RGB':2, 'P':3, 'LA':4, 'RGBA':6} # LA & RGBA are not part of the PDF standard (yet)
        if mode not in png_color_types: err(f'unsupported mode: {mode}')
        color_type = png_color_types[mode]

        PNG_header = bytes([137,80,78,71,13,10,26,10])

        # print([width, height, bpc, color_type])

        IHDR = PdfFilter.make_png_chunk(b'IHDR', struct.pack(b'!LLBBBBB', width, height, bpc, color_type, 0, 0, 0))
        PLTE = PdfFilter.make_png_chunk(b'PLTE', palette) if mode == 'P' else b''
        IDAT = PdfFilter.make_png_chunk(b'IDAT', stream)
        IEND = PdfFilter.make_png_chunk(b'IEND', b'')

        # msg(f'Length of PLTE: {len(PLTE)}')
        return Image.open(BytesIO(PNG_header + IHDR + PLTE + IDAT + IEND))

    @staticmethod
    def make_png_chunk(chunk_type:bytes, chunk_data:bytes):
        chunk_len = struct.pack('!L',len(chunk_data))
        chunk_crc = struct.pack('!L',zlib.crc32(chunk_type + chunk_data))
        return chunk_len + chunk_type + chunk_data + chunk_crc

    # -------------------------------------------------------------------- rle_decode()

    @staticmethod
    def rle_decode(string:bytes):
        # PDF Ref. 1.7 Sec. 3.3.4
        s = b''
        i = 0
        while i < len(string):
            runLength = string[i]
            if runLength == 128: break
            elif 0 <= runLength < 128:
                j = (i + 1) + (runLength + 1)
                s += string[i+1, j]
                i = j
            else:
                s += string[i+1] * (257 - runLength)
                i += 2
        string = s

    # -------------------------------------------------------------------- rle_encode()

    @staticmethod
    def rle_encode(string:bytes):
        # PDF Ref. 1.7 Sec. 3.3.4
        i,j,N = 0,1,len(string)
        MAX_SEQ_LENGTH = 128

        runs = []
        while i<N:
            while j<N and string[j] == string[i] and j-i < MAX_SEQ_LENGTH: j += 1
            runs.append((string[i],j-i))
            i,j = j,j+1

        seq = b''
        result = b''
        for char, runLength in runs:
            if len(seq) + runLength <= MAX_SEQ_LENGTH and (len(seq) == 0 and runLength < 2 or runLength < 3):
                seq += bytes([char]) * runLength
            else:
                if len(seq)>0: result += bytes([len(seq)-1]) + seq; seq = b''
                result += bytes([257 - runLength, char])
        if len(seq)>0: result += bytes([len(seq)-1]) + seq

        return result
    
 
    # -------------------------------------------------------------------- lzw_decode()

    @staticmethod
    def lzw_decode(byteString:bytes, earlyChange = 1):
        '''
        Decodes an LZW-encoded byteString into a generator of bytes (See TIFF Rev. 6.0 & Adobe PDF Ref.)
        '''        
        # The initial table
        table = [bytes([i]) for i in range(256)] + [b''] * (4096 - 256)

        bits = 0
        chunk = 0
        string = None
        L = 258
        codeBits = 9
        maxSize = 512

        for byte in byteString:

            assert 9 <= codeBits <= 12
            assert L < 4096

            # Get the code
            chunk = (chunk << 8) + byte
            bits += 8
            extraBits = bits - codeBits

            if extraBits < 0:
                continue

            code = chunk >> extraBits
            chunk = chunk - (code << extraBits)
            bits = extraBits

            # Interpret the code
            if code == 256:
                string = None
                L = 258
                codeBits = 9
                maxSize = 512
            elif code == 257:
                break
            else:
                if code > L:
                    raise ValueError(f'bad code: {code}, table length: {L}')
                if code < L:
                    entry = table[code]
                    if string:
                        table[L] = string + entry[:1]
                        L += 1
                        if (L + earlyChange) // maxSize: codeBits += 1; maxSize *= 2
                else:
                    assert string
                    entry = string + string[:1]
                    table[L] = entry
                    L += 1
                    if (L + earlyChange) // maxSize: codeBits += 1; maxSize *= 2

                yield entry
                string = entry
  
