#!/usr/bin/env python3

import struct, zlib, base64, re
from io import BytesIO
import numpy as np

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from pdfrw import PdfName, PdfDict, IndirectPdfDict, py23_diffs
from .common import warn, get_key, encapsulate, decapsulate

# ========================================================================== PdfFilter

class PdfFilter:

    FILTER_NAMES = {
        '/ASCIIHexDecode':'AHx',
        '/AHx':'AHx',
        '/ASCII85Decode':'A85',
        '/A85':'A85',
        '/FlateDecode':'ZIP',
        '/Fl':'ZIP',
        '/LZWDecode':'LZW',
        '/LZW':'LZW',
        '/RunLengthDecode':'RLE',
        '/RL':'RLE',
        '/DCTDecode':'JPEG',
        '/CCITTFaxDecode':'FAX',
        '/JBIG2Decode':'JB2',
        '/JPXDecode':'JPX',
        'null':'None',
        None:'None'
    }

    # -------------------------------------------------------------------- filters_as_list()

    @staticmethod
    def filters_as_list(obj:IndirectPdfDict):
        '''
        Returns `obj.Filter` as a list of short filter names; see PdfFilter.FILTER_NAMES.
        '''
        return [PdfFilter.FILTER_NAMES.get(f, f[1:]) for f in encapsulate(obj.Filter)]

    # -------------------------------------------------------------------- uncompress()

    @staticmethod
    def uncompress(obj:IndirectPdfDict,
                   fixFlateStreams:bool = False):
        '''
        Returns an uncompressed version of `obj`. Supports objects compressed with one or more of the following filters:

        * `/ASCIIHexDecode` and `/ASCII85Decode`
        * `/FlateDecode` and `/LZWDecode` (all values of /Predictor are supported)
        * `/RunLengthDecode`

        Normally, the function does not modify the `obj` itself. However

        * The function is able to uncompress `/FlateDecode` streams that use RFC1950 (deflate)
        algorithm, which is different from the more well-known RFC1951 (zlib) algorithm.
        For such RFC1950 streams it successfully returns the result.
        If `fixFlateStreams` is `True` it will also fix the `obj.stream` itself by
        turning it into an RFC1951 (zlib) stream
        (together with updating other `obj` entries: `/Filters`, `/DecodeParms`, `/Length`).

        '''
        assert isinstance(obj, PdfDict)
        stream = obj.stream
        if stream == None or len(stream) == 0: return obj
        if isinstance(stream, str): stream = py23_diffs.convert_store(stream)

        filters = encapsulate(obj.Filter)
        parms = encapsulate(obj.DecodeParms)

        streamNeedsFixing = False
        bpcReduced = False

        f = 0
        while f < len(filters):

            if bpcReduced:
                raise ValueError(f'PIL limitation: a PNG filter with bpc > 8 cannot be followed by another filter')

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

                # First, uncompress; PDF Ref. 1.7 Sec. 3.3.3

                if filter in ['/FlateDecode','/Fl']:
                    try: stream = zlib.decompress(stream)
                    except:
                        try:
                            warn(f'zlib.decompress() failed; recovering')
                            stream = zlib.decompress(stream[2:], wbits = -15)
                            streamNeedsFixing = True
                        except:
                            raise ValueError(f'failed to inflate stream')
                else:
                    earlyChange = int(get_key(parm, '/EarlyChange', '1'))
                    stream = b''.join(x for x in PdfFilter.lzw_decode(stream, earlyChange))

                # Second, deal with the /Predictor filters

                predictor = int(get_key(parm, '/Predictor', '1'))
                columns = int(get_key(parm, '/Columns', '1'))
                colors = int(get_key(parm, '/Colors', '1'))
                bpc = int(get_key(parm, '/BitsPerComponent', '8'))

                if predictor == 1: # No predictor

                    pass

                elif 10 <= predictor <= 15: # PNG filters

                    if bpc > 8:
                        if obj.Subtype != '/Image':
                            raise ValueError(f'PIL limitation: /FlateDecode-compressed streams with bpc = {bpc} are only supported for images')
                        bpcReduced = True

                    stream = PdfFilter.remove_png_predictor(stream = stream,
                                                            columns = columns,
                                                            colors = colors,
                                                            bpc = bpc)
                                     
                elif predictor == 2: # TIFF Predictor 2 filter

                    array = PdfFilter.unpack_pixels(stream=stream, width=columns, cpp=colors, bpc=bpc)
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

        if fixFlateStreams and streamNeedsFixing:
            warn(f'fixed Flate stream')
            obj.Filter = PdfName.FlateDecode
            obj.DecodeParms = None
            obj.stream = py23_diffs.convert_load(zlib.compress(stream))
            obj.Length = len(obj.stream)

        if result.BitsPerComponent is not None and int(result.BitsPerComponent) > 8:
            warn(f'PIL is limited to 8 bpc; changing /BitsPerComponent: {result.BitsPerComponent} -> 8')
            result.BitsPerComponent = 8

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

        If `truncate` is True and there's a single extra byte at the end of the stream and if that
        byte is the newline char this byte is truncated. After that, the presence of any extra bytes
        at the end of the stream raises an exception.

        The `dtype` of the returned numpy array is the either `uint8`, `uint16` or `uint32`, whichever is
        the smallest one that is sufficient to hold `bpc` bits.
        '''
        assert bpc <= 32
        samplesPerLine = width * cpp
        bytesPerLine = (samplesPerLine * bpc + 7) // 8
        height = len(stream) // bytesPerLine

        # Truncate
        e = len(stream) - bytesPerLine * height
        if truncate and e == 1 and stream[-1] == 10:
            warn(f"truncating stream of length {len(stream)}")
            stream = stream[:-1]
        e = len(stream) - bytesPerLine * height
        if e > 0:
            warn(f'{e} extra bytes in stream: length = {len(stream)}, bytesPerLine = {bytesPerLine}; truncating')
            stream = stream[:-e]

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

    @staticmethod
    def remove_png_predictor(stream:bytes, columns:int, colors:int, bpc:int):
        '''
        '''
        # Predictors operate on blocks whose size is determined by
        # the entries in obj.DecodeParm and not by the image parameters!
        # So, in general n_blocks != height, block_size != width + 1
        row_size = (columns * colors * bpc + 7) // 8 + 1 # +1 is the predictor byte at the start
        n_rows = (len(stream) + row_size - 1 ) // row_size

        # Pad with null bytes if needed
        s = stream
        if res := n_rows * row_size - len(stream):
            s = stream + bytes([0]*res)

        if colors in [1,2,3,4]:

            mode = {1:'L', 2:'LA', 3:'RGB', 4:'RGBA'}.get(colors)

            # For fastest decoding, make a PNG image from the stream and decode it using PIL
            stream = PdfFilter.make_png_image(width = columns,
                                            height = n_rows,
                                            bpc = bpc,
                                            mode = mode,
                                            stream = zlib.compress(stream),
                                            palette = None).tobytes()
        else:

            # Raise an exception for now to catch an actual PDF with colors > 4 to test the code below
            raise ValueError(f'unsupported number of /Colors = {colors} with filter: {filter}')

            # # If # of colors is not 1 or 3 we transform each color component into a
            # # grayscale PNG image and decode it using PIL; the components are then reassembled

            # array = np.frombuffer(stream, dtype='uint8').reshape(n_rows, row_size)

            # predVector = array[:,0].reshape(n_rows, 1)
            # array = array[:,1:].reshape(n_rows, row_size, colors)
            # array = np.transpose(array,(2,0,1))

            # result = np.zeros(height*width*colors, dtype='uint8').reshape(colors,height,width)
            # for c in range(colors):
            #     encoded = np.hstack((predVector,array[c])).tobytes()
            #     decoded = PdfFilter.make_png_image(width,height,bpc,'L',zlib.compress(encoded),None).tobytes()
            #     component = np.frombuffer(decoded,dtype='uint8').reshape(height, width)
            #     result[c] = component

            # stream = np.transpose(result,(1,2,0)).tobytes()


        # Remove padding
        if res:
            stream = stream[:-res]

        return stream 
           

    # -------------------------------------------------------------------- make_png_image()

    @staticmethod
    def make_png_image(width:int, height:int, bpc:int, mode:str, stream:bytes, palette:bytes):
        '''
        Creates a PNG PIL Image from the zlib (/Deflate) compressed bytes stream and, for mode == 'P', the palette bytes.
        PNG supports the following modes: 'L', 'RGB', 'P', 'LA', 'RGBA' (see 'color types' in the PNG spec.)
        '''
        # LA & RGBA are not part of the PDF standard (yet)
        png_color_types = {'1':0, 'L':0, 'RGB':2, 'P':3, 'LA':4, 'RGBA':6}

        color_type = png_color_types.get(mode)
        if mode is None:
            raise ValueError(f'unsupported mode: {mode}')

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
        result = []
        read = 0
        repeat = 0
        for b in string:
            if read > 0:
                result.append(b)
                read -= 1
            elif repeat > 0:
                result += [b]*repeat
                repeat = 0
            elif b == 128:
                break
            elif 0 <= b <= 127:
                read = b + 1
            else:
                repeat = 257 - b
        return bytes(result)

    # -------------------------------------------------------------------- rle_encode()

    @staticmethod
    def rle_encode(string:bytes):
        # PDF Ref. 1.7 Sec. 3.3.4
        MAX_SEQ_LENGTH = 128

        if len(string) == 0:
            return b''

        # Convert a string to a list of runs
        runs = []
        b_prev = string[0]
        runLength = 0
        for b in string:
            if b == b_prev and runLength < MAX_SEQ_LENGTH:
                runLength += 1
            else:
                runs.append((b_prev, runLength))
                b_prev = b
                runLength = 1
        runs.append((b_prev, runLength))
 
        # Optimize and encode the list of runs
        seq = []
        result = []
        for char, runLength in runs:
            if len(seq) + runLength <= MAX_SEQ_LENGTH and (len(seq) == 0 and runLength < 2 or runLength < 3):
                seq += [char] * runLength
            else:
                if len(seq)>0:
                    result += [len(seq)-1] + seq
                    seq = []
                result += [257 - runLength, char]
        if len(seq)>0:
            result += [len(seq)-1] + seq

        return bytes(result)
    
 
    # -------------------------------------------------------------------- lzw_decode()

    @staticmethod
    def lzw_decode(byteString:bytes, earlyChange = 1):
        '''
        Decodes an LZW-encoded `byteString` into a generator of bytes (See TIFF Rev. 6.0 & Adobe PDF Ref.)
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

            if not (9 <= codeBits <= 12 and L <= 4096):
                warn(f'decoding error, stream truncated')
                return

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

                        if codeBits < 12 and L >= maxSize - earlyChange:
                            codeBits += 1
                            maxSize <<= 1

                else:
                    assert string
                    entry = string + string[:1]
                    table[L] = entry
                    L += 1

                    if codeBits < 12 and L >= maxSize - earlyChange:
                        codeBits += 1
                        maxSize <<= 1

                yield entry
                string = entry
  
