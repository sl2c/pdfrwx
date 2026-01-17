#!/usr/bin/env python3

# Write this man about the bug in PIL Image.paste()!
# https://sudonull.com/post/129265-Alpha_composite-optimization-history-in-Pillow-20
# https://habr.com/ru/post/98743/
# https://github.com/homm/
# https://twitter.com/wouldntfix

from pdfrw import PdfWriter, PdfObject, PdfName, PdfArray, PdfDict, IndirectPdfDict
from pdfrw import py23_diffs # Ugly, but necessary: https://github.com/pmaupin/pdfrw/issues/161

from pdfrwx.pdfreaderx import PdfReaderX
from pdfrwx.common import err, msg, warn, eprint, get_key, get_any_key, \
                            getExecPath, encapsulate, decapsulate, formatWithCommas, formatSize

from pdfrwx.pdffilter import PdfFilter
from pdfrwx.pdfobjects import PdfObjects
from pdfrwx.pdffunctionparser import PdfFunction

from simage import SImage

from attrdict.attrdict import AttrDict

from typing import Union

import argparse, struct, zlib, base64, sys, re, os, subprocess, tempfile

from io import BytesIO
import numpy as np
import shutil

from PIL import Image, TiffImagePlugin, ImageChops, ImageCms
Image.MAX_IMAGE_PIXELS = None

from glymur import Jp2k

from pprint import pprint

CMYK_DEFAULT_ICC_PROFILE = open(os.path.join(getExecPath(),'color_profiles/USWebCoatedSWOP.icc'), 'rb').read()

# ============================================================== timeit

from time import time
tStart = time()
def timeit(s): print(f'TIME: {s}: {time()-tStart:0.3f} sec')

# ============================================================== class OS

class OS:

    def execute(cmd:list[str]):
        '''
        Execute command + arguments given a list: cmd[0] is the executable's name,
        cmd[1:] are the command's arguments. On Windows, the .exe extension should be omitted.
        '''
        exe = '.exe' if sys.platform == 'win32' else ''
        result = subprocess.run([cmd[0] + exe] + cmd[1:], capture_output=True)
        if result.returncode != 0: sys.exit(f'execute() error: {result.stderr.decode()}')
        return result.stdout.decode()

# ========================================================================== class PdfColorSpace

# Typedef
    
CS_TYPE = Union[PdfName, PdfArray]

class PdfColorSpace:

    # A mapping from color space name to components per pixel; None means that cpp varies
    spaces = {
        '/DeviceGray': 1, '/CalGray': 1, '/Indexed': 1, '/Separation': 1,
        '/DeviceRGB': 3, '/CalRGB': 3, '/Lab': 3, '/DeviceCMYK': 4,
        '/ICCBased': None, '/DeviceN': None, '/NChannel': None
    }

    # A mapping from components per pixel to PIL image modes
    modes = {1:'L', 3:'RGB', 4:'CMYK'}

    @staticmethod
    def toStr(cs:CS_TYPE):
        return cs if isinstance(cs, str) \
            else [type(x) if type(x) in [PdfArray, PdfDict, bytes] else x for x in cs]

    @staticmethod
    def get_name(cs:CS_TYPE):
        '''
        Returns the name of the color space (one of the PdfColorSpace.spaces.keys())
        '''
        name = cs if not isinstance(cs,PdfArray) \
                    else '/NChannel' if cs[0] == '/DeviceN' and len(cs) == 5 and cs[4].Subtype == '/NChannel' \
                    else cs[0]
        if name != None and name not in PdfColorSpace.spaces:
            raise ValueError(f'bad color space: {cs}')
        return name

    @staticmethod
    def get_cpp(cs:CS_TYPE):
        '''
        Return the number of components per pixel for a give color space.
        This always returns an int unless `cs` is None, in which case None is returned.
        '''
        name = PdfColorSpace.get_name(cs)
        return int(cs[1].N) if name == '/ICCBased' else \
                len(cs[1]) if name in ['/DeviceN', '/NChannel'] else \
                PdfColorSpace.spaces.get(name)

    @staticmethod
    def get_mode(cs:CS_TYPE, bpc:int):
        '''
        '''
        return '1' if bpc == 1 else PdfColorSpace.modes.get(PdfColorSpace.get_cpp(cs))

    @staticmethod
    def get_icc_profile(cs:CS_TYPE):
        '''
        For calibrated /CalGray, /CalRGB, /Lab and /ICCBased color spaces, returns
        the ICC color profile as a bytes object. For the /Indexed color space,
        returns the ICC color profile of the 'base' color space, if present, or None.
        For all other color spaces returns None.
        '''
        name = PdfColorSpace.get_name(cs)
        if name == '/Indexed': return PdfColorSpace.get_icc_profile(cs[1])
        elif name == '/ICCBased': return py23_diffs.convert_store(PdfFilter.uncompress(cs[1]).stream)
        elif name in ['/CalGray','/CalRGB', '/Lab']: return PdfColorSpace.create_profile(cs)
        else: None

    @staticmethod
    def get_palette(cs:CS_TYPE):
        '''
        Return a tuple (paletteColorSpace, paletteArray), where paletteArray is a numpy array of shape (-1, cpp),
        cpp is the number of color components in the paletteColorSpace.
        '''
        if PdfColorSpace.get_name(cs) != '/Indexed': return None, None
        name, base, hival, pal = cs

        # pal is either a IndirectPdfDict, with palette in its stream, or a PdfString
        pal = py23_diffs.convert_store(PdfFilter.uncompress(pal).stream) if isinstance(pal,PdfDict) \
                else pal.to_bytes()

        # Fix the tail if necessary
        palette_cpp = PdfColorSpace.get_cpp(base)
        size = (int(hival) + 1) * palette_cpp
        while len(pal) > size and pal[-1] in b'\n\r':
            pal = pal[:-1]
        if len(pal) != size:
            raise ValueError(f'palette size mismatch: expected {size}, got {len(pal)}')

        pal_array = np.frombuffer(pal, 'uint8')
        if palette_cpp != 1:
            pal_array = pal_array.reshape(-1, palette_cpp)
        return base, pal_array

    @staticmethod
    def create_profile(cs:CS_TYPE):
        '''
        Creates an ICC profile for a given color space, when it's one of ['/CalGray', ..], ['/CalRGB', ..], ['/Lab', ..]
        '''
        name = PdfColorSpace.get_name(cs)
        if name not in ['/CalGray','/CalRGB', '/Lab']: return None
        dic = cs[1]

        # Calculate the CIR xyY values of the white point
        X,Y,Z = [float(v) for v in dic.WhitePoint]
        x,y = X/(X+Y+Z), Y/(X+Y+Z)

        try:
            # LittleCMS allows to specify the white point directly when creating a Lab profile
            import littlecms as lc
        except:
            lc = None

        if lc:
            ctx = lc.cmsCreateContext(None, None)
            white = lc.cmsCIExyY()
            white.x, white.y, white.Y = x, y, Y

            if name == '/Lab':

                profile = lc.cmsCreateLab2Profile(white)

            if name == '/CalGray':

                gamma = float(dic.Gamma) if dic.Gamma != None else 1
                transferFunction = lc.cmsBuildGamma(ctx,gamma)
                profile = lc.cmsCreateGrayProfile(white, transferFunction)

            if name == '/CalRGB':

                primaries = [float(v) for v in dic.Matrix] if dic.Matrix != None else [1,0,0,0,1,0,0,0,1]
                primaries = [primaries[3*i:3*i+3] for i in range(3)]
                primaries = [[X/(X+Y+Z),Y/(X+Y+Z),Y] for X,Y,Z in primaries]
                gamma = [float(v) for v in dic.Gamma] if dic.Gamma != None else [1,1,1]
                transferFunction = [lc.cmsBuildGamma(ctx, g) for g in gamma]

                pt = lc.cmsCIExyYTRIPLE()
                pt.Red = lc.cmsCIExyY(); pt.Red.x, pt.Red.y, pt.Red.Y = primaries[0]
                pt.Green = lc.cmsCIExyY(); pt.Green.x, pt.Green.y, pt.Green.Y = primaries[1]
                pt.Blue = lc.cmsCIExyY(); pt.Blue.x, pt.Blue.y, pt.Blue.Y = primaries[2]
                
                profile = lc.cmsCreateRGBProfile(white, pt, transferFunction)

            with tempfile.TemporaryDirectory() as tmp:

                T = lambda fileName: os.path.join(tmp, fileName)
                lc.cmsSaveProfileToFile(profile, T('profile.cms'))
                icc_profile = open(T('profile.cms'),'rb').read()

            # with BytesIO() as bs:
            #     lc.cmsSaveProfileToStream(profile, bs)
            #     icc_profile = bs.getvalue()
        else:
            warn('LittleCMS not installed')
            warn('using ImageCMS with the (approximate) McCamy\'s formula for correlated temperature')
            n = (x - 0.3320) / (0.1858 - y)
            CCT = 449 * n*n*n + 3525 * n*n + 6823.3 * n + 5520.33
            if name == '/Lab':
                icc_profile = ImageCms.createProfile('LAB', colorTemp = CCT)
            if name == '/CalRGB':
                return None

        if dic.BlackPoint != None: warn(f'/BlackPoint ignored: {cs}')
        if dic.Range != None: warn(f'/Range ignored: {cs}')

        return icc_profile
    
    @staticmethod
    def reduce(cs:CS_TYPE, array:np.ndarray, Decode:list[float] = None, bpc:int = 8, mask:PdfDict = None):
        '''
        Reduces the image array:
        
        * decodes it using the specified Decode array if it's not None or otherwise
        the default decode array, which is determined based on the specified colorspace and bpc;
        see PDF Ref. 1.7 sec. 4.8.4, Table 4.40
        * un-multiplies alpha if mask.Matte is not None;
        * if the colorspace is one of `/Separation`, `/DeviceN`, `/NChannel`, reduces the colorspace
        by remapping the image array to the corresponding alternate colorspace;
        * encodes the image array using the default Decode array based on the target colorspace
        (the original or the alternate one, depending on whether the colorspace has been reduced
        in the previous step) and the value of bpc == 8.
        
        Returns the tuple (newColorSpace, array).
        '''
        SPECIAL = ['/Separation', '/DeviceN', '/NChannel']

        name = PdfColorSpace.get_name(cs)
        h,w = array.shape[:2]
        assert array.ndim in [2,3]

        msg(f'applying color space: {name}')
        Matte = mask.Matte if mask else None
        default_decode = PdfDecodeArray.get_default(cs, bpc)
        applyDecode = bpc != 8 or Matte or Decode and ((name in SPECIAL) or (Decode != default_decode))

        # Decode
        if applyDecode:
            Decode = Decode or default_decode
            array = PdfDecodeArray.decode(array, Decode, bpc)

        if bpc == 1:
            if cs != '/DeviceGray':
                warn(f'cannot apply colorspace {name} to a bitonal image; skipping')
            return cs, array

        # Matte
        if Matte:

            # Get matte array
            matte_array = np.array([float(x) for x in Matte])[np.newaxis, np.newaxis, :]

            # Render alpha
            alpha = PdfImage(obj = mask)
            alpha.render()

            # Resize alpha
            if alpha.w() != w or alpha.h() != h:
                msg(f'resizing alpha: {alpha.w()} x {alpha.h()} --> {w} x {h}')
                alpha.resize(w,h)

            # Get decoded alpha array
            alpha_array = alpha.get_array()
            alpha_decode_default = PdfDecodeArray.get_default(alpha.ColorSpace, alpha.bpc)
            alpha_array = PdfDecodeArray.decode(alpha.get_array(), alpha_decode_default, alpha.bpc)
            alpha_array = alpha_array[:,:,np.newaxis]

            # Un-multiply alpha
            msg(f'Unmultiplying alpha; Matte = {Matte}')
            array =  matte_array + (array - matte_array) / np.maximum(alpha_array, 0.000001)

            # Remove Matte entry from the mask
            mask.Matte = None

        # These color spaces are function-based, so apply this function
        if name in SPECIAL:

            altColorSpace, colorTransformFunction = cs[2], cs[3]
            array = array if array.ndim == 3 else np.dstack([array])
            stack = np.moveaxis(array,-1,0)
            stack = PdfFunction(colorTransformFunction).process(stack)
            array = np.moveaxis(stack,0,-1)
            cs = altColorSpace

        # Encode with bpc == 8
        if applyDecode:
            array = PdfDecodeArray.encode(array, PdfDecodeArray.get_default(cs, 8))

        return cs, array

    # @staticmethod
    # def apply_default_page_colorspace_icc_profile(page:PdfDict, cs:CS_TYPE):
    #     '''
    #     If image.inf['icc_profile'] == None, apply the ICC profile from the page's default color space
    #     (an entry in page.Resources.Colorspace, if present) to the image (in-place).
    #     '''
    #     if page == None or image.info.get('icc_profile'): return
    #     try:
    #         default_cs = page.Resources.ColorSpace[cs]
    #     except:
    #         pass

    # @staticmethod
    # def make_indexed_colorspace(palette:bytes, baseColorspace):
    #     '''
    #     Returns an /Indexed colorspace based on the image's palette and baseColorspace, or
    #     None if image has no palette.
    #     '''
    #     # palette = image.getpalette()
    #     # if palette == None: return None
    #     # palette = b''.join(v.to_bytes(1,'big') for v in image.getpalette())

    #     cpp = PdfColorSpace.get_cpp(baseColorspace)
    #     assert len(palette) % cpp == 0
        
    #     palette_xobj = IndirectPdfDict(Filter = PdfName.FlateDecode,
    #                                    stream = zlib.compress(palette).decode('Latin-1'))
    #     return PdfArray([PdfName.Indexed, baseColorspace, len(palette) // cpp, palette_xobj])

    @staticmethod
    def make_icc_based_colorspace(icc_profile:bytes, N:int):
        '''
        Returns an /ICCBased colorspace made from the icc_profile with N being the number of color components.
        '''
        icc_xobj = IndirectPdfDict(
            N = N,
            Filter = PdfName.FlateDecode,
            stream = zlib.compress(icc_profile).decode('Latin-1')
        )
        return PdfArray([PdfName.ICCBased, icc_xobj])
    
# ========================================================================== class PdfDecodeArray

class PdfDecodeArray:

    '''
    The class facilitates processing of image.Decode arrays.
    '''

    @staticmethod
    def get_actual(Decode:PdfArray, cs:CS_TYPE, bpc:int):
        '''
        Returns actual Decode array as list[float] if self.image.Decode != None, or self.get_default(colorSpace) otherwise.
        ''' 
        FLOAT = lambda array: [float(a) for a in array]
        return FLOAT(Decode) if Decode != None else PdfDecodeArray.get_default(cs, bpc)

    @staticmethod
    def get_default(cs:CS_TYPE, bpc:int):
        '''
        Returns the default Decode array for a give color space; see Adobe PDF Ref. 1.7, table 4.40 (p.345)
        ''' 
        FLOAT = lambda array: [float(a) for a in array]

        name = PdfColorSpace.get_name(cs)
        cpp = PdfColorSpace.get_cpp(cs)

        if name == '/Lab':
            d = [0.0, 100.0] + (FLOAT(cs[1].Range) if cs[1].Range != None else [-100.0, 100.0] * 2)
        elif name == '/ICCBased' and cs[1].Range != None:
            d = FLOAT(cs[1].Range)
        elif name == '/Indexed':
            d = [0, (1<<bpc)-1]
        elif name == '/Pattern':
            d = None
        else:
            d = [0.0, 1.0] * cpp

        return d

    @staticmethod
    def decode(array:np.ndarray, Decode:list[float], bpc:int):
        '''
        Translates a numpy array from encoded ("stream space": int, 1..2**bpc -1)
        to decoded ("image space", float: mostly 0.0..1.0) representation.
        The array should be (D+1) dimensional, where D is the # of dimensions of the image
        and the last dimension is for the color index.
        '''
        assert Decode
        msg(f'applying Decode: {Decode}')

        if bpc == 1:
            assert Decode == [0,1] or Decode == [1,0]
            return array if Decode == [0,1] else np.logical_not(array)

        INTERPOLATE = lambda x, xmin, xmax, ymin, ymax: ymin + ((x - xmin) * (ymax - ymin)) / (xmax - xmin)
        iMax = float((1<<bpc) - 1)
        if array.ndim == 2: array = np.dstack([array])
        N = array.shape[-1] # number of colors
        assert len(Decode) == 2*N
        array = np.clip(array, 0, iMax).astype(float)
        for i in range(N):
            array[...,i] = INTERPOLATE(array[...,i], 0.0, iMax, Decode[2*i], Decode[2*i + 1])
        return array if N > 1 else array[...,0]

    @staticmethod
    def encode(array:np.ndarray, Decode:list[float]):
        '''
        Translates a numpy array from decoded ("image space", float: mostly 0.0..1.0)
        to encoded ("stream space", uint8: 0..255) representation.
        '''
        msg(f'applying Encode: {Decode}')

        if array.dtype == bool:
            assert Decode == [0,1] or Decode == [1,0]
            return array if Decode == [0,1] else np.logical_not(array)

        INTERPOLATE = lambda x, xmin, xmax, ymin, ymax: ymin + ((x - xmin) * (ymax - ymin)) / (xmax - xmin)
        if array.ndim == 2: array = np.dstack([array])
        N = array.shape[-1] # number of colors
        assert len(Decode) == 2*N
        for i in range(N):
            array[...,i] = INTERPOLATE(array[...,i].astype(float), Decode[2*i], Decode[2*i + 1], 0, 255)
        array = np.clip(np.round(array), 0, 255).astype(np.uint8)
        return array if N > 1 else array[...,0]


# ========================================================================== class PdfDecodedImage

class PdfImage(AttrDict):

    formats = ['array', 'pil', 'tiff', 'jbig2', 'jpeg', 'jp2', 'png']

    def __init__(self,
                 obj:IndirectPdfDict = None,
                 pil:Image.Image = None,
                 array:np.ndarray = None,
                 jpeg:bytes = None,
                 jp2:bytes = None):
        '''
        Creates a PdfImage.
        '''
        if sum(1 for x in [obj, pil, array,jp2,jpeg] if x is not None) != 1:
            raise ValueError(f'exactly one of the [pil, array, obj] arguments should be given to constructor')
        
        self.bpc = None
        self.ColorSpace = None
        self.dpi = None

        self.pil = pil
        self.array = array
        self.jpeg = jpeg
        self.jp2 = None

        self.Decode = None
        self.Mask = None
        self.SMask = None

        self.isRendered = False

        if obj:

            assert obj.Subtype in ['/Image', PdfName.Image]
            if obj.stream == None: raise ValueError(f'image has no stream: {obj}')

            obj = PdfFilter.uncompress(obj)

            self.Decode = obj.Decode
            self.ColorSpace = PdfImage.xobject_get_cs(obj)
            self.bpc = PdfImage.xobject_get_bpc(obj)

            def toBytes(s:str): return s.encode('Latin-1')
            stream = toBytes(obj.stream)

            if obj.Filter == None:

                width, height = int(obj.Width), int(obj.Height)
                cpp = self.get_cpp()

                # # Sometimes /BitsPerComponent is incorrect
                # bytesPerLine = len(stream) / height
                # if bytesPerLine == int(bytesPerLine):
                #     bpc_implied = (bytesPerLine * 8) / (width * cpp) if len(stream) > 1 else 1
                #     if bpc_implied != self.bpc and int(bpc_implied) == bpc_implied:
                #         warn(f'replacing bad image xobject\'s bpc = {self.bpc} with the implied bpc = {int(bpc_implied)}')
                #         warn(f'images size: {width}x{height}, cpp: {cpp}, stream length: {len(stream)}')
                #         warn(f'{stream}')
                #         self.bpc = int(bpc_implied)

                array = PdfFilter.unpack_pixels(stream, width, cpp, self.bpc, truncate = True)
                if array.shape[0] > height:
                    warn(f'more rows in stream than expected: {array.shape[0]} vs {height}; truncating')
                    array = array[:height]
                if array.shape[0] != height:
                    raise ValueError(f'image height mismatch: expected {height}, read {array.shape[0]}, bpc = {self.bpc}')
                array = SImage.normalize(array)
                if self.bpc == 1: array = array.astype(bool)
                self.set_array(array)

            elif obj.Filter == '/CCITTFaxDecode':

                parm = obj.DecodeParms
                assert parm

                K = int(parm.K or '0')
                EncodedByteAlign = parm.EncodedByteAlign == PdfObject('true')

                if K == -1 and EncodedByteAlign:

                    from pdfrwx.ccitt import Group4Decoder
                    decoder = Group4Decoder()
                    Columns = int(parm.Columns)

                    result = decoder.decode(data = stream,
                                            Columns = Columns,
                                            EncodedByteAlign = EncodedByteAlign)

                    width, height = int(obj.Width), int(obj.Height)
                    pil = Image.frombytes('1',(width,height),result)
                    self.set_pil(pil)

                    if parm.BlackIs1 != PdfObject('true'):
                        msg(f'inverting /CCITTFaxDecode image with /EncodedByteAlign = true because BlackIs1 is {parm.BlackIs1}')
                        self.set_pil(ImageChops.invert(self.pil))

                else:

                    if EncodedByteAlign:
                        warn(f'*** /CCITTFaxDecode Group3 (T4) decompression with /EncodedByteAlign is in beta-testing, check results ***')

                    header = PdfImage._tiff_make_header(obj)
                    self.set_pil(Image.open(BytesIO(header + stream)))

            elif obj.Filter == '/JBIG2Decode':

                jbig2_locals = stream
                try: jbig2_globals = toBytes(PdfFilter.uncompress(obj.DecodeParms.JBIG2Globals).stream)
                except: jbig2_globals = None
                self.set_pil(PdfImage._jbig2_read([jbig2_locals, jbig2_globals]))

            elif obj.Filter == '/DCTDecode': 

                pil = Image.open(BytesIO(stream))
                self.set_pil(pil)
                self.jpeg = stream # original

            elif obj.Filter == '/JPXDecode':

                self.read_jp2_to_array(stream)
                self.jp2 = stream # original

            else:
                raise ValueError(f'bad filter: {obj.Filter}')

            if obj.DecodeParms and obj.Filter in ['/DCTDecode', '/JPXDecode', None]:
                warn(f'ignoring /DecodeParms: {obj.DecodeParms}')

            # Handle transparency
            
            self.Mask  = obj.Mask
            self.SMask = obj.SMask

            # Invert /DeviceCMYK & /DeviceN JPEGs
            if self.pil and self.pil.mode == 'CMYK' and self.pil.format == 'JPEG':
                msg(f'inverting CMYK JPEG after decoding')
                self.set_pil(ImageChops.invert(self.pil))

            # # Invert image masks
            # if obj.ImageMask == PdfObject('true'):
            #     msg(f'inverting bitonal image mask after decoding')
            #     self.set_pil(ImageChops.invert(self.get_pil()))


        else:

            if jpeg:

                self.pil = Image.open(BytesIO(jpeg))

            if self.pil:

                # Handle transparency
                if self.pil.mode in ['PA','LA','RGBA']:
                    self.pil, alpha = PdfImage._pil_split_alpha(pil)
                    self.SMask = PdfImage(pil = alpha).encode()

                self.ColorSpace = PdfImage._pil_get_colorspace(self.pil)
                self.bpc = 1 if self.pil.mode == '1' else 8
                self.dpi = self.pil.info.get('dpi')

            if jp2:

                self.read_jp2_to_array(jp2)
                self.dpi = Image.open(BytesIO(jp2)).info.get('dpi')

            if array is not None:
                SImage.validate(array)
                N = SImage.nColors(array)
                self.ColorSpace = PdfName.DeviceGray if N == 1 \
                                    else PdfName.DeviceRGB if N == 3 \
                                    else PdfName.DeviceCMYK if N == 4 \
                                    else None
                self.bpc = 1 if SImage.isBitonal(array) else 8
 
    # -------------------------------------------------------------------------------------- Basic functions

    def w(self):
        '''Returns width'''
        return self.array.shape[1] if self.array is not None else self.pil.width

    def h(self):
        '''Returns height.'''
        return self.array.shape[0] if self.array is not None else self.pil.height

    def get_cpp(self):
        '''Returns components per pixel.'''
        return PdfColorSpace.get_cpp(self.ColorSpace)

    def get_mode(self):
        '''Returns a PIL mode that is appropriate for the current colorspace and bpc values.'''
        return PdfColorSpace.get_mode(self.ColorSpace, self.bpc)

    def to_bytes(self):
        '''Returns pixel intensities as a bytes object.'''
        return self.pil.tobytes() if self.pil else PdfFilter.pack_pixels(self.array, self.bpc)

    def invert(self):
        '''Inverts image'''
        if self.pil: self.set_pil(ImageChops.invert(self.pil))
        elif self.array is not None: self.set_array(SImage.invert(self.array))
        else: raise ValueError(f'bad PdfImage: {self}')

    def __str__(self):
        '''Returns a string representation of the image that contains
        various image parameters in a compact format'''
        attrs = [a for a in ['pil','array','jp2','mask'] if self.__getattr__(a) is not None]
        if self.isRendered: attrs.append('isRendered')
        attrs = '/'.join(attrs)
        return f'{self.w()}x{self.h()}, DPI={self.dpi}, CS={PdfColorSpace.get_name(self.ColorSpace)}' \
                 + f', BPC={self.bpc}, STATE: {attrs}'


    # -------------------------------------------------------------------------------------- render()

    def render(self, pdfPage:PdfDict = None, debug:bool = False):
        '''
        This is a wrapper for `PdfImage.apply_colorspace(self.ColorSpace, self.SMask)` which also checks for
        the page's default colorspaces, if they are defined, and uses them in place of
        the (uncalibrated) `/DeviceGray`,`/DeviceRGB` and `/DeviceCMYK`.
        '''
        if self.isRendered:
            warn('image already rendered; skipping')
            return

        if debug: msg('rendering started')

        # Default colorspace
        try:
            cs2cs = {'/DeviceGray':'/DefaultGray', '/DeviceRGB':'/DefaultRGB', '/DeviceCMYK':'/DefaultCMYK'}
            cs = pdfPage.Resources.ColorSpace[cs2cs[self.ColorSpace]] or self.ColorSpace
            if cs != None and debug:
                msg(f'Page default colorspace: {PdfColorSpace.toStr(self.ColorSpace)} --> {PdfColorSpace.toStr(cs)}')
        except:
            cs = self.ColorSpace

        if debug: msg(f'applying image colorspace: {PdfColorSpace.toStr(cs)}')

        modified = self.apply_colorspace(cs, self.SMask) # Mask is needed to unmultiply alpha if necessary

        # # Apply page default colorspace
        # try:
        #     # This will fail if cs is not among the entries of the page default colorspaces dict
        #     cs2cs = {'/DeviceGray':'/DefaultGray', '/DeviceRGB':'/DefaultRGB', '/DeviceCMYK':'/DefaultCMYK'}
        #     default_cs = pdfPage.Resources.ColorSpace[cs2cs[self.ColorSpace]]
        #     msg(f'applying page default colorspace: {PdfColorSpace.toStr(default_cs)}')
        #     self.apply_colorspace(default_cs)
        # except: pass

        self.isRendered = True
        if debug: msg('rendering ended')

        return modified

    # -------------------------------------------------------------------------------------- apply_colorspace()

    def apply_colorspace(self, cs:CS_TYPE, mask:PdfDict = None):
        '''
        Applies the given colorspace to self, using the following steps:

        1) decodes the image using the `self.Decode` or the default `/Decode` array
        (see help in `PdfDecodeArray.get_default()`);
        2) if `cs` is an `/Indexed` colorspace, remaps the image using the provided palette;
        3) if `cs` is a `/Separation`, `/DeviceN` or `/NChannel` colorspace, 
        or `mask` and `mask.Matte` are both not `None` (the mask is a pre-multiplied alpha)
        calls `PdfColorSpace.reduce(cs, mask)` on the resulting image.
        4) encodes the image using the appropriate default `/Decode` array for the obtained colorspace.

        Returns `True`/`False` depending on whether `self` has actually been modified.
        '''
        actualDecode  = PdfDecodeArray.get_actual(self.Decode, cs, self.bpc)
        defaultDecode = PdfDecodeArray.get_default(cs, self.bpc)

        cs_name = PdfColorSpace.get_name(cs)

        modified = False

        if cs_name == '/Indexed':

            # In the /Indexed color space, all color space transformations are performed on the palette.
            # Note: palette_cs is never /Indexed (see Adobe PDF Ref.)

            # PDF standard allows for remapping of palette indices via Decode arrays (why oh why?)
            if actualDecode != defaultDecode:
                warn(f'decoding indices in the /Indexed colorspace; this is strange, check results')
                array = self.get_array()
                array = PdfDecodeArray.decode(array, actualDecode, self.bpc)
                array = np.clip(np.round(array),0,255).astype('uint8') # clip right away: no encoding step later
                self.set_array(array)
                self.bpc = 8
                self.Decode = None
                
            # Transform the palette
            palette_cs, palette_array = PdfColorSpace.get_palette(cs)

            if PdfColorSpace.get_name(palette_cs) in ['/Separation', '/DeviceN', '/NChannel']:

                # Apply palette_cs to palette_array; both palette_array & palette_cs can change
                palette_decode = PdfDecodeArray.get_default(palette_cs, 8)
                palette_array = np.array([palette_array])
                palette_cs, palette_array = PdfColorSpace.reduce(palette_cs, palette_array, palette_decode, 8, mask)
                palette_array = palette_array[0]

                # Create new /Indexed colorspace
                self.ColorSpace = PdfArray([PdfName.Indexed, palette_cs, palette_array.shape[0], palette_array.tobytes()])

            # Reduce palette; comment this of if you do not want render() to reduce palettes
            self.ColorSpace = palette_cs
            arr = self.get_array().astype(np.uint8)
            self.set_array(palette_array[arr])
            self.bpc = 8
            self.Decode = None

            modified = True

        elif cs_name in ['/Separation', '/DeviceN', '/NChannel'] \
                or actualDecode != defaultDecode \
                or mask != None and mask.Matte != None:

            self.ColorSpace, array = PdfColorSpace.reduce(cs, self.get_array(), actualDecode, self.bpc, mask)
            self.set_array(array)
            if self.bpc != 1:
                self.bpc = 8
            self.Decode = None

            modified = True

        return modified

    # -------------------------------------------------------------------------------------- change_mode()

    def change_mode(self, mode:str, intent:str = '/Perceptual'):
        '''
        Changes image mode. The mode argument can be any one of: 'L', 'RGB', 'CMYK'.
        Returns True if self is modified as a results of the conversion,
        otherwise prints a warning and returns False.
        '''


        if mode not in ['L', 'RGB', 'CMYK']: raise ValueError(f'cannot convert to mode: {mode}')

        cs_name = PdfColorSpace.get_name(self.ColorSpace)

        if mode == self.get_mode() and cs_name in ['/DeviceGray', '/DeviceRGB', '/DeviceCMYK', '/ICCBased']:
            warn(f'old and new modes are the same: {mode}')
            return False

        msg(f'converting ({cs_name}, {self.get_mode()}) -> {mode}')

        if not self.isRendered:
            self.render()
            msg(f'rendered into colorspace: {PdfColorSpace.get_name(self.ColorSpace)}')

        if mode == self.get_mode():
            return True
        
        if self.get_mode() == '1':
            if mode == 'L':
                if self.array is not None: self.set_array(SImage.toGray(self.array))
                elif self.pil: self.set_pil(self.pil.convert('L'))
                else: raise ValueError(f'bad PdfImage: {self}')
                self.ColorSpace = '/DeviceGray'
                self.bpc = 8
                return True
            elif mode == 'RGB':
                if self.array is not None: self.set_array(SImage.toColor(self.array))
                elif self.pil: self.set_pil(self.pil.convert('RGB'))
                else: raise ValueError(f'bad PdfImage: {self}')
                self.ColorSpace = '/DeviceRGB'
                self.bpc = 8
                return True
            else:
                warn(f'cannot convert {"1"} --> {mode}')
                return False

        # Render
        pil = self.get_pil()

        # Get intent
        intents = {'/Perceptual': ImageCms.Intent.PERCEPTUAL,
                   '/RelativeColorimetric': ImageCms.Intent.RELATIVE_COLORIMETRIC,
                   '/Saturation': ImageCms.Intent.SATURATION,
                   '/AbsoluteColorimetric': ImageCms.Intent.ABSOLUTE_COLORIMETRIC }
        renderingIntent = intents.get(intent, ImageCms.Intent.PERCEPTUAL)

        msg(f'converting pil.mode = {pil.mode} -> {mode} (intent = {intent})')

        # Standard sRGB & CMYK profiles
        cmyk_profile = ImageCms.ImageCmsProfile(BytesIO(CMYK_DEFAULT_ICC_PROFILE))
        srgb_profile = ImageCms.createProfile('sRGB')

        # Input profile
        icc_profile = PdfColorSpace.get_icc_profile(self.ColorSpace)
        input_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile)) if icc_profile \
                            else cmyk_profile if pil.mode == 'CMYK' \
                            else srgb_profile if pil.mode in ['RGB', 'P'] else None
        if not input_profile:
            warn(f'cannot convert {self.get_mode()} --> {mode}')
            return False

        # Output profile
        output_profile = cmyk_profile if mode == 'CMYK' else srgb_profile if mode == 'RGB' else None

        if not output_profile:
            # Conversion to L: first convert to RGB to account for input_profile, then to L
            pil = ImageCms.profileToProfile(pil, input_profile, srgb_profile,
                                            renderingIntent = renderingIntent, outputMode='RGB')
            pil = pil.convert('L')
        else:
            pil = ImageCms.profileToProfile(pil, input_profile, output_profile,
                                            renderingIntent = renderingIntent, outputMode=mode)
        
        if 'icc_profile' in pil.info: del pil.info['icc_profile']

        self.ColorSpace = PdfName.DeviceGray if pil.mode == 'L' \
                            else PdfName.DeviceRGB if pil.mode == 'RGB' \
                            else PdfName.DeviceCMYK
                            # else PdfColorSpace.make_icc_based_colorspace(CMYK_DEFAULT_ICC_PROFILE, N=4)

        self.set_pil(pil)
        self.bpc = 8
        return True

    # -------------------------------------------------------------------------------------- get_pil()

    def get_pil(self):
        '''
        Creates a PIL Image representation in self.pil, if not yet created, and returns it.
        '''
        if not self.pil:

            assert self.array is not None
            assert self.ColorSpace is not None
            assert self.bpc is not None

            msg('array -> pil')
            mode = PdfColorSpace.get_mode(self.ColorSpace, self.bpc)
            # A PIL bug: https://stackoverflow.com/questions/50134468/convert-boolean-numpy-array-to-pillow-image
            self.pil = Image.fromarray(self.array, mode = mode) if mode != '1' \
                else Image.frombytes(mode='1', size=self.array.shape[::-1], data=np.packbits(self.array, axis=1))

        return self.pil

    # -------------------------------------------------------------------------------------- get_array()

    def get_array(self):
        '''
        Creates a numpy array representation in self.array, if not yet created, and returns it.
        '''
        if self.array is None:
            assert self.pil
            msg('pil -> array')
            self.array = np.array(self.pil)
        SImage.validate(self.array)
        return self.array

    # -------------------------------------------------------------------------------------- set_pil()

    def set_pil(self, pil:Image.Image):
        '''
        Sets self.pil to the given PIL image, and self.array & self.jp2 to None
        '''
        self.pil = pil
        self.array = None
        self.jp2 = None
        self.jpeg = None

    # -------------------------------------------------------------------------------------- set_array()

    def set_array(self, array:np.ndarray):
        '''
        Sets self.array to the given numpy image array, and self.pil, self.jpeg & self.jp2 to None
        '''
        self.pil = None
        self.array = array
        self.jp2 = None
        self.jpeg = None

    # -------------------------------------------------------------------------------------- set_jp2()

    def read_jp2_to_array(self, jp2:bytes):
        '''
        Reads jp2 into self.array and sets self.bpc.
        If self.ColorSpace is None it is set to the internal colorspace of the jp2 image.
        Also, self.jp2 is set to jp2, unless self.ColorSpace == ['/Indexed', ..].

        Having self.array, self.ColorSpace and self.bpc set allows for subsequent image analysis/modification.
        This leaves self.jp2 as a kind of storage for the original JPEG2000 file's bytes
        in case we want to save the jp2 image data without re-encoding.

        Note: when self.ColorSpace is an /Indexed colorspace, self.array will be read as
        a /DeviceRGB or /DeviceCMYK colorspace, and therefore, and therefore 1) self.ColorSpace
        will be set to /DeviceRGB or /DeviceCMYK to reflect that; 2) jp2 will not be stored
        in self.jp2.
        '''
        self.array, jp2_cs, self.bpc = PdfImage._jp2_read(jp2)
        cs = self.ColorSpace
        isIndexed = isinstance(cs, PdfArray) and cs[0] == '/Indexed'
        if not isIndexed:
            self.jp2 = jp2
        if not cs or isIndexed: # obj.ColorSpace overrides internal JPX colorspace
            self.ColorSpace = jp2_cs
            self.Decode = None
        self.pil = None
        self.jpeg = None

    # -------------------------------------------------------------------------------------- resize()

    def resize(self, width:int, height:int):
        '''
        Resize the image to the given dimensions.
        '''
        self.set_pil(self.get_pil().resize((width, height)))

    # -------------------------------------------------------------------------------------- saveAs()

    def saveAs(self, Format:str,
                    Q:float = None,
                    CR:float = None,
                    PSNR:int = None,
                    optimize = True,
                    render = False,
                    intent:str = '/Perceptual',
                    embedProfile:bool = True,
                    invertCMYK = False):
        '''
        Returns a tuple `(imageStream[bytes], fileExtension[str])`.
        The encoding is done in the given `Format` using the
        `Q` (quality), `CR` (compression ratio) and `optimize` parameters.

        The image has to be rendered first in order to produce a faithful representation.
        '''
        ext = {'TIFF':'.tif', 'PNG':'.png', 'JPEG':'.jpg', 'JPEG2000':'.jp2', 'JBIG2':'jb2'}

        addAlpha = render and (self.Mask is not None or self.SMask is not None)

        if Format == 'JPEG' and addAlpha:
            raise ValueError(f'cannot save an image transparency as JPEG')

        if Format == 'auto':
            Format = self.pil.format if self.pil and self.pil.format in ext \
                        else 'JPEG2000' if self.jp2 \
                        else 'JPEG' if self.jpeg \
                        else 'TIFF'
            if addAlpha and Format in ['JPEG','JPEG2000']:
                Format = 'TIFF'

        if Format not in ext: raise ValueError(f'cannot save in format: {Format}')


        if addAlpha:
            msg('rendering alpha')
            alpha = PdfImage(obj = self.Mask or self.SMask)
            alpha.render()
            if alpha.w() != self.w() or alpha.h() != self.h():
                if self.w() * self.h() > alpha.w() * alpha.h():
                    msg(f'resizing alpha to fit image')
                    alpha.resize(self.w(), self.h())
                else:
                    msg(f'resizing image to fit alpha')
                    self.resize(alpha.w(), alpha.h())

            # # ???????????????????????????????
            # if alpha.mode == '1':
            #     alpha = alpha.convert('L')

        if Format == 'JPEG2000':

            if self.jp2:
                stream  = self.jp2
            else:
                assert Q or CR
                if Q: assert 0 < Q <= 100
                if CR: assert CR > 1
                if addAlpha:
                    raise ValueError(f'JPEG2000 encoding with alpha channel is not implemented')
                stream = PdfImage._jp2_write(array = self.get_array(),
                                            cs = self.ColorSpace,
                                            Q = Q,
                                            CR = CR,
                                            PSNR = PSNR)
        else:
 
            modified = False
            if render:

                modified = self.render()
                if modified:
                    msg(f'rendered image')

                if addAlpha and self.get_mode() not in ['L', 'RGB']:
                    newMode = 'RGB' if self.get_mode() != '1' else 'L'
                    self.change_mode(newMode, intent)
                    msg(f'changed mode {self.get_mode()} -> {newMode} to allow adding alpha')
                    modified = True

            mustInvertCMYK = invertCMYK and self.get_mode() == 'CMYK' and Format == 'JPEG'
 
            if Format == 'JPEG' and self.jpeg \
                    and not modified and not addAlpha and not mustInvertCMYK \
                    and Q is None:
                msg(f'saving original (unmodified) JPEG file')
                return self.jpeg, '.jpg'

            pil = self.get_pil()
            PdfImage._pil_set_colorspace(pil, self.ColorSpace)

            if addAlpha:

                msg('adding alpha')

                # alpha.render()
                assert alpha.get_cpp() == 1
                print('DEBUG:', alpha.get_mode())
                alpha.change_mode('RGB' if alpha.get_mode() != '1' else 'L', intent)
                print('DEBUG:', alpha.get_mode())
                alpha_pil = alpha.get_pil()

                # invert bitonal masks
                if self.Mask and self.Mask.ImageMask == PdfObject('true'):
                    alpha_pil = ImageChops.invert(alpha_pil)

                PdfImage._pil_set_colorspace(alpha_pil, alpha.ColorSpace)
                pil.putalpha(alpha_pil)

            # Shouldn't this be inverted before adding alpha?
            if mustInvertCMYK:
                msg('inverting CMYK JPEG before encoding')
                pil = ImageChops.invert(pil)

            bs = BytesIO()
            Q = Q if Q else 'keep' if pil.format == 'JPEG' and Format == 'JPEG' else 100
            # tiff_compression = 'group4' if pil.mode == '1' else 'tiff_lzw' if Format == 'TIFF' else None
            tiff_compression = 'group4' if pil.mode == '1' else 'tiff_adobe_deflate' if Format == 'TIFF' else None
            icc_profile = pil.info.get('icc_profile') if pil.info and embedProfile else None

            dpi = self.dpi if self.dpi not in (None, (1,1)) else (72,72)

            stream = None

            if Format == 'JPEG':
                msg(f'saving with Format = {Format}, Q = {Q}')
                pil.save(bs, Format, quality=Q, optimize = optimize, icc_profile = icc_profile,
                            info = pil.info, dpi=dpi)
            elif Format == 'PNG':
                pil.save(bs, Format, optimize = optimize, icc_profile = icc_profile)
            elif Format == 'TIFF':
                pil.save(bs, Format, compression = tiff_compression, icc_profile = icc_profile)
            else:
                raise ValueError(f'format not supported in saveAs(): {Format}')
 
            if stream is None:
                stream = bs.getvalue()

            # A bug found in some JPEG files
            if Format == 'JPEG':
                s = stream.rstrip(b'\n')
                if s[-2:] != b'\xff\xd9':
                    # stream += b'\xff\xd9'
                    raise ValueError(f'bad JPEG stream tail: {s[-10:]}')

        return stream, ext[Format]


    # -------------------------------------------------------------------------------------- to_pdf_page()

    def to_pdf_page(self, isMask:bool = False):
        '''
        Convert the image to a PDF page.
        '''
        # Encode image
        xobj = self.encode(isMask = isMask)

        print('RESULT:')
        pprint(xobj)

        dpi = self.dpi if self.dpi not in [None, (1,1)] else (72,72)

        # Create page
        w,h = self.w(), self.h()
        w,h = w * 72/dpi[0], h*72/dpi[1]
        p = lambda x: round(x*1000000)/1000000
        q = lambda x: f'{p(x):f}'.rstrip('0').rstrip('.')
        page = IndirectPdfDict(
            Type = PdfName.Page,
            MediaBox = [0, 0, p(w), p(h)],
            Contents = IndirectPdfDict(stream=f'{q(w)} 0 0 {q(h)} 0 0 cm\n/Im1 Do\n'),
            Resources = PdfDict(XObject = PdfDict(Im1 = xobj))
        )

        return page

    # -------------------------------------------------------------------------------------- compress()

    @staticmethod
    def recompress(image_obj:IndirectPdfDict,
                   Format:str = None,
                   Q:float = None,
                   PSNR:int = None,
                   CR:float = None,
                   minsize:int = None):
        '''
        (Re-)compress image stream. The format can be any one of:
        
        * 'ZIP': compress with /FlateDecode without PNG predictors
        * 'PNG': compress with /FlateDecode with PNG predictors
        * 'RLE': compress with /RunLengthDecode
        * 'JPEG': compress with /DCTDecode; compressionRatio should be specified
        * 'JPEG2000': compress with /JPXDecode; either compressedQuality or compressionRatio should be specified

        The obj.stream is replaced only if the (re-)compression decreases stream size. An exception to this
        is the 'ZIP' compression format, for which the stream is always (re-)compressed, even if
        its size increases as a result. Therefore use 'ZIP' to (re-)compress only if you need
        the /FlateDecode filter for compatibility/tests. For stream size reduction with
        /FlateDecode try 'PNG' compression instead.

        Returns True/False depending on whether image stream has been replaced.
        '''

        def print_size_change(size_old, size_new):
            return f'Size: {size_old} --> {size_new} ({((size_new - size_old)*100)//size_old:+d}%)'

        size_old = int(image_obj.Length)

        bigEnough = size_old > (minsize or 0)
        if not bigEnough:
            msg(f'Image size {size_old} < minsize {minsize}; skipping')
            return False

        obj = PdfFilter.uncompress(image_obj)
        image = PdfImage(obj = obj)

        if PdfColorSpace.get_name(image.ColorSpace) == '/Indexed':
            msg('rendering /Indexed colorspace before re-compressing')
            image.render()
        
        bpc = image.bpc
        cs = image.ColorSpace

        DecodeParms = None

        fmt = Format

        if fmt == 'PNG' and PdfColorSpace.get_cpp(PdfImage.xobject_get_cs(image_obj)) not in [1,3]:
            msg(f'PNG compression only works for 1- or 3-component images; falling back to ZIP')
            fmt = 'ZIP'
 
        if fmt == 'ZIP':

            stream = zlib.compress(image.to_bytes())
            Filter = PdfName('FlateDecode')

        elif fmt == 'RLE':

            stream = PdfFilter.rle_encode(image.to_bytes())
            Filter = PdfName('RunLengthDecode')

        elif fmt == 'PNG':

            cpp = image.get_cpp()

            if cpp not in [1, 3]:
                msg(f'cpp = {cpp} not supported by PNG; skipping')
                return False

            Filter = PdfName('FlateDecode')

            png, _ = image.saveAs('PNG')

            stream, Predictor, columns, bpc_png, cpp_png, interlace = PdfImage._png_get_stream_and_params(png)

            assert bpc_png == 8
            assert cpp_png == cpp
            assert interlace == 0
            assert columns == image.w()
            
            msg(f'compressing with PNG predictor: {Predictor}')
            DecodeParms = PdfDict(Colors = cpp,
                                    Columns = columns,
                                    Predictor = Predictor)
            bpc = 8

        elif fmt == 'JPEG':

            cpp = image.get_cpp()
            if cpp not in [1,3,4]:
                msg(f'cpp = {cpp} not supported by JPEG; skipping')
                return False


            stream, _ = image.saveAs(Format = 'JPEG', Q = Q, invertCMYK = True)
            Filter = PdfName('DCTDecode')

        elif fmt == 'JPEG2000':

            if image.get_mode() not in ['L', 'RGB']:
                msg(f'mode {image.get_mode()} not supported by JPEG2000; skipping')
                return False

            assert Q or CR
            if Q: assert 0 < Q <= 100
            if CR: assert CR > 1

            stream, _ = image.saveAs('JPEG2000', Q=Q, CR=CR, PSNR=PSNR)
            Filter = PdfName('JPXDecode')

        else:
            raise ValueError(f'bad Format: {Format}')

        # Compare sizes before/after
        size_new = len(stream)
        if Format == 'ZIP' or Format == 'RLE' \
                or Format == 'PNG' and size_new < size_old \
                or size_new * 10 < size_old * 9:
            msg(f're-compressing {image_obj.Filter} --> {Filter}: {print_size_change(size_old, size_new)}')
            obj.stream = stream.decode('Latin-1')
            obj.Filter = Filter
            obj.DecodeParms = DecodeParms
            obj.BitsPerComponent = bpc
            obj.ColorSpace = cs

            PdfImage.xobject_copy(obj, image_obj)
            image_obj.Decode = image.Decode

            return True
        else:
            msg(f'skipping {image_obj.Filter} --> {Filter}: {print_size_change(size_old, size_new)}')
            return False

    # -------------------------------------------------------------------------------------- encode()

    def encode(self, isMask:bool = False):
        '''
        Encode a PdfImage into an image xobject. The isMask argument sets the resulting
        obj.ImageMask for bitonal (self.bpc == 1) images; setting isMask == True while
        encoding an non-bitonal image throws an exception.
        '''
        if isMask and self.bpc != 1:
            raise ValueError(f'cannot set ImageMask = true with self.bpc = {self.bpc}')

        DecodeParms = None
        Decode = None
        bpc = self.bpc

        if bpc == 1:

            Filter = 'CCITTFaxDecode'
            pil = self.get_pil()

            # if isMask:
            #     msg('inverting bitonal image mask before encoding')
            #     pil = ImageChops.invert(pil)

            stream = PdfImage._tiff_get_strip(PdfImage._tiff_write(pil))
            DecodeParms = PdfDict(K = -1,
                                  Columns = self.w(),
                                  Rows = self.h(),
                                  BlackIs1 = PdfObject('true'))
            
        elif self.jp2:

            Filter = 'JPXDecode'
            stream = py23_diffs.convert_load(self.jp2)

        elif not self.pil or self.pil.format in ['RAW', 'GIF', 'PNG', 'BMP', 'TIFF', 'JPEG2000', None]:

            Filter = 'FlateDecode'
            stream = zlib.compress(self.to_bytes())
 
        elif self.pil.format == 'JPEG':

            Filter = 'DCTDecode'
            stream, _ = self.saveAs('JPEG', embedProfile = False, invertCMYK = True)

        else:
            raise ValueError(f'unsupported format: {self.pil.format}')

        return IndirectPdfDict(
            Type = PdfName.XObject,
            Subtype = PdfName.Image,
            Width = self.w(),
            Height = self.h(),
            BitsPerComponent = bpc,
            ColorSpace = self.ColorSpace if not isMask else None,
            ImageMask = PdfObject('true') if isMask else None,
            Filter = PdfName(Filter),
            DecodeParms = DecodeParms,
            Decode = Decode or self.Decode if not self.isRendered else None,
            Mask = self.Mask,
            SMask = self.SMask,
            stream = py23_diffs.convert_load(stream)
        )


    # -------------------------------------------------------------------------------------- jbig2_encode()

    # @staticmethod
    # def jbig2_encode():
    #     '''
    #     '''

    #     # stream = self.jbig2[0]
    #     # Filter = 'JBIG2Decode'
    #     # if self.jbig2[1]:
    #     #     globals = IndirectPdfDict(Filter = PdfName.FlateDecode, stream = zlib.compress(self.jbig2[1]))
    #     # DecodeParms = PdfDict(JBIG2Globals = globals)

    # -------------------------------------------------------------------------------------- xobject_from_inline_image()

    @staticmethod
    def xobject_from_inline_image(inline_img:list):
        '''Converts a PDF inline image (represented as an element of a parsed PDF stream tree: ['BI', [{header_dict}, "data_str"]])
        to a PDF image object. This helps treat both inline and object images in PDF in a uniform way.
        '''
        if inline_img[0] != 'BI': return None

        header, data = inline_img[1]

        data = data.strip(' \n\r')
        if data[:2] != 'ID' or data[-2:] != 'EI': return None
        data = data[3:-2]
        data = data.rstrip('\r\n')
        # data = data.rstrip('\n') if data[-1] == '\n' else data.rstrip('\r')

        filter_name = get_any_key(header, '/F','/Filter')
        filter_name_aliases = {'/AHx':'/ASCIIHexDecode', '/A85':'/ASCII85Decode', '/LZW':'/LZWDecode', '/Fl':'/FlateDecode',
                                '/RL':'/RunLengthDecode', '/CCF':'/CCITTFaxDecode', '/DCT':'/DCTDecode'}
        if filter_name in filter_name_aliases: filter_name = filter_name_aliases[filter_name]

        obj = IndirectPdfDict(
            Type = '/XObject',
            Subtype = '/Image',
            Width = get_any_key(header, '/W','/Width'),
            Height = get_any_key(header, '/H','/Height'),
            Filter = filter_name,
            DecodeParms = get_any_key(header, '/DP', '/DecodeParms'),
            BitsPerComponent = get_any_key(header, '/BPC','/BitsPerComponent'),
            ColorSpace = get_any_key(header, '/CS','/ColorSpace'),
            Decode = get_any_key(header, '/D','/Decode'),
            ImageMask = PdfObject(get_any_key(header, '/IM','/ImageMask')),
            Intent = get_any_key(header, '/Intent'),
            Interpolate = get_any_key(header, '/I','/Interpolate')
        )
        obj.stream = data

        return obj

    # -------------------------------------------------------------------------------------- xobject_to_inline_image_stream()

    @staticmethod
    def xobject_to_inline_image_stream(obj:PdfObject):
        '''
        Converts a PDF image xobject to a PDF inline image stream.
        '''
        p = lambda z: '<<'+' '.join(f'{x} {y}' for (x,y) in z.items())+'>>' if isinstance(z,PdfDict) \
            else '['+' '.join(f'{x}'for x in z)+']' if isinstance(z,PdfArray) \
            else f'{z}'
         
        if obj == None or obj.Subtype != PdfName.Image: return None

        s = f'BI\n/W {obj.Width} /H {obj.Height}'
        if obj.Filter != None: s += f' /F {obj.Filter}'
        if obj.DecodeParms != None: s += f' /DP ' + p(obj.DecodeParms)
        if obj.BitsPerComponent != None: s += f' /BPC {obj.BitsPerComponent}'
        if obj.ColorSpace != None: s += f' /CS {obj.ColorSpace}'
        if obj.Decode != None: s += f' /D {obj.Decode}'
        if obj.ImageMask != None: s += f' /IM {obj.ImageMask}'
        if obj.Intent != None: s += f' /Intent {obj.Intent}'
        if obj.Interpolate != None: s += f' /I {obj.Interpolate}'
        s += '\nID\n'
        s += obj.stream
        s += '\nEI\n'

        return s

    # -------------------------------------------------------------------------------------- xobject_str()

    @staticmethod
    def xobject_str(obj:PdfDict):
        '''
        A compact string representation of an image XObject.
        '''
        filters_str = '+'.join(PdfFilter.filters_as_list(obj))
        w, h = int(obj.Width), int(obj.Height)
        bpc = PdfImage.xobject_get_bpc(obj)
        cs_name = PdfColorSpace.get_mode(PdfImage.xobject_get_cs(obj), bpc)
        size = formatSize(int(obj.Length))

        s = f'{cs_name} ({w},{h})'
        if bpc not in [8, None]:
            s += f' ({bpc}bit)'
        s += f' {filters_str} {size}'
        if obj.Mask:
            s += f' -> Mask: {PdfImage.xobject_str(obj.Mask)}'
        if obj.SMask:
            s += f' -> SMask: {PdfImage.xobject_str(obj.SMask)}'
        return s

    # -------------------------------------------------------------------------------------- xobject_get_bpc()

    @staticmethod
    def xobject_get_bpc(obj:PdfDict):
        '''
        Returns bits per pixel
        '''
        if obj.Subtype != PdfName.Image: raise ValueError(f"Not an image: {obj}")
        return 1 if obj.ImageMask == PdfObject('true') \
            else None if obj.Filter == PdfName.JPXDecode \
            else int(obj.BitsPerComponent)

    # -------------------------------------------------------------------------------------- xobject_get_colorspace()

    @staticmethod
    def xobject_get_cs(obj:PdfDict) -> CS_TYPE:
        '''
        Returns obj.ColorSpace if it's set, or /DeviceGray if obj.ImageMask == true,
        or None otherwise.
        '''
        if obj.Subtype != PdfName.Image: raise ValueError(f"Not an image: {obj}")
        return obj.ColorSpace if obj.ColorSpace != None \
            else PdfName.DeviceGray if obj.ImageMask == PdfObject('true') \
            else None

    # -------------------------------------------------------------------------------------- xobject_copy()

    @staticmethod
    def xobject_copy(obj_from:IndirectPdfDict, obj_to:IndirectPdfDict):
        '''
        '''
        obj_to.clear()
        for k,v in obj_from.items():
            obj_to[k] = v
        obj_to.stream = obj_from.stream


    # -------------------------------------------------------------------------------------- xobject_copy()

    @staticmethod
    def xobject_make_transparency_group_graphics_state(obj:IndirectPdfDict):
        '''
        Make an extended graphics state with the image obj used as a source of the transparency
        alpha. Use it like this:
        ```
        GS1 = PdfImage.xobject_make_transparency_group_graphics_state(obj)
        xobject.Resources.ExtGState = PdfDict(GS1 = GS1)
        xobject.stream += 'q /GS1 gs [graphics operators] Q'
        ```
        '''
        return  PdfDict(
                    Type = PdfName.ExtGState,
                    BM = PdfName.Multiply,
                    CA = 1, ca = 1,
                    SMask = PdfDict(
                        BC = [0,0,0], # Backdrop color (black)
                        S = PdfName.Luminosity, # source of mask's alpha
                        # S = PdfName.Alpha, # source of mask's alpha

                        # Transparency group XObject to be used as mask
                        G = IndirectPdfDict(
                            Type = PdfName.XObject,
                            Subtype = PdfName.Form,
                            BBox = [0,0,1,1],
                            Group = IndirectPdfDict(
                                CS = PdfName.DeviceRGB,
                                I = PdfObject('true'), # Isolated, i.e. mask's background is transparent
                                K = PdfObject('false'), # Knockout is false, see PDF Ref
                                S = PdfName.Transparency
                            ),
                            Resources = PdfDict(XObject = PdfDict(mask = obj)),
                            stream = '/mask Do\n'
                        )
                    )
                )

    # -------------------------------------------------------------------------------------- _jbig2_read()

    @staticmethod
    def _jbig2_read(jbig2:tuple[bytes, bytes]):
        '''
        Returns a PIL image
        '''
        loc, glob = jbig2
        with tempfile.TemporaryDirectory() as tmp:
            T = lambda fileName: os.path.join(tmp, fileName)
 
            open(T('locals'),'wb').write(loc) 
            if glob:
                open(T('globals'),'wb').write(glob)
                OS.execute(['jbig2dec', '-e', '-o', T('out.png'), T('globals'), T('locals')])
            else:
                OS.execute(['jbig2dec', '-e', '-o', T('out.png'), T('locals')])
            
            return Image.open(T('out.png'))

        # If you have to re-assemble the JBIG2 file for some reason, just add the header:
        # 1-page, sequential order; see JBIG2 Specs
        # header = b'\x97\x4A\x42\x32\x0D\x0A\x1A\x0A\x01\x00\x00\x00\x01'

    # -------------------------------------------------------------------------------------- _jp2_read()

    @staticmethod
    def _jp2_read(jp2:bytes):
        '''
        Reads a JPEG2000 file (a bytes object) and returns a tuple (numpyArray, colorSpace, bitsPerComponent).
        '''
        with tempfile.TemporaryDirectory() as tmp:
            T = lambda fileName: os.path.join(tmp, fileName)

            open(T('temp.jp2'),'wb').write(jp2)
            jp2k = Jp2k(T('temp.jp2'))          # call print(jp2k) to inspect the boxes

            # Debug
            # print(jp2k)

            # Get the data
            try: bpc = int(jp2k.box[2].box[0].bits_per_component)
            except:
                try: bpc = int(jp2k.box[3].box[0].bits_per_component)
                except:
                    bd = jp2k.codestream.segment[1].bitdepth
                    bpc = bd[0]
                    if not all(d == bpc for d in bd):
                        raise ValueError(f'JPEG images with channel-specific bpc are not supported: {bd}')

            data = jp2k[:] # This parses all JPEG2000 image slices for a full resolution image
            if bpc == 1: data = data.astype(bool)
            elif bpc <= 8: data = data.astype('uint8')
            else: raise ValueError(f'unsupported JPEG2000 bit depth: {bpc}')
            data = SImage.normalize(data)
            SImage.validate(data)

            # Get the colorspace
            try: EnumCS = jp2k.box[2].box[1].colorspace
            except:
                try: EnumCS = jp2k.box[3].box[1].colorspace
                except: EnumCS = None

            if EnumCS != None:
                cs_name = {12:'CMYK', 16:'RGB', 17:'Gray', 18:'YCC', 20:'e-sRGB', 21:'ROMM-RGB'}.get(EnumCS)
                if cs_name in [None, 'YCC', 'e-sRGB', 'ROMM-RGB']:
                    raise ValueError(f'unsupported JPX colorspace: {cs_name}')
                cs = PdfName('Device' + cs_name)
            else:
                nColors = SImage.nColors(data)
                try: icc_profile = jp2k.box[3].box[1].icc_profile
                except: icc_profile = None
                if icc_profile is not None:
                    cs = PdfColorSpace.make_icc_based_colorspace(icc_profile, N = nColors)
                else:
                    nColors2cs = {3:'RGB', 4:'CMYK'}
                    cs = nColors2cs.get(nColors)

        return data, cs, bpc

    # -------------------------------------------------------------------------------------- _jp2_write()

    @staticmethod
    def _jp2_write(array:np.ndarray,
                   cs:CS_TYPE,
                   Q:int = None,
                   CR:float = None,
                   PSNR:int = None):
        '''
        Encode array as a JPEG2000 image.
        '''
        # make sure colorspace is either grayscale or RGB
        cpp = PdfColorSpace.get_cpp(cs)
        if cpp not in [1,3]:
            raise ValueError(f'Jp2k: cpp = {cpp} not supported (must be 1 or 3)')

        # limit the number of resolutions for small images
        numres = 6 # the default for large images
        while 1 << numres > min(array.shape[:2]):
            numres -= 1
        numres = max(numres, 1)

        # encode
        with tempfile.TemporaryDirectory() as tmp:
            T = lambda fileName: os.path.join(tmp, fileName)
            # can also try: cratios=[CR*4, CR*2, CR]
            if PSNR:
                print(f'encoding JPEG2000 with PSNR = [{PSNR}]')
                Jp2k(T('encoded.jp2'), array, colorspace='RGB' if cpp == 3 else 'Gray', psnr=[PSNR], numres=numres)
            else:
                # determine compression ratio
                assert CR or Q
                CR = int(round(CR)) if CR != None else int(round(2 ** ((100 - Q)/10.0)))
                print(f'encoding JPEG2000 with cratios = [{CR}]')
                Jp2k(T('encoded.jp2'), array, colorspace='RGB' if cpp == 3 else 'Gray', cratios=[CR], numres=numres)
            
            return open(T('encoded.jp2'), 'rb').read()

    # -------------------------------------------------------------------------------------- _tiff_make_header()

    @staticmethod
    def _tiff_make_header(obj:PdfDict):
        '''
        Make a TIFF header.

        The CCITTFaxDecode -encoded stream is essentially a TIFF file without the header, so
        this function reconstructs the missing TIFF file header.
        '''
        assert obj.Subtype in ['/Image', PdfName.Image] and obj.Filter == '/CCITTFaxDecode'
 
        # ------------------------------------------------------------------ Get info

        parm = obj.DecodeParms
        width, height = int(obj.Width), int(obj.Height)
        length = int(obj.Length)

        K = get_key(parm, '/K', '0')
        tiff_compression = 3 if int(K) >= 0 else 4

        BlackIs1 = get_key(parm, '/BlackIs1', 'false')
        # tiff_photometric = 0 if BlackIs1 == 'true' else 1
        tiff_photometric = 1 if BlackIs1 == 'true' else 0           # ??? Why is this backwards ???
        # if obj.ImageMask == PdfObject('true'): tiff_photometric = 1 - tiff_photometric

        bpc = 1
        cpp = 1

        predictor = 1

        encodedByteAlign = get_key(parm, '/EncodedByteAlign') == 'true'
        if encodedByteAlign:
            if tiff_compression != 3:
                raise ValueError(f'/CCITTFaxDecode Group4 (T6) compression with /EncodedByteAlign == true not supported')
            warn(f'*** /EncodedByteAlign == true, this is in beta-testing, check results ***')

        nTags = 11
        tiff_header_struct = '<' + '2s' + 'H' + 'L' + 'H' + 'HHLL' * nTags + 'L'
        tiff_header = struct.pack(tiff_header_struct,

                        # --- Image header starts here
                        b'II',  # Byte order indication: Little indian
                        42,  # Version number (always 42)
                        8,  # Offset to first IFD

                        # --- IFD starts here
                        nTags,  # the number of tags in IFD
                        256, 4, 1, width,  # ImageWidth, LONG, 1, width
                        257, 4, 1, height,  # ImageLength, LONG, 1, length
                        258, 3, 1, bpc,  # BitsPerSample, SHORT, 1, 1; this is the default, so omit?
                        259, 3, 1, tiff_compression,  # Compression, SHORT, 1, 3 = Group 3, 4 = Group 4, 8 - Deflate (ZIP)
                        262, 3, 1, tiff_photometric,  # Photometric interpretation, SHORT, 1, 0 = WhiteIsZero
                        273, 4, 1, struct.calcsize(tiff_header_struct),  # StripOffsets, LONG, 1, len of header
                        277, 3, 1, cpp,  # SamplesPerPixel, LONG, 1, component per pixel (cpp)
                        278, 4, 1, height,  # RowsPerStrip, LONG, 1, height
                        279, 4, 1, length,  # StripByteCounts, LONG, 1, size of image
                        292, 4, 1, 4 if encodedByteAlign else 0,  # TIFFTAG_GROUP3OPTIONS, LONG, 1, 0..7 (flags)
                        317, 3, 1, predictor,  # Predictor, SHORT, 1, predictor = 1 (no prediction) or 2 (=left)
                        0  # --- IFD ends here; offset of the next IFD, 4 0-bytes b/c no next IFD
                        )
        return tiff_header

    # -------------------------------------------------------------------------------------- _tiff_write()

    @staticmethod
    def _tiff_write(pil:Image.Image):

        # Make sure Pillow produces single-strip TIFF; this works with Pillow versions < 8.3.0 or >=8.4.0  
        # See: https://github.com/python-pillow/Pillow/pull/5744
        # For Pillow versions from 8.3.0 to 8.3.9 see:
        # https://gitlab.mister-muffin.de/josch/img2pdf/commit/6eec05c11c7e1cb2f2ea21aa502ebd5f88c5828b
        # https://gitlab.mister-muffin.de/josch/img2pdf/issues/46
        TiffImagePlugin.STRIP_SIZE = 2 ** 31

        bs = BytesIO()
        pil.save(bs, format="TIFF", compression='group4' if pil.mode == '1' else 'tiff_lzw')
        # pil.save(bs, format="TIFF", compression='group4' if pil.mode == '1' else 'tiff_adobe_deflate')

        return bs.getvalue()
    
    # -------------------------------------------------------------------------------------- _tiff_get_strip()

    @staticmethod
    def _tiff_get_strip(tiff:bytes):
        '''
        Returns the body of a single-strip TIFF file
        '''

        bs = BytesIO(tiff)
        img = Image.open(bs)

        # Read the TIFF tags to find the offset(s) and length of the compressed data strips.
        strip_offsets = img.tag_v2[TiffImagePlugin.STRIPOFFSETS]
        strip_bytes = img.tag_v2[TiffImagePlugin.STRIPBYTECOUNTS]
        rows_per_strip = img.tag_v2.get(TiffImagePlugin.ROWSPERSTRIP, 2 ** 32 - 1)
        if len(strip_offsets) != 1 or len(strip_bytes) != 1:
            raise ValueError("Expected a single strip")
        (offset,), (length,) = strip_offsets, strip_bytes

        bs.seek(offset)
        return bs.read(length)

    # -------------------------------------------------------------------------------------- _png_get_stream_and_params()

    @staticmethod
    def _png_get_stream_and_params(png:bytes):
        '''
        Returns a tuple `(IDAT_chunk, predictor, width, bpc, cpp, interlace)`, where
        * `predictor` = 10..15 is the PNG predictor
        * `width` is the width of the image
        * `bpc` is BitsPerComponent
        * `cpp` is ComponentsPerPixel
        * `interlace` is 1 if PNG is interlaced and 0 otherwise
        '''

        png = BytesIO(png)
        idatChunks = []
        png.seek(0)
        png.read(8)

        width, bpc, cpp, interlace = None, None, None, None

        while True:

            length = struct.unpack(b'!L', png.read(4))[0]
            type = png.read(4)
            data = png.read(length)
            crc  = struct.unpack(b'!L', png.read(4))[0]

            if crc != zlib.crc32(type + data):
                warn(f'crc error in chunk: {type}')
                return None, None, None, None, None

            if type == b'IDAT':
                idatChunks.append(data)

            if type == b'IHDR':
                
                assert length == 13

                width, height, bpc, color_type, _, _, interlace = struct.unpack(b'!LLBBBBB', data)

                cpp = {0:1, 2:3, 3:1, 4:2, 6:4}.get(color_type) # color types: L, RGB, P, LA, RGBA

                assert cpp is not None

                bpp = (bpc * cpp + 7) // 8  # bytes per pixel (rounded up)

            if type == b'IEND': break

        if len(idatChunks) == 0:
            warn(f'no IDAT chunks in PNG')
            return None, None, None, None, None
        
        stream = b''.join(chunk for chunk in idatChunks)
        decompressed = zlib.decompress(stream)

        # Each row: 1 filter byte + row data
        stride = 1 + (width * bpp)

        filters = set()

        for y in range(height):
            row_start = y * stride
            filters.add(decompressed[row_start])

        N = len(filters)
        assert N > 0
        predictor = 15 if N > 1 else filters.pop() + 10

        return stream, predictor, width, bpc, cpp, interlace

    # -------------------------------------------------------------------------------------- _pil_get_colorspace()

    @staticmethod
    def _pil_get_colorspace(image:Image.Image):
        '''
        '''
        mode2cpp = {'L':1, 'RGB':3, 'CMYK':4}
        mode2cs = {'1':PdfName.DeviceGray, 'L':PdfName.DeviceGray, 'RGB':PdfName.DeviceRGB, 'CMYK':PdfName.DeviceCMYK}

        assert image.mode != 'PA' # reduce 'PA' modes before calling this function
        mode = image.mode if image.mode[-1] != 'A' else image.mode[:-1]

        cs = None

        if mode == 'P':
            # palette = image.getpalette()
            # if palette == None: return None
            # palette = b''.join(v.to_bytes(1,'big') for v in image.getpalette())

            palette_bytes = image.palette.getdata()[1]
            palette_mode = image.palette.mode
            assert palette_mode in mode2cpp

            cpp = mode2cpp.get(palette_mode)
            assert len(palette_bytes) % cpp == 0

            base = mode2cs.get(palette_mode)
            hival = len(palette_bytes) // cpp - 1
            assert hival <= 255
            palette_xobj = IndirectPdfDict(Filter = PdfName.FlateDecode,
                                            stream = zlib.compress(palette_bytes).decode('Latin-1'))
            cs = PdfArray([PdfName.Indexed, base, hival, palette_xobj])

        icc_profile = image.info.get('icc_profile')
        if icc_profile:
            if mode == 'P':
                cs[1] = PdfColorSpace.make_icc_based_colorspace(icc_profile, N = cpp)
            else:
                cpp = mode2cpp.get(mode)
                cs = PdfColorSpace.make_icc_based_colorspace(icc_profile, N = cpp)

        if not cs:
            cs = mode2cs[mode]

        return cs

    # -------------------------------------------------------------------------------------- _pil_set_colorspace()

    @staticmethod
    def _pil_set_colorspace(image:Image.Image, cs:CS_TYPE = None):
        '''
        '''
        if cs == None:
            del image.palette
            del image.info['icc_profile']
            return

        # Palette
        palette_cs, palette_array = PdfColorSpace.get_palette(cs)
        if palette_cs != None:
            palette_mode = PdfColorSpace.get_mode(palette_cs, 8)
            assert palette_mode == 'RGB'
            assert palette_array.shape[-1] == 3

            if image.mode == '1':
                image = Image.fromarray(np.array(image).astype(np.uint8))
                assert image.mode == 'L'

            image.putpalette(palette_array.tobytes(), rawmode='RGB')

        # ICC Profile
        if icc_profile := PdfColorSpace.get_icc_profile(cs):
            image.info['icc_profile'] = icc_profile

    # -------------------------------------------------------------------------------------- _pil_split_alpha()

    @staticmethod
    def _pil_split_alpha(image:Image.Image):
        '''
        Returns a tuple (base, alpha) if the PIL image contains an alpha channel (i.e., if its mode
        is PA, LA or RGBA), or a tuple (pil, None) otherwise.
        '''
        alpha = None

        if image.mode == 'PA':

            icc_before = image.info.get('icc_profile')
            image = image.convert('RGBA')
            assert icc_before == image.info.get('icc_profile')

        if image.mode in ['LA', 'RGBA']:

            icc_profile = image.info.get('icc_profile')
            if image.mode == 'LA':
                image, alpha = image.split()
            if image.mode == 'RGBA':
                r, g, b, alpha = image.split()
                image = Image.merge('RGB',(r,g,b))
            if 'icc_profile' in alpha.info:
                del alpha.info['icc_profile']
            if icc_profile:
                image.info['icc_profile'] = icc_profile

        return image, alpha

# ============================================================================= jbig2_compress()

def jbig2_compress(bitonal_images:dict, cpc:bool = False):
    '''
    Compress bitonal images with the JBIG2 codec
    '''
    if len(bitonal_images) == 0: warn('no images to compress') ; return
    msg('started')

    def RUN(cmdList:list):
        if not isinstance(cmdList, list): sys.exit(f'bad cmdList: {cmdList}')
        result = subprocess.run(cmdList, capture_output=True)
        if result.returncode:
            print(f'ERROR: command failed: {cmdList}')
            print(result.stdout.decode('utf-8'))
            print(result.stderr.decode('utf-8'))
            sys.exit(1)

    def intToPrefix(i:int):
        s = ''
        for _ in range(3):
            s = chr(ord('a') + i % 26) + s
            i = i // 26
        return s


    with tempfile.TemporaryDirectory() as tmp:
        T = lambda fileName: os.path.join(tmp, fileName)

        image_objects = list(bitonal_images.values())

        # Dump images to tmpDir
        tif_paths = []
        msg(f'decoding and extracting {len(bitonal_images)} bitonal images')
        for n, obj in enumerate(image_objects):
            pprint(obj)
            image = PdfImage(obj = obj)
            array = image.get_array()

            # In order for the JBIG2 compression to work effectively the image has to be mostly white
            # If it's not, invert it and change the /Decode array appropriately
            if np.mean(array) < 0.5:
                image.set_array(np.logical_not(array))
                decode = [float(x) for x in obj.Decode] if obj.Decode != None else None
                obj.Decode = None if decode == [1,0] else PdfArray([1,0])
                image.Decode = obj.Decode

            tif_stream, _ = image.saveAs('TIFF')
            tif_path = T(f'in-{n:04d}.tif')
            open(tif_path, 'wb').write(tif_stream)
            tif_paths.append(tif_path)

        if cpc:
            cpcCmd = ['wine', '/Users/user/Code/Win/CPCTool-530-Win32-X86_regged.exe']
            msg(f'compressing {len(tif_paths)} bitonal images with CPCTool')
            RUN(cpcCmd + tif_paths +['-o', T('cpc.cpc')])
            RUN(cpcCmd + [T('cpc.cpc'), '-o', T('cpc.tif')])
            RUN(['tiffsplit', T('cpc.tif'), T('split-')])

            for i, name in enumerate(tif_paths):
                shutil.move(T('split-') + intToPrefix(i) + '.tif', name)

        msg(f'compressing {len(tif_paths)} bitonal images with JBIG2')
        RUN(['jbig2', '-p', '-s', '-t', '0.97999', '-b', T('result')] + tif_paths)

        try:
            globals = IndirectPdfDict(stream = py23_diffs.convert_load(open(T('result.sym'),'rb').read()))
        except:
            globals = None
        for n, obj in enumerate(image_objects):
            obj.Filter = PdfName.JBIG2Decode
            obj.DecodeParms = PdfDict(JBIG2Globals = globals) if globals else None
            obj.stream = py23_diffs.convert_load(open(T(f'result.{n:04d}'),'rb').read())

    msg('ended')

def fix_png_predictors(obj:IndirectPdfDict):
    '''
    For all filters that use PNG predictors (PDF Ref. p. 76),
    the function will set the value of the `/Predictor` parameter to 15:
    the actual value of the predictor parameter doesn't matter, according the PDF Ref.,
    as long as it is in the PNG predictors range (10..15).
    Setting it to 15 prevents errors in some PDF viewers/processors that erroneously
    base assumptions about the actual predictor values used in the stream on the value of
    the `/Predictor` entry. Returns True if the object's predictors have been modified.
    '''
    assert isinstance(obj, PdfDict)
    assert obj.Subtype == PdfName.Image

    filters = encapsulate(obj.Filter)
    parms = encapsulate(obj.DecodeParms)

    MODIFIED = False
    for f in range(len(filters)):

        filter = filters[f]
        parm = parms[f] if f < len(parms) else PdfDict()
        if parm == 'null': parm = PdfDict()

        if filter in ['/FlateDecode', '/Fl', '/LZWDecode', '/LZW']:
            predictor = int(get_key(parm, '/Predictor', '1'))
            if 10 <= predictor <= 14:
                parms[f].Predictor = 15
                obj.DecodeParms = decapsulate(parms)
                MODIFIED = True

    if obj.SMask:
        MODIFIED |= fix_png_predictors(obj.SMask)

    return MODIFIED

# ============================================================================= modify_image_xobject()

def modify_image_xobject(image_obj:IndirectPdfDict, pdfPage:PdfDict, options):
    '''
    This function performs in-place modifications of an image object, such as: JPEG/JPEG2000
    compression, resizing/upsampling, color space conversions.
    '''

    MODIFIED = False

    image = PdfImage(obj = image_obj)

    intent = image_obj.Intent if options.intent == 'native' else options.intent


    # PDF Ref. v1.7, p.555: if SMask is present and has a Matte attribute,
    # the dimensions of the SMask and the parent image must equal.
    # Therefore, any resizing of images with Matte SMasks must be done
    # together with the resizing of the SMasks. We resize both at all times,
    # not just when the Matte attribute is preset. Just something to keep in mind.

    bigEnough = min(image.w(), image.h()) > (options.mindim or 0)

    # Subsample
    N = options.subsample
    if N and N > 1 and bigEnough:
        a = image.get_array()
        msg(f'subsampling /{N}')
        image.set_array(a[::N, ::N])
        MODIFIED = True
        if image.SMask:
            msg('subsampling SMask')
            modify_image_xobject(image.SMask, pdfPage, options)

    if image.bpc == 1:

        # scale2x
        if options.scale2x:
            a = image.get_array()
            msg(f'upsampling with scale2x')
            image.set_array(SImage.scale2x(a)[0])
            MODIFIED = True

        # despeckle
        if options.despeckle:
            a = image.get_array()
            msg(f'despeckling, threshold = {options.despeckle}')
            image.set_array(SImage.despeckle(a, threshold = options.despeckle))
            MODIFIED = True

    else:

        # Detect gray/color images that are pure b/w and convert them to bitonal
        if options.auto:
            image.render(pdfPage = pdfPage)
            gray = SImage.toGray(image.get_array())
            if np.all(np.logical_or(gray == 0, gray == 255)):
                msg('auto-thresholding')
                image = PdfImage(array = gray > 127)
                MODIFIED = True

        # colorspace
        if options.colorspace:
            mode = {'cmyk':'CMYK', 'rgb':'RGB', 'gray':'L', 'grey':'L'}.get(options.colorspace.lower())
            MODIFIED = image.change_mode(mode, intent)

        # contrast
        if options.contrast and image.get_mode() in ['RGB', 'CMYK']:
            a = image.get_array()
            msg('correcting tint' if options.tint else 'adjusting luminosity')
            ranges = SImage.getRanges(a, mode=options.contrast, method=options.hist) if options.tint \
                        else SImage.getRanges(SImage.toGray(a), mode = options.contrast, method=options.hist)
            if image_obj.Decode is None and options.tint and options.preserve:
                p = lambda x: round(x * 10000)/10000
                ranges = [(r[0]/255, r[1]/255) for r in ranges]
                decode = [(p(-r[0]/(r[1]-r[0])), p((1-r[0])/(r[1]-r[0]))) for r in ranges]
                msg(f'preserving original image data, using DECODE array to correct tint: {decode}')
                image_obj.Decode = PdfArray([f for d in decode for f in d])
                return True
            else:
                msg(f'replacing image')
                image.set_array(SImage.contrast(a, ranges = ranges))
                MODIFIED = True

        # upsample
        if options.upsample:
            if PdfColorSpace.get_name(image.ColorSpace) == '/Indexed':
                image.render()
            a = image.get_array()
            msg(f'upsampling: alpha = {options.alpha}, bounds = {options.bounds}')
            image.set_array(SImage.superResolution(a, alpha = options.alpha, bounds = options.bounds))
            MODIFIED = True
            if image.SMask:
                msg('upsampling SMask')
                modify_image_xobject(image.SMask, pdfPage, options)

        # resample
        if options.resample and (options.resample >= 1 or bigEnough):
            f = options.resample
            if PdfColorSpace.get_name(image.ColorSpace) == '/Indexed':
                image.render()
            a = image.get_array()
            msg(f'resampling: factor = {f}, method = bicubic')
            image.set_array(SImage.resize(a, int(image.w() * f), int(image.h() * f), method = options.method))
            MODIFIED = True
            if image.SMask:
                msg('resampling SMask')
                modify_image_xobject(image.SMask, pdfPage, options)

        # bitonal
        if options.bitonal and image.get_mode() != '1':
            image.render(pdfPage = pdfPage)
            cs_name = PdfColorSpace.get_name(image.ColorSpace)
            msg(f'converting {cs_name} --> L')
            image.change_mode('L', intent)
            a = image.get_array()
            msg('converting L -> 1')
            image = PdfImage(array = SImage.toBitonal(a, threshold = options.threshold))
            MODIFIED = True

    # zip
    if options.zip and image.bpc != 1:
        msg(f'converting to zip')
        image.set_array(image.get_array())
        MODIFIED = True

    if MODIFIED or options.colorspace and image.ColorSpace != image_obj.ColorSpace:
    
        obj = image.encode(isMask = image_obj.ImageMask)

        image_obj.Filter = obj.Filter
        image_obj.stream = obj.stream
        image_obj.ColorSpace = obj.ColorSpace
        image_obj.BitsPerComponent = obj.BitsPerComponent
        image_obj.DecodeParms = obj.DecodeParms
        image_obj.Decode = obj.Decode
        image_obj.Width = obj.Width
        image_obj.Height = obj.Height

    return MODIFIED
 
def getPageRange(s:str):
    '''
    Parses a page range string formatted as 'N1[,N2-N3[,..]]' and returns a list of the form: [N1, N2, N2+1, ... N3].
    '''
    interval = lambda i: i if len(i) == 1 else [n for n in range(i[0], i[1]+1)] if len(i) == 2 else []
    return [n for r in re.split(',', options.pages) for n in interval([int(x) for x in re.split('-',r)]) ]

# ============================================================================= main()

if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument('inputPaths', nargs='+', metavar='FILE', help='input files: images or PDF')
    ap.add_argument('-output', '-o', type=str, metavar='PATH', help='output PDF file path')
    ap.add_argument('-pages', type=str, metavar='RANGE', help='process selected pages; RANGE = N1[,N2-N3[,..]]')
    ap.add_argument('-dpi', type=float, metavar='N', help='set resolution of input images to DPI')
    ap.add_argument('-debug', action='store_true', help='turns debugging on')

    ap.add_argument('-bitonal', action='store_true', help='convert color/gray images to bitonal using Otsu\'s algorithm')
    ap.add_argument('-auto', action='store_true', help='detect if gray/color images are in fact bitonal and convert them')
    ap.add_argument('-threshold', type=int, metavar='T', default=0, help='threshold adjustment; T = -255..+255; def = 0')

    ap.add_argument('-scale2x', action='store_true', help='upsample bitonal images with the scale2x algorithm')
    ap.add_argument('-despeckle', type=int, default=0, metavar='T', help='despeckle bitonal images; T = [-12..12]; def = 0')
    ap.add_argument('-jbig2', action='store_true', help='compress bitonal images with JBIG2 (lossless)')
    ap.add_argument('-cpc', action='store_true', help='pre-process bitonal images with CPCTool before JBIG2 compression')

    ap.add_argument('-dict', type=int, default=20, metavar='N', help='pages per JBIG2 symbol dictionary; def = 20')

    ap.add_argument('-rle', action='store_true', help='compress color/gray images with RLE (Run-Length Encoding)')
    ap.add_argument('-zip', action='store_true', help='compress color/gray images with ZIP')
    ap.add_argument('-png', action='store_true', help='re-compress color/gray images compressed: ZIP -> PNG (with predictors)')
    ap.add_argument('-jpeg', action='store_true', help='compress color/gray images with JPEG')
    ap.add_argument('-j2k', action='store_true', help='compress color/gray images with JPEG 2000')
    ap.add_argument('-quality', '-q', type=int, metavar='Q', default=95, help='JPEG/JPEG 2000 compression quality; Q=0..100 (def=95)')
    ap.add_argument('-psnr', type=int, nargs='?', metavar='P', const = 35, help='PSNR for use with -j2k; def = 35')
    ap.add_argument('-minsize', type=int, metavar='S', help='Minimum image size in bytes to optimize with -jpeg/-j2k')
 
    ap.add_argument('-colorspace', '-cs', type=str, metavar='S', choices=['gray', 'rgb', 'cmyk'], help='convert images to color space; S = gray|rgb|cmyk')

    ap.add_argument('-contrast', type=str, nargs='?', metavar='Mode', const='both', choices=['lows','highs','both'], help='boost contrast; Mode = [lows|highs|both], def=both')
    ap.add_argument('-hist', type=int, metavar='Method', choices=[1,2,3], help='histogram estimation method for use with -contrast, def = 1')
    ap.add_argument('-tint', action='store_true', help='when used with -contrast, removes tint')
    ap.add_argument('-preserve', action='store_true', help='when used with -contrast -tint, uses Decode arrays, preserves original image data intact')

    ap.add_argument('-intent', type=str, metavar='I', choices = ['absolute', 'relative', 'perceptual', 'saturation', 'native', 'none'],
                    default = 'perceptual',
                    help='rendering intent; I = absolute|relative|perceptual|saturation|native|none (def=\'perceptual\')')

    ap.add_argument('-predictors', action='store_true', help='fixes incorrect PNG predictor values in images\' DecodeParms dicts')

    ap.add_argument('-subsample', type=int, metavar='N', help='subsample images: keep only pixels whose coordinates are multiples of N')
    ap.add_argument('-resample', type=float, metavar='F', help='resample color/gray images by factor F with bicubic interpolation')
    ap.add_argument('-method', type=str, metavar='M', choices=['nearest','bilinear','bicubic'] , help='resampling method (nearest|bilinear|bicubic); def = auto')
    ap.add_argument('-mindim', type=int, metavar='D', help='Minimum image dimensions in pixels to downsample or resample with factor < 1')

    ap.add_argument('-upsample', action='store_true', help='upsample color/gray images x2 using Bayesian algorithm')
    ap.add_argument('-alpha', type=int, metavar='A', default=2, help='alpha for the upsampling algo, higher=sharper; A=1..10, def=2')
    ap.add_argument('-bounds', type=str, metavar='B', default='none', choices=['softmax','local','none'], help='bounds for the upsampling algo, B=softmax|local|none, def=none')

    ap.add_argument('-descreen', type=float, metavar='C', help='descreen images; try confidence C = 3 (in units of stand. dev.)')

    ap.add_argument('-interpolate', type=str, nargs='?', metavar='BOOL', choices=['true','false'], help='set /Interpolate flag on images to true/false')
    ap.add_argument('-mask', action='store_true', help='treat all bitonal images as image masks (make bg transparent)')

    options = ap.parse_args()

    LINE_SINGLE = '-'*64
    LINE_DOUBLE = '='*64

    intents =  {'absolute':'/AbsoluteColorimetric',
                'relative':'/RelativeColorimetric',
                'saturation':'/Saturation',
                'perceptual':'/Perceptual',
                'native':'native',
                'none':'none'}
    options.intent = intents.get(options.intent, None)
    assert options.intent != None

    fileBase, fileExt = os.path.splitext(options.inputPaths[0])

    # ---------- Input is PDF: compress and extract images ----------
    if fileExt.lower() == '.pdf':

        assert len(options.inputPaths) == 1
        pdf = PdfReaderX(options.inputPaths[0])
        N = len(pdf.pages)
        eprint(f"[PAGES]: {N}")

        pageRange = getPageRange(options.pages) if options.pages else [n+1 for n in range(N)]

        # JBIG2 compression
        if options.jbig2:
            cache = set()
            bitonal_images = {}
            pageCount = 0
            for pageNo in pageRange:
                print(f'[{pageNo}]')
                page = pdf.pages[pageNo-1]

                bitonal_images_new = {
                    id(obj):obj
                        for name, obj in PdfObjects(page, cache=cache) 
                            if isinstance(obj, PdfDict) 
                                and obj.Subtype == PdfName.Image
                                and (
                                        obj.BitsPerComponent == '1' 
                                        or obj.ImageMask == PdfObject('true')
                                    )
                }

                print(f'Adding {len(bitonal_images_new)} images')
                bitonal_images = bitonal_images | bitonal_images_new

                pageCount += 1
                if pageCount == options.dict:
                    jbig2_compress(bitonal_images, cpc = options.cpc)
                    pageCount = 0
                    bitonal_images = {}
                    cache = set()
            
            if len(bitonal_images) > 0:
                jbig2_compress(bitonal_images, cpc = options.cpc)

            pdfOutPath = fileBase + ('-cpc' if options.cpc else '') + f'-jbig2' + fileExt
            print(LINE_DOUBLE)
            print(f'Writing output to {pdfOutPath}')
            PdfWriter(pdfOutPath, trailer=pdf, compress=True).write()

            sys.exit()

        # Iterate over pages
        test = lambda name, obj: isinstance(obj, PdfDict) and obj.Subtype == PdfName.Image \
                    and name not in [PdfName.Mask, PdfName.SMask]

        PROCESSING_REQUESTED = False

        MODIFIED_GLOBAL = False

        cache = set()

        for pageNo in pageRange:


            page = pdf.pages[pageNo-1]

            objects = PdfObjects(page, test = test, cache = cache)

            images = {name+f'_{id(obj)}':obj for name, obj in objects 
                        if isinstance(obj, PdfDict) and obj.Subtype == PdfName.Image 
                        and name not in [PdfName.Mask, PdfName.SMask]}


            # print(page)
            if len(images) == 0:
                continue

            try: defaultColorSpaces = page.Resources.ColorSpace.keys()
            except: defaultColorSpaces = None

            print(LINE_SINGLE)
            print(f'Page {pageNo}, # images: {len(images)}, page cs: {defaultColorSpaces}')

            for n,idx in enumerate(images):

                MODIFIED = False

                obj_original = images[idx]
 
                obj = obj_original.copy()
 
                print(f'[{pageNo}:{n+1}] {PdfImage.xobject_str(obj)}')

                # ---------- Modify images ----------

                if options.predictors:

                    PROCESSING_REQUESTED = True
                    res = fix_png_predictors(obj)
                    if res:
                        print(' --> fixed PNG predictor')
                    MODIFIED |= res

                if options.upsample \
                        or options.resample \
                        or options.subsample \
                        or options.colorspace \
                        or options.contrast \
                        or options.bitonal \
                        or options.despeckle \
                        or options.scale2x \
                        or options.auto:

                    PROCESSING_REQUESTED = True

                    pageArg = page if options.colorspace else None

                    res = modify_image_xobject(obj, pageArg, options)
                    MODIFIED |= res

                # ---------- Descreen images ----------
                if options.descreen:

                    PROCESSING_REQUESTED = True

                    if PdfImage.xobject_get_bpc(obj) != 1:
                        msg('descreening image')
                        image = PdfImage(obj = obj)
                        image.set_array(SImage.descreen(image.get_array(), options.descreen))
                        obj_new = image.encode()
                        PdfImage.xobject_copy(obj_new, obj)
                        MODIFIED = True

                # ---------- Recompress images ----------

                if options.jpeg or options.j2k or options.zip or options.png or options.rle:

                    PROCESSING_REQUESTED = True

                    if PdfImage.xobject_get_bpc(obj) != 1:
                        Format = 'JPEG' if options.jpeg \
                                    else 'JPEG2000' if options.j2k \
                                    else 'RLE' if options.rle \
                                    else 'PNG' if options.png \
                                    else 'ZIP'
                        msg(f're-compressing with {Format}')
                        res = PdfImage.recompress(obj,
                                                    Format = Format,
                                                    Q = options.quality,
                                                    PSNR = options.psnr,
                                                    minsize = options.minsize)

                        size_old = len(obj_original.stream)
                        size_new = len(obj.stream)
                        print(LINE_SINGLE)
                        if res:
                            print(f'Compressed: {size_old} -> {size_new}')
                            MODIFIED = True
                        else:
                            print(f'Unmodified {size_old} -> {size_new}')
                            MODIFIED = False

                        # if image_obj.SMask:
                        #     msg('recompressing mask')
                        #     PdfImage.recompress(image_obj.SMask, Format='JPEG', Q=options.quality)

                if options.interpolate:
                    PROCESSING_REQUESTED = True
                    newInterpolate = PdfObject('true') if options.interpolate == 'true' else PdfObject('false')
                    if obj.Interpolate != newInterpolate:
                        obj.Interpolate = newInterpolate
                        MODIFIED = True

                if MODIFIED:
                    PdfImage.xobject_copy(obj, obj_original)
                    print("Result:")
                    pprint(obj_original)

                MODIFIED_GLOBAL |= MODIFIED

                # ---------- Extract images ----------
                if not PROCESSING_REQUESTED:

                    # image = PdfImage.decode(obj,page, adjustColors = False, applyMasks = True, applyIntent = options.applyIntent)
                    image = PdfImage(obj=obj)
                    if options.dpi:
                        image.dpi = (options.dpi, options.dpi)
                        print("DEBUG:", image.dpi)
                    stream, ext = image.saveAs('auto', render = True)
                    outputPath = fileBase  +f'.page{pageNo:04d}.image{n+1:04d}' + ext
                    open(outputPath, 'wb').write(stream)

        # ---------- Write processed results to PDF ----------
        if MODIFIED_GLOBAL:
            suffix = ''
            if options.auto: suffix += f'-auto'
            if options.despeckle: suffix += f'-despeckle={options.despeckle}'
            if options.scale2x: suffix += f'-scale2x'
            if options.upsample: suffix += f'-upsample={options.alpha}'
            if options.bounds != 'none': suffix += f'-bounds={options.bounds}'
            if options.resample: suffix += '-resample'
            if options.subsample: suffix += '-subsample'
            if options.colorspace: suffix += f'-{options.colorspace}'
            if options.contrast: suffix += '-contrast'
            if options.zip: suffix += f'-zip'
            if options.rle: suffix += f'-rle'
            if options.png: suffix += f'-png'
            if options.jpeg: suffix += f'-jpeg-q={options.quality}'
            if options.j2k: suffix += f'-j2k-psnr={options.psnr}' if options.psnr else f'-j2k-q={options.quality}'
            if options.bitonal: suffix += '-bitonal'
            if options.predictors: suffix += '-pred'
            if options.descreen: suffix += f'-descreen={options.descreen}'
            if options.interpolate == 'true': suffix += f'-interpolate=true'
            if options.interpolate == 'false': suffix += f'-interpolate=false'
            assert suffix != ''
            pdfOutPath = fileBase + suffix + fileExt
            print(LINE_DOUBLE)
            if (options.j2k): print('Warning: *** JPEG 2000 output may be buggy; please check manually ***')
            print(f'Writing output to {pdfOutPath}')
            PdfWriter(pdfOutPath, trailer=pdf, compress=True).write()
        else:
            print(f'[pdfimage.py] file not modified, no output is produced')

    # ---------- Input is images: convert to PDF ----------
    else:

        pdfPath = options.output or options.inputPaths[0]+".pdf"

        pdf = PdfWriter(pdfPath,compress=True)
        N = len(options.inputPaths)
        for n,imagePath in enumerate(options.inputPaths):
            base,ext = os.path.splitext(imagePath)
            if ext.lower() == '.jp2':
                image = PdfImage(jp2 = open(imagePath,'rb').read())
            elif ext.lower() in ['.jpg', '.jpeg']:
                image = PdfImage(jpeg = open(imagePath,'rb').read())
            else:
                image = PdfImage(pil = Image.open(imagePath))

            print(f'[{n+1}/{N}] {image}')
            print(f'[{n+1}/{N}] {imagePath}')

            image.dpi = (options.dpi, options.dpi) if options.dpi != None \
                    else image.dpi if image.dpi not in [None, (1,1)] \
                    else (72,72)

            page = image.to_pdf_page(isMask = options.mask)
            pdf.addPage(page)

        pdf.write()
        print(f'Output written to {pdfPath}')
