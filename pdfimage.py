#!/usr/bin/env python3

# Write this man about the bug in PIL Image.paste()!
# https://sudonull.com/post/129265-Alpha_composite-optimization-history-in-Pillow-20
# https://habr.com/ru/post/98743/
# https://github.com/homm/
# https://twitter.com/wouldntfix

import argparse, struct, zlib, base64, sys, re, os, subprocess, tempfile

from io import BytesIO
import numpy as np

from PIL import Image, TiffImagePlugin, ImageChops, ImageCms
Image.MAX_IMAGE_PIXELS = None

from glymur import Jp2k


from pdfrw import PdfReader, PdfWriter, PdfObject, PdfName, PdfArray, PdfDict, IndirectPdfDict, py23_diffs
from pdfrwx.common import err, msg, warn, eprint, get_key, get_any_key
from pdfrwx.pdffilter import PdfFilter
from pdfrwx.pdfobjects import PdfObjects
from pdfrwx.pdffunctionparser import PdfFunction

from pprint import pprint

CMYK_DEFAULT_ICC_PROFILE = open(os.path.join(os.path.dirname(__file__),'color_profiles/USWebCoatedSWOP.icc'), 'rb').read()

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

class PdfColorSpace(PdfArray):

    # A mapping from color space name to components per pixel; None means that cpp varies
    spaces = {
        '/DeviceGray': 1, '/CalGray': 1, '/Indexed': 1, '/Separation': 1,
        '/DeviceRGB': 3, '/CalRGB': 3, '/Lab': 3, '/DeviceCMYK': 4,
        '/ICCBased': None, 'DeviceN': None, '/NChannel': None
    }

    # A mapping from components per pixel to PIL image modes
    modes = {1:'L', 3:'RGB', 4:'CMYK'}

    def get_name(cs):
        '''
        Returns the name of the color space (on of the PdfColorSpace.spaces.keys())
        '''
        name = cs if not isinstance(cs,PdfArray) \
                    else '/NChannel' if cs[0] == '/DeviceN' and len(cs) == 5 and cs[4].Subtype == '/NChannel' \
                    else cs[0]
        if name not in PdfColorSpace.spaces:
            raise ValueError(f'bad color space: {cs}')
        return name

    def get_cpp(cs):
        '''
        Return the number of components per pixel for a give color space.
        '''
        name = PdfColorSpace.get_name(cs)
        return int(cs[1].N) if name == '/ICCBased' else \
                len(cs[1]) if name in ['/DeviceN', '/NChannel'] else \
                PdfColorSpace.spaces.get(name, None)

    def get_mode(cs):
        '''
        '''
        return PdfColorSpace.modes.get(PdfColorSpace.get_cpp(cs), None)

    def get_icc_profile(cs):
        '''
        For calibrated /CalGray, /CalRGB, /Lab and /ICCBased color spaces, returns
        the ICC color profile as a bytes object. For all other color spaces returns None.
        '''
        name = PdfColorSpace.get_name(cs)
        if name == '/ICCBased': return py23_diffs.convert_store(PdfFilter.uncompress(cs[1]).stream)
        elif name in ['/CalGray','/CalRGB', '/Lab']: return PdfColorSpace.create_profile(cs)
        else: None

    def get_palette(cs):
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

        return base, np.frombuffer(pal, 'uint8').reshape(-1, palette_cpp)

    def create_profile(cs):
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
                primaries = [primaries[3*i,3*i+3] for i in range(3)]
                primaries = [[X/(X+Y+Z),Y/(X+Y+Z),Y] for X,Y,Z in primaries]
                gamma = [float(v) for v in dic.Gamma] if dic.Gamma != None else [1,1,1]
                transferFunction = [lc.cmsBuildGamma(ctx, g) for g in gamma]
                profile = lc.cmsCreateRGBProfile(white, primaries, transferFunction)
            with BytesIO() as bs:
                lc.cmsSaveProfileToStream(profile, bs)
                icc_profile = bs.getvalue()
        except:
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
    
    def apply_colorspace(colorSpace, array:np.ndarray):
        '''
        Returns the tuple (newColorSpace, icc_profile, array).

        * for device-based color spaces (/DeviceGray, /DeviceRGB, /DeviceCMYK),
        the originals are returned as nothing needs to be done;
        * for /Indexed color spaces the originals are returned; such color spaces should be processed
        separately.
        * for calibrated color spaces (/CalGray, /CalRGB, /Lab, /ICCBased), a copy of the image with
        an ICC profile inserted is returned;
        * for /Separation, /DeviceN and /NChannel color spaces, the image is remapped to the
        alternate color space using the transform function, and the result is returned;

        In last case, the new (target/raw) color space is recursively applied to the result.
        '''
        cs = colorSpace
        name = PdfColorSpace.get_name(cs)
        msg(f'applying color space: {name}')

        if name in ['/CalGray', '/CalRGB', '/Lab', '/ICCBased']:
            icc_profile = PdfColorSpace.get_icc_profile(cs)
            return colorSpace, icc_profile, array

        if name in ['/Separation', '/DeviceN', '/NChannel']:
            altColorSpace, colorTransformFunction = cs[2], cs[3]
            stack = np.moveaxis(array,-1,0)
            stack = PdfFunction(colorTransformFunction).process(stack)
            array = np.moveaxis(stack,0,-1)
            return PdfColorSpace.apply_colorspace(altColorSpace, array)

        return colorSpace, None, array


    def apply_default_page_colorspace_icc_profile(page:PdfDict, image:Image.Image):
        '''
        If image.inf['icc_profile'] == None, apply the ICC profile from the page's default color space,
        if present, to the image (in-place).
        '''
        if page == None or image.info['icc_profile'] == None: return
        try:
            default_cs_names = {'L':'/DefaultGray', 'RGB':'/DefaultRGB', 'CMYK':'/DefaultCMYK'}
            default_cs_name = default_cs_names[image.mode]
            default_cs = page.Resources.ColorSpace[default_cs_name]
            image.info['icc_profile'] = PdfColorSpace.get_icc_profile(default_cs)
            msg(f'applied default page colorspace ICC profile: {PdfColorSpace.get_name(default_cs)}')
        except:
            pass

# ========================================================================== class PdfDecodeArray

class PdfDecodeArray:

    '''
    The class facilitates processing of image.Decode arrays.
    '''

    def get_actual(Decode:PdfArray, colorSpace, bpc):
        '''
        Returns actual Decode array as list[float] if self.image.Decode != None, or self.get_default(colorSpace) otherwise.
        ''' 
        FLOAT = lambda array: [float(a) for a in array]
        return FLOAT(Decode) if Decode != None else PdfDecodeArray.get_default(colorSpace, bpc)

    def get_default(colorSpace, bpc:int):
        '''
        Returns the default Decode array for a give color space; see Adobe PDF Ref. 1.7, table 4.40 (p.345)
        ''' 
        FLOAT = lambda array: [float(a) for a in array]

        cs = colorSpace
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

    def decode(array:np.ndarray, Decode:list, bpc:int):
        '''
        Translates a numpy array from encoded ("stream space": int, 1..2**bpc -1)
        to decoded ("image space", float: mostly 0.0..1.0) representation.
        '''
        msg(f'applying Decode: {Decode}')
        INTERPOLATE = lambda x, xmin, xmax, ymin, ymax: ymin + ((x - xmin) * (ymax - ymin)) / (xmax - xmin)

        N = array.shape[-1] # number of colors
        assert len(Decode) == 2*N
        iMax = float((1<<bpc) - 1)
        array = np.clip(array, 0, iMax).astype(float)
        for i in range(N):
            array[...,i] = INTERPOLATE(array[...,i], 0.0, iMax, Decode[2*i], Decode[2*i + 1])
        return array

    def encode(array:np.ndarray, Decode:list):
        '''
        Translates a numpy array from decoded ("image space", float: mostly 0.0..1.0)
        to encoded ("stream space", uint8: 0..255) representation.
        '''
        msg(f'applying Encode: {Decode}')
        INTERPOLATE = lambda x, xmin, xmax, ymin, ymax: ymin + ((x - xmin) * (ymax - ymin)) / (xmax - xmin)

        N = array.shape[-1] # number of colors
        assert len(Decode) == 2*N
        for i in range(N):
            array[...,i] = INTERPOLATE(array[...,i].astype(float), Decode[2*i], Decode[2*i + 1], 0, 255)
        array = np.clip(np.round(array), 0, 255).astype(np.uint8)
        return array


# ========================================================================== class PdfImage

class PdfImage:

    # -------------------------------------------------------------------- encode()

    def encode(image:Image.Image, \
               compressionFormat:str = None, \
               compressedQuality:float = None, \
               compressionRatio:float = None):
        '''
        Converts a PIL image to a PDF image XObject.
        Returns a tuple (xobj, dpi), where xobj is the created PDF image XObject
        and dpi is a tuple (xdpi, ydpi) of the image's resolution. Processing depends on image.mode/format:
        
        * JPEG (image.format is 'JPEG') images are stored "as-is" using the /DCTDecode filter (no transcoding is done)

        * Bitonal (image.mode is '1') images are compressed using the /CCITTFaxDecode filter with Group4 encoding.

        * All other images are compressed using the /FlateDecode filter.

        For images with transparency, the alpha-channel is stored in xobj.SMask.
        For palette-based/indexed (image.mode == 'P') images, the palette is used to create an /Indexed color space.
        The ICC color profile (image.info['icc_profile']), if present, is used to set up the corresponding
        /ICCBased color space.

        If compressionFormat is 'JPEG' or 'JPEG2000', the lossy compression into that format with
        the specified quality is attempted, and if this reduces the size of the image
        compared to the original lossless compression, the lossy-compressed version is encoded.
        '''

        # A lambda for printing out size changes in the form old_size --> new size (+-change%)
        print_size_change = lambda size_old, size_new: \
            f'Size: {size_old} --> {size_new} ({((size_new - size_old)*100)//size_old:+d}%)'

        w,h = image.size
        DecodeParms = None
        Decode = None
        # ImageMask = None

        # Save the ICC profile so that it's not lost when splitting off the alpha-channel
        icc_profile = image.info.get('icc_profile')

        # Separate alpha channel
        alpha = None
        if image.mode == 'LA':
            image, alpha = image.split()
        if image.mode == 'RGBA':
            r, g, b, alpha = image.split()
            image = Image.merge('RGB',(r,g,b))

        # Encode alpha
        if alpha != None:
            image.info['icc_profile'] = icc_profile
            # msg('Image encoding started')
            xobj, dpi = PdfImage.encode(image, compressionFormat, compressedQuality, compressionRatio)
            # msg('Image encoding ended')
            if 'icc_profile' in alpha.info: del alpha.info['icc_profile'] # alpha should not have icc_profile
            # msg('Mask encoding started')
            xobj_alpha, _ = PdfImage.encode(alpha, compressionFormat, compressedQuality, compressionRatio)
            # msg('Mask encoding ended')
            xobj.SMask = xobj_alpha
            return xobj, dpi

        modes = {
            '1':(PdfName.DeviceGray,1,1),
            'L':(PdfName.DeviceGray,1,8),
            'RGB':(PdfName.DeviceRGB,3,8),
            'CMYK':(PdfName.DeviceCMYK,4,8)
        }

        # Make sure not to use mode anywhere outside this block
        mode = image.mode if image.mode != 'P' else image.palette.mode
        if mode not in modes: raise ValueError(f"unsupported image or palette mode: {mode} ")
        cs, cpp, bpc = modes[mode]
    
        # Set up /ICCBased color space
        if icc_profile != None:
            icc_xobj = IndirectPdfDict(
                N = cpp,
                Filter = PdfName.FlateDecode,
                stream = py23_diffs.convert_load(zlib.compress(icc_profile))
            )
            cs = PdfArray([PdfName.ICCBased, icc_xobj])

        # Set up /Indexed color space
        if image.mode == 'P':
            palette = b''.join(v.to_bytes(1,'big') for v in image.getpalette())
            palette_xobj = IndirectPdfDict(
                Filter = PdfName.FlateDecode,
                stream = py23_diffs.convert_load(zlib.compress(palette))
            )
            # Current color space becomes base for the /Indexed color space!
            cs = PdfArray([PdfName.Indexed, cs, len(palette)//3, palette_xobj])

        # Start encoding

        if image.format == 'JPEG':

            # msg('JPEG --> /DCTDecode')

            with BytesIO() as bs:
                image.save(bs, format='JPEG', quality='keep')
                stream = bs.getvalue()
                if stream[-2:] != b'\xff\xd9': stream += b'\xff\xd9' # fix a bug found in some jpeg files
            filter = 'DCTDecode'

        elif image.format == 'JPEG2000':

            # msg('JPEG 2000 --> /JPXDecode')
            with BytesIO() as bs:
                image.save(bs, format='JPEG2000', quality='keep')
                stream = bs.getvalue()
            filter = 'JPXDecode'

        elif image.mode == '1': # TIFF

            # msg('Bitonal --> /CCITTFaxDecode')
            # msg(image.info)
            stream = ImageUtils.PIL_image_to_PDF_stream_with_TIFF_codecs(image)
            filter = 'CCITTFaxDecode'
            # ImageMask = PdfObject('true')
            DecodeParms = PdfDict(K = -1, Columns = w, Rows = h, BlackIs1 = PdfObject('true'))
            # DecodeParms = PdfDict(K = -1, Columns = w, Rows = h)

        else: # PNG and others

            # msg('Color --> /FlateDecode')
            # ToDo: a) use optional predictors; b) feed the PNG IDAT chunk without re-encoding
            # Update: optimization using predictors is slow and gives only an extra 5% reduction in size


            # The baseline
            stream = zlib.compress(image.tobytes())

            # # Encoding via optimized PNG (uses predictors?)
            # if image.mode in ['L', 'RGB']:
            #     stream_png = ImageUtils.PIL_image_to_PNG_IDAT_chunk(image)
            #     if stream_png != None:
            #         size_zlib, size_png = len(stream), len(stream_png)
            #         if stream_png != None and size_png < size_zlib:
            #             print(f'PNG optimization: {size_zlib} -> {size_png}')
            #             stream = stream_png
            #             DecodeParms = PdfDict(Predictor = 15, Columns = w, Colors = cpp)
            #             # l = len(zlib.decompress(stream_png))
            #             # if not l == (w * cpp + 1) * h:
            #             #     warn(f'size mismatch: expected {w}, got {((l / h - 1)/cpp)}')
            #         else:
            #             print(f'PNG optimization not performed: {size_zlib} -> {size_png}')

            # # The optimization with TIFF Predictor 2
            # if image.mode != 'P':
            #     array = PdfFilter.unpack_pixels(image.tobytes(), w, h, cpp, bpc)
            #     array = np.hstack((array[:,0,:].reshape(h,1,cpp), np.diff(array, axis=1))) # difference
            #     stream_predicted = PdfFilter.pack_pixels(array, w, h, cpp, bpc)
            #     stream_predicted = zlib.compress(stream_predicted)
            #     sizeOld, sizeNew = len(stream), len(stream_predicted)
            #     if sizeNew < sizeOld:
            #         stream = stream_predicted
            #         DecodeParms = PdfDict(Predictor = 2, Columns = w, Colors = 3)
            #         print(f'optimized with TIFF Predictor 2: {sizeOld} -> {sizeNew}')
            #     else:
            #         print(f'skipping TIFF Predictor 2 optimization: {sizeOld} -> {sizeNew}')


            # stream = zlib.compress(image.tobytes())

            filter = 'FlateDecode'


        # Re-compress streams as ZIP if requested; do not recompress palette-based images
        if compressionFormat == 'ZIP' and filter != 'FlateDecode' and image.mode != 'P':
            with BytesIO() as bs:
                stream_compressed = zlib.compress(image.tobytes())

                size, size_compressed = len(stream), len(stream_compressed)
                # size gains of < 10% do not justify loss of quality in re-compression
                # if size_compressed * 10 < size * 9:
                if True:
                    msg(f're-compressed [{compressionFormat}]; ' + print_size_change(size, size_compressed))
                    stream = stream_compressed
                    filter = 'FlateDecode'
                else:
                    msg(f'skipped re-compression [{compressionFormat}]; ' + print_size_change(size, size_compressed))

        # Re-compress streams as JPEG if requested; do not recompress palette-based images
        if compressionFormat in ['JPEG', 'JPEG2000'] and filter in ['FlateDecode', 'DCTDecode', 'JPXDecode'] and image.mode != 'P':
            with BytesIO() as bs:
                if compressionFormat == 'JPEG':
                    assert 0 < compressedQuality <= 100
                    image.save(bs, format='JPEG', quality=compressedQuality, optimize=True)
                    stream_compressed = bs.getvalue()
                else:
                    assert compressionRatio != None or compressedQuality != None

                    CR = int(round(compressionRatio)) if compressionRatio != None \
                        else int(round(2 ** ((100 - compressedQuality)/10.0)))
                    
                    # image.save(bs, format='JPEG2000', quality_mode='rates', quality_layers = [0.1])

                    # # The pylibjpeg-openjpeg is buggy: it produces images of much lower quality
                    # from openjpeg import encode as openjpegEncode
                    # import numpy as np
                    # array = np.array(image)
                    # print(f'JPEG2000 compression ratio: {CR}')
                    # stream_compressed = openjpegEncode(array, compression_ratios=[CR * 4, CR * 3, CR * 2, CR])

                    array = np.array(image)
                    # Jp2k('_glymur_tmp.jp2', array, cratios=[CR*4, CR*2, CR])
                    Jp2k('_glymur_tmp.jp2', array, cratios=[CR])
                    stream_compressed = open('_glymur_tmp.jp2', 'rb').read()
                    os.remove('_glymur_tmp.jp2')

                # fix a bug found in some jpeg files
                if compressionFormat == 'JPEG' and stream[-2:] != b'\xff\xd9':
                    stream += b'\xff\xd9'

            size, size_compressed = len(stream), len(stream_compressed)
            # size gains of < 10% do not justify loss of quality in re-compression
            if size_compressed * 10 < size * 9:
                X = f'CR={int(CR * 100)/100}' if compressionFormat == 'JPEG2000' else f'Q={compressedQuality}'
                msg(f're-compressed [{compressionFormat}, {X}]; ' + print_size_change(size, size_compressed))
                stream = stream_compressed
                filter = 'DCTDecode' if compressionFormat == 'JPEG' else 'JPXDecode'
            else:
                msg(f'skipped re-compression [{compressionFormat}]; ' + print_size_change(size, size_compressed))

        # PDF expects inverted CMYK JPEGs
        if image.mode == 'CMYK' and filter == 'DCTDecode':
            Decode = [1,0,1,0,1,0,1,0]
            msg(f'CMYK JPEG: inserting /Decode = {Decode}')

        # Create the image XObject
        xobj = IndirectPdfDict(
            Type = PdfName.XObject,
            Subtype = PdfName.Image,
            Width = w, Height = h,
            ColorSpace = cs,
            BitsPerComponent = bpc,
            Filter = PdfName(filter),
            # ImageMask = ImageMask,
            DecodeParms = DecodeParms,
            Decode = Decode,
            stream = py23_diffs.convert_load(stream) # Ugly, but necessary: https://github.com/pmaupin/pdfrw/issues/161
        )

        # Determine DPI
        dpi = [float(x) for x in image.info['dpi']] if 'dpi' in image.info else None

        return xobj, dpi

    # -------------------------------------------------------------------- decode()

    def decode(obj:IndirectPdfDict, pdfPage:PdfDict = None, invertCMYK = True, applyMasks = True, applyColorSpace = True, intent:str = 'native'):
        '''
        Returns a tuple (pil_image, encoded), encoded is the raw encoded image bytes stream, or (None, None)
        if decoding fails.

        If applyMasks is True, the soft mask (SMask) of the image, if present, becomes the alpha-channel of the
        resulting image. At this stage, the conversion from the original non-RGB image.mode to
        RGB occurs, which becomes RGBA after the insertion of the alpha-mask. The conversion
        to RGB utilizes image's icc_profile.

        if fixAdobeCMYK is True, the function checks if the image is an Adobe CMYK image
        and, if so, fixes it by calling ImageUtils.pil_image_invert_cmyk_colors_if_needed().

        Intent specifies pixel mapping during color space conversions. Allowed values are:
        '/AbsoluteColorimetric', '/RelativeColorimetric', '/Saturation', '/Perceptual', 'native' or 'none'.

        Note: if invertCMYK == True and the image is a /DecideCMYK with /DCTDecode (JPEG),
        the /Decode array is always (!) applied, even if applyColorSpace == False.
        [EXTEND THE DESCRIPTION]
        '''

        intents = ['/AbsoluteColorimetric', '/RelativeColorimetric', '/Saturation', '/Perceptual', 'native', 'none']

        assert intent in intents

        assert obj.Subtype in ['/Image', PdfName.Image]

        if obj.stream == None:
            warn(f'image has no stream: {obj}')
            return None, None

        # Remove compression for all filters except the image-specific ones
        obj = PdfFilter.uncompress(obj)

        # pprint(obj)

        stream = py23_diffs.convert_store(obj.stream) if isinstance(obj.stream,str) else obj.stream

        filter, parm = obj.Filter, obj.DecodeParms

        width, height = int(obj.Width), int(obj.Height)

        bpc = PdfImage.get_bpc(obj)

        cs = PdfImage.get_colorspace(obj)
        cs_name = PdfColorSpace.get_name(cs)
        cpp = PdfColorSpace.get_cpp(cs)
        mode = PdfColorSpace.get_mode(cs) if bpc != 1 else '1'

        # msg(f'image specs: {bpc}, {cs}')

        img = None
        encoded = None
        applyDecodeToDeviceCMYK = True

        # ------------------------------------------------------------------------------------- Filters -> Image

        if filter == None:

            pass

        elif filter in ['/CCITTFaxDecode', '/CCF']: # --> TIFF

            # msg('/CCITTFaxDecode --> TIFF')
            K = get_key(parm, '/K', '0')
            BlackIs1 = get_key(parm, '/BlackIs1', 'false')
            tiff_compression = 3 if int(K) >= 0 else 4
            tiff_photometric = 0 if BlackIs1 == 'true' else 1
            if obj.ImageMask == PdfObject('true'): tiff_photometric = 1 - tiff_photometric
            encodedByteAlign = get_key(parm, '/EncodedByteAlign') == 'true'
            if encodedByteAlign:
                warn(f'/EncodedByteAlign == True not supported yet; skipping')
                return None, None
            img = ImageUtils.PDF_stream_to_PIL_TIFF(width, height, 1, 1,
                                                    tiff_compression, tiff_photometric, stream,
                                                    encodedByteAlign = encodedByteAlign)

        elif filter == '/JBIG2Decode': # JBIG2 --> bitonal PNG

            locals = py23_diffs.convert_store(obj.stream)
            globals = py23_diffs.convert_store(PdfFilter.uncompress(obj.DecodeParms.JBIG2Globals).stream) \
                if obj.DecodeParms != None and obj.DecodeParms.JBIG2Globals != None else None
            with tempfile.TemporaryDirectory() as tmp:
                T = lambda fileName: os.path.join(tmp, fileName)
                open(T('locals'),'wb').write(locals) 
                if globals:
                    open(T('globals'),'wb').write(globals)
                    OS.execute(['jbig2dec', '-e', '-o', T('out.png'), T('globals'), T('locals')])
                else:
                    OS.execute(['jbig2dec', '-e', '-o', T('out.png'), T('locals')])
                img = Image.open(T('out.png'))

            # If you have to re-assemble the JBIG2 file for some reason, just add the header:
            # 1-page, sequential order; see JBIG2 Specs
            # header = b'\x97\x4A\x42\x32\x0D\x0A\x1A\x0A\x01\x00\x00\x00\x01'

        elif filter in ['/DCTDecode', '/DCT']: # --> JPEG

            # msg('/DCTDecode --> JPEG')
            if (ct := get_any_key(parm, '/ColorTransform')) is not None: warn(f'/DecodeParms: /ColorTransform {ct} ignored')
            if (ct := get_any_key(parm, '/Decode')) is not None: warn(f'/DecodeParms: /Decode {ct} ignored')
            img = Image.open(BytesIO(stream))
            encoded = ['JPEG', stream]

            # print(img.info)
            # # img.info.clear()
            # print(img.info)
            
        elif filter == '/JPXDecode': # --> JPEG 2000

            # msg('/JPXDecode --> JPEG2000')
            warn(f'JPEG2000 decoding is buggy')
            img = Image.open(BytesIO(stream))
            encoded = ['JPEG2000', stream]

        else:
            warn(f'unsupported stream filter: {filter}')
            return None, None

        # ------------------------------------------------------------------------------------- Raw stream -> Array

        array = None if img != None else \
            PdfFilter.unpack_pixels(stream, width, height, cpp, bpc, truncate = True)

        # ------------------------------------------------------------------------------------- Invert CMYK

        if invertCMYK and img != None and img.format == 'JPEG' and cs_name == '/DeviceCMYK':
            actualDecode  = PdfDecodeArray.get_actual(obj.Decode, cs, bpc)
            if actualDecode == [1,0,1,0,1,0,1,0]:
                msg('inverted CMYK + inverted /Decode = no change')
                applyDecodeToDeviceCMYK = False
            else:
                msg('inverting CMYK')
                img = ImageChops.invert(img)
                encoded = None
                defaultDecode = PdfDecodeArray.get_default(cs, bpc)
                if actualDecode != defaultDecode:
                    array = np.array(img); img = None
                    array = PdfDecodeArray.decode(array, actualDecode, bpc)
                    array = PdfDecodeArray.encode(array, defaultDecode)

        # ------------------------------------------------------------------------------------- A note

        # From this point on: either img != None, or array != None, but not both

        # ------------------------------------------------------------------------------------- Apply Color Space

        if applyColorSpace:

            # ------------------------------------------------------------------------------------- Decode

            actualDecode  = PdfDecodeArray.get_actual(obj.Decode, cs, bpc)
            defaultDecode = PdfDecodeArray.get_default(cs, bpc)

            # ------------------------------------------------------------------------------------- Apply colorSpace

            if cs_name == '/Indexed':

                # In the /Indexed color space, all color space transformations are performed on the palette.
                # Note: palette_cs is never /Indexed (see Adobe PDF Ref.)

                # PDF standard allows for remapping of palette indices via Decode arrays (why?)
                if actualDecode != defaultDecode:
                    if array is None: array = np.array(img); img = None
                    array = PdfDecodeArray.decode(array, actualDecode, bpc)
                    array = np.clip(np.round(array),0,255).astype('uint8') # clip right away: no encoding step later
                
                if img == None:
                    img = Image.fromarray(array[...,0], mode='P')
                    array = None
    
                # Get palette
                palette_cs, palette_array, = PdfColorSpace.get_palette(cs)

                # Decode palette
                palette_decode = PdfDecodeArray.get_default(palette_cs, 8)
                palette_array = PdfDecodeArray.decode(palette_array, palette_decode, 8)

                # Apply palette_cs to palette_array; both palette_array & palette_cs can change
                palette_cs, icc_profile, palette_array = \
                    PdfColorSpace.apply_colorspace(palette_cs, palette_array)

                # Encode palette
                palette_decode = PdfDecodeArray.get_default(palette_cs, 8)
                palette_array = PdfDecodeArray.encode(palette_array, palette_decode)

                img.putpalette(palette_array.tobytes(), rawmode=PdfColorSpace.get_mode(palette_cs))

            elif cs_name in ['/Separation', '/DeviceN', '/NChannel']:

                # img -> array
                if array is None: array = np.array(img); img = None; encoded = None

                # Decode
                array = PdfDecodeArray.decode(array, actualDecode, bpc)

                # Apply cs to array; both array & cs can change
                cs, icc_profile, array = PdfColorSpace.apply_colorspace(cs, array)

                # Encode
                defaultDecode = PdfDecodeArray.get_default(cs, 8)
                array = PdfDecodeArray.encode(array, defaultDecode)


            elif cs_name == '/JPX': # Internal color space (surrogate) from the JPEG2000 file stream

                icc_profile = img.info['icc_profile'] if img != None and img.format == 'JPEG2000' else None

            else:

                # Encode & decode, if necessary
                if actualDecode != defaultDecode and applyDecodeToDeviceCMYK:
                    # img -> array
                    if array is None: array = np.array(img); img = None; encoded = None
                    array = PdfDecodeArray.decode(array, actualDecode, bpc)
                    array = PdfDecodeArray.encode(array, defaultDecode)

                # Get ICC Profile
                icc_profile = PdfColorSpace.get_icc_profile(cs) \
                    if cs_name in ['/CalGray', '/CalRGB', '/Lab', '/ICCBased'] else None

        # ------------------------------------------------------------------------------------- Make image

        # Make image
        if img == None:
            mode = '1' if bpc == '1' else PdfColorSpace.get_mode(cs)
            if mode not in ['1', 'L']:
                img = Image.fromarray(array, mode=mode)
            elif mode == 'L':
                img = Image.fromarray(array[...,0], mode='L')
            else:
                # A bug in PIL: https://stackoverflow.com/questions/32159076/python-pil-bitmap-png-from-array-with-mode-1
                img = Image.fromarray(array[...,0].astype('uint8')*255, mode='L').convert('1')
            array = None

        # ------------------------------------------------------------------------------------- Default color spaces

        if applyColorSpace:

            # Insert ICC profile
            img.info['icc_profile'] = icc_profile

            # Apply default page color space ICC profile, if present
            if img.info.get('icc_profile', None) == None:
                PdfColorSpace.apply_default_page_colorspace_icc_profile(pdfPage, img)

            # Remember icc_profile
            icc_profile = img.info.get('icc_profile', None)

        # ------------------------------------------------------------------------------------- Masks

        obj_mask = obj.Mask if obj.Mask != None else obj.SMask if obj.SMask != None else None
        if applyMasks and obj_mask != None:

            # print("Mask -->", obj_mask)
            # msg('Mask decoding started')
            mask, _ = PdfImage.decode(obj_mask)
            # msg('Mask decoding ended')

            if mask.mode == '1':
                mask = ImageChops.invert(mask).convert('L') # Black is 1 in Mask
            if mask.size != img.size:
                # msg("Resizing Mask")
                mask = mask.resize(img.size)

            if obj_mask.Matte != None: # Pre-multiplied alpha
                
                matte_color = tuple(round(255*float(x)) for x in obj_mask.Matte)
                # msg(f'Undoing alpha pre-blending with Matte color: {matte_color}')

                mode = img.mode
                color = np.array(img).astype('int32')

                cpp = color.shape[2]
                # if len(matte_color) > cpp: matte_color = matte_color[:cpp]
                matte = np.tile(np.array(matte_color,dtype='int32'), (height, width, 1))
                alpha = np.tile(np.array(mask)[...,None], cpp).astype('int32')
                alpha = np.clip(alpha, 1,255)

                # Un-multiply
                unmultiplied =  matte + ((color - matte) * 255) // alpha
                unmultiplied = np.clip(unmultiplied, 0, 255).astype('uint8')

                img = Image.fromarray(unmultiplied, mode=mode)

                # Restore icc_profile
                img.info['icc_profile'] = icc_profile

            if img.mode != 'RGB':
                intent = None if intent=='none' else obj.Intent if intent == 'native' else intent
                msg(f"Converting {img.mode} to sRGB with rendering intent: {intent}")
                img = ImageUtils.pil_image_to_srgb(img, intent)
                if 'icc_profile' in img.info: del img.info['icc_profile']

            # msg("Inserting alpha: RGB + Mask --> RGBA")
            img.putalpha(mask)
            img.format = 'PNG'

            encoded = None

        return img, encoded

    # -------------------------------------------------------------------- get_bpc()

    def get_bpc(obj:PdfDict):
        '''
        Returns bits per pixel
        '''
        if obj.Subtype != '/Image': raise ValueError(f"Not an image: {obj}")
        return 1 if obj.ImageMask == PdfObject('true') \
            else None if obj.Filter == PdfName.JPXDecode \
            else int(obj.BitsPerComponent)

    # -------------------------------------------------------------------- get_colorspace()

    def get_colorspace(obj:PdfDict):
        '''
        Returns obj.ColorSpace if it's set, or /DeviceGray if obj.ImageMask == true,
        or /JPX (a surrogate color space) if obj.Filter == /JPXDecode, or None otherwise.
        '''
        if obj.Subtype != '/Image': raise ValueError(f"Not an image: {obj}")
        return obj.ColorSpace if obj.ColorSpace != None \
            else '/DeviceGray' if obj.ImageMask == PdfObject('true') \
            else '/JPX' if obj.Filter == '/JPXDecode' else None

    # -------------------------------------------------------------------- inline_image_to_image_obj()

    def inline_image_to_image_obj(inline_img:list):
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

    # -------------------------------------------------------------------- inline_image_to_image_obj()

    def image_obj_to_inline_image_stream(obj:PdfObject):
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

    # -------------------------------------------------------------------- pil_image_to_pdf_page()

    def pil_image_to_pdf_page(image:Image.Image):
        '''Converts a PIL image to a PDF page: the page size is set automatically
        so that the image fills entire page at given resolution.
        The dpi, which is a tuple (xdpi, ydpi), overrides any dpi extracted from the input image.
        If no resolution is specified and none can be extracted from the input image, the
        value of (72,72) is used.

        The function returns a tuple (pdfPage, dpiActual), where
        dpiActual is the actual resolution used in placing the image on the page.
        

        During the transformation, the function is able to preserve transparency and color profiles.
        It achieves this by essentially doing all the steps in PdfImage.decode() in reverse order:
        * Encodes alpha channel as the SMask attribute of the PDF image XObject
        * Stores icc_profile in pdfPage.Resources.ColorSpace (the default color space profile
        for the image's color space)
        '''

        # Encode image
        xobj, dpi = PdfImage.encode(image)
        if dpi == None: dpi = (72,72)
        print(xobj)

        # Create page
        w,h = image.size
        w,h = w * 72/dpi[0], h*72/dpi[1]
        p = lambda x: round(x*1000000)/1000000
        q = lambda x: f'{p(x):f}'.rstrip('0').rstrip('.')
        page = IndirectPdfDict(
            Type = PdfName.Page,
            MediaBox = [0, 0, p(w), p(h)],
            Contents = IndirectPdfDict(stream=f'{q(w)} 0 0 {q(h)} 0 0 cm\n/Im1 Do\n'),
            Resources = PdfDict(XObject = PdfDict(Im1 = xobj))
        )

        return page, dpi

    # -------------------------------------------------------------------- size()

    def size(obj:PdfObject):
        '''
        Calculates the size of the PDF object as a sum of lengths of all dictionary streams that the object refers to.
        This means ignoring the so-called document overhead - the space taken by the dictionary header structures.
        This also means that PdfImage.size(obj) == 0 if obj is neither a PdfDict nor a PdfArray.
        '''
        if isinstance(obj, PdfDict):
            return sum(PdfImage.size(v) for v in obj.values()) + (int(obj.Length) if obj.Length != None else 0)
        elif isinstance(obj, PdfArray):
            return sum(PdfImage.size(v) for v in obj)
        else:
            return 0

    # -------------------------------------------------------------------- modify_image_xobject()

    def modify_image_xobject(image_obj:IndirectPdfDict, pdfPage:PdfDict, options:PdfDict):
        '''
        This function performs in-place modifications of an image object, such as: JPEG/JPEG2000
        compression, resizing/upsampling, color space conversions.
        '''

        obj_masks = [image_obj.Mask, image_obj.SMask]

        if options.predictors:
            msg('Checking images PNG predictors')
            _ = PdfFilter.uncompress(image_obj)
            for mask in obj_masks:
                if mask == None: continue
                msg('Checking mask PNG predictors')
                _ = PdfFilter.uncompress(mask)
            return

        # A lambda for printing out size changes in the form old_size --> new size (+-change%)
        print_size_change = lambda size_old, size_new: \
            f'Size: {size_old} --> {size_new} ({((size_new - size_old)*100)//size_old:+d}%)'

        # ColorSpace
        cs = PdfImage.get_colorspace(image_obj)
        bpc = PdfImage.get_bpc(image_obj)

        # Decode image
        applyColorSpace = options.colorspace == 'rgb' and PdfColorSpace.get_name(cs) != '/DeviceRGB' and bpc != 1 \
                            or options.gray and img.mode not in ['L', '1'] \
                            or options.bitonal and img.mode != '1'
        msg(f'Image decoding started, applyColorSpace = {applyColorSpace}')
        img, _ = PdfImage.decode(image_obj, pdfPage,
                                        invertCMYK = True,
                                        applyMasks = applyColorSpace,
                                        applyColorSpace = applyColorSpace,
                                        intent = options.intent)
        msg('Image decoding ended')

        if img == None: warn(f'Failed to decode image'); return


        # Force CMYK --> RGB, if requested (unless CMYK is actually /DeviceN)
        if options.colorspace == 'rgb' and img.mode not in ['RGB','RGBA','1']:
            intent = image_obj.Intent if options.intent == 'native' else options.intent
            msg(f'Forcing {img.mode} --> RGB with rendering intent: {intent}')
            img = ImageUtils.pil_image_to_srgb(img, intent)
            if 'icc_profile' in img.info: del img.info['icc_profile']

        # Convert to gray
        if options.gray and img.mode not in ['L', '1']:
            msg(f'Conversion {img.mode} --> L')
            img = img.convert('L')
            if 'icc_profile' in img.info: del img.info['icc_profile']

        # Convert to bitonal
        if options.bitonal and img.mode != '1':
            msg(f'Conversion {img.mode} --> 1')
            img = img.convert('1', dither=Image.Dither.NONE)
            if 'icc_profile' in img.info: del img.info['icc_profile']

        # Resizing
        if options.bicubic or options.upsample:
            f = options.bicubic
            if options.upsample:
                from simage.simage import SImage
            w,h = img.size

            if options.bicubic:
                msg(f'Resizing image by factor {f}')
                img = img.resize((int((w+0.5/f)*f), int((h+0.5/f)*f)), resample=Image.Resampling.BICUBIC)

            if options.upsample:
                msg(f'Upsampling image, alpha = {options.alpha}, bounds = {options.bounds}')
                array = np.array(img)
                array = SImage.scale2x(array) if SImage.isBitonal(array) else \
                    SImage.superResolution(array, alpha = options.alpha, bounds = options.bounds)

                # A bug in PIL: https://stackoverflow.com/questions/32159076/python-pil-bitmap-png-from-array-with-mode-1
                img = Image.fromarray(array, mode = img.mode) if img.mode != '1' \
                        else Image.fromarray(array.astype('uint8')*255, mode = 'L').convert('1')

            if not applyColorSpace:

                obj_masks_downsampled = []

                for obj_mask in obj_masks:
                    if obj_mask == None: obj_masks_downsampled.append(None); continue
                    msg('Mask decoding started')
                    mask, encoded = PdfImage.decode(obj_mask) # Should this be rendered with colorSpace applied?

                    msg('Mask decoding ended')
                    msg(f'Mask mode: {mask.mode}')

                    if options.bicubic:
                        msg(f'Resampling mask by factor {f}')
                        mask = mask.resize((int(round(w/f)), int(round(h/f))), resample=Image.Resampling.BICUBIC)

                    if options.upsample:
                        msg(f'Upsampling mask, alpha = {options.alpha}, bounds = {options.bounds}')
                        array = np.array(mask)
                        array,_ = SImage.scale2x(array) if SImage.isBitonal(array) else \
                            SImage.superResolution(array, alpha = options.alpha, bounds = options.bounds)
                        if mask.mode != '1':
                            mask = Image.fromarray(array, mode = mask.mode)
                        else:
                            # A bug in PIL: https://stackoverflow.com/questions/32159076/python-pil-bitmap-png-from-array-with-mode-1
                            mask = Image.fromarray(array.astype('uint8')*255, mode = 'L').convert('1')
                        
                    msg(f'Mask mode: {mask.mode}')
                    compressionFormat = 'JPEG' if options.jpeg and mask.mode != '1' else None
                    msg(f"Mask encoding started, compress = {compressionFormat or 'ZIP'}")
                    obj_mask_new,_ = PdfImage.encode(mask,
                                                 compressionFormat = compressionFormat,
                                                 compressedQuality = options.quality)
                    msg('Mask encoding ended')

                    obj_mask_new.Matte = obj_mask.Matte
                    obj_mask_new.ImageMask = obj_mask.ImageMask
                    obj_masks_downsampled.append(obj_mask_new)

                obj_masks = obj_masks_downsampled


        # Convert PIL image to a PDF Image XObject and compress it with jpegQuality (if it's != 'keep')
        compressionFormat = 'JPEG' if options.jpeg else 'ZIP' if options.zip else None
        msg(f"Image encoding started, format = {img.format}, compress = {compressionFormat or 'ZIP'}")
        image_obj_new,_ = PdfImage.encode(img,
                                            compressionFormat = compressionFormat,
                                            compressedQuality = options.quality)
        msg('Image encoding ended')

        # If applyColorSpace == False, put ColorSpace, Decode & masks back in
        if not applyColorSpace:
            if img.mode != 'CMYK' or img.format != 'JPEG': image_obj_new.Decode = image_obj.Decode
            image_obj_new.ColorSpace = image_obj.ColorSpace
            image_obj_new.Mask, image_obj_new.SMask = obj_masks
            image_obj_new.Intent = image_obj.Intent
            image_obj_new.Interpolate = image_obj.Interpolate

        
        # Preserve /Intent
        image_obj_new.Intent = image_obj.Intent # What should the intent be after intent has already been applied?

        # Preserve /ImageMask
        image_obj_new.ImageMask = image_obj.ImageMask

        # Calculate the before/after sizes and decide if we want to go ahead with the compression
        size_old, size_new = PdfImage.size(image_obj), PdfImage.size(image_obj_new)
        msg(print_size_change(size_old, size_new))
        if not (size_new * 10 < size_old * 9) \
            and not options.bicubic and not options.upsample and not options.colorspace and not options.zip:
                msg('Skipping') ; return

        # Copy the the result
        image_obj.clear()
        for k,v in image_obj_new.items():
            image_obj[k] = v
        image_obj.stream = image_obj_new.stream

# ========================================================================== class ImageUtils

class ImageUtils:

    def PIL_image_to_PNG_IDAT_chunk(image:Image):
        '''
        Returns a PDF image xobject stream encoded PNG filters (predictors) & ZLIB compression.
        '''

        idatChunks = []
        bs = BytesIO()
        image.save(bs, format="PNG", optimize = True)
        bs.seek(0)
        bs.read(8)

        while True:
            length = struct.unpack(b'!L', bs.read(4))[0]
            type = bs.read(4)
            data = bs.read(length)
            crc  = struct.unpack(b'!L', bs.read(4))[0]
            if crc != zlib.crc32(type + data): warn(f'crc error in chunk: {type}'); return None
            if type == b'IDAT': idatChunks.append(data)
            if type == b'IHDR':
                assert length == 13
                width, height, bpc, color_type, compression, filter, interlace = struct.unpack(b'!LLBBBBB', data)
                assert (width, height) == image.size
                assert bpc == 8
                assert color_type in [0, 2] # image.mode == 'RGB' or 'L'
                assert interlace == 0
            if type == b'IEND': break

        if len(idatChunks) == 0: warn(f'no IDAT chunks in PNG') ; return None
        return b''.join(chunk for chunk in idatChunks)

    def PIL_image_to_PDF_stream_with_TIFF_codecs(image:Image, compression = 'group4'):
        '''
        Returns a PDF image xobject stream encoded using one of TIFF's codecs: 'group4' or 'tiff_lzw'.
        '''
        if compression not in ['group4', 'tiff_lzw']:
            raise ValueError(f'unsupported compression method: {compression}')

        if compression == 'group4' and image.mode != '1':
            raise ValueError(f'cannot use group4 compression with image.mode: {image.mode}')
        
        if compression == 'tiff_lzw' and image.mode != 'RGB':
            raise ValueError(f'cannot use LZW compression with image.mode: {image.mode}')

        # Make sure Pillow produces single-strip TIFF; this works with Pillow versions < 8.3.0 or >=8.4.0  
        # See: https://github.com/python-pillow/Pillow/pull/5744
        # For Pillow versions from 8.3.0 to 8.3.9 see:
        # https://gitlab.mister-muffin.de/josch/img2pdf/commit/6eec05c11c7e1cb2f2ea21aa502ebd5f88c5828b
        # https://gitlab.mister-muffin.de/josch/img2pdf/issues/46

        TiffImagePlugin.STRIP_SIZE = 2 ** 31

        with BytesIO() as bs:
            image.save(bs, format="TIFF", compression=compression)
            bs.seek(0)
            img = Image.open(bs)

            # Read the TIFF tags to find the offset(s) and length of the compressed data strips.
            strip_offsets = img.tag_v2[TiffImagePlugin.STRIPOFFSETS]
            strip_bytes = img.tag_v2[TiffImagePlugin.STRIPBYTECOUNTS]
            rows_per_strip = img.tag_v2.get(TiffImagePlugin.ROWSPERSTRIP, 2 ** 32 - 1)
            if len(strip_offsets) != 1 or len(strip_bytes) != 1:
                err("Expected a single strip")
            (offset,), (length,) = strip_offsets, strip_bytes

            bs.seek(offset)
            stream = bs.read(length)

        return stream

    def PDF_stream_to_PIL_TIFF(width:int, height:int, bpc:int, cpp:int,
                               tiff_compression:int, tiff_photometric:int, stream:bytes,
                               encodedByteAlign = False,
                               predictor = 1
                               ):
        '''Wraps /CCITTFaxDecode PDF stream in the TIFF format and creates a PIL image out of it.
        '''
        if encodedByteAlign:
            warn(f'*** /EncodedByteAlign == true, this is in beta-testing, check results ***')
            assert tiff_compression == 3 # encodedByteAlign only possible with Group 3 compression
        tiff_header_struct = '<' + '2s' + 'H' + 'L' + 'H' + 'HHLL' * 11 + 'L'
        tiff_header = struct.pack(tiff_header_struct,
                        b'II',  # Byte order indication: Little indian
                        42,  # Version number (always 42)
                        8,  # Offset to first IFD
                        8,  # --- IFD starts here; the number of tags in IFD
                        256, 4, 1, width,  # ImageWidth, LONG, 1, width
                        257, 4, 1, height,  # ImageLength, LONG, 1, length
                        258, 3, 1, bpc,  # BitsPerSample, SHORT, 1, 1; this is the default, so omit?
                        259, 3, 1, tiff_compression,  # Compression, SHORT, 1, 3 = Group 3, 4 = Group 4, 8 - Deflate (ZIP)
                        262, 3, 1, tiff_photometric,  # Photometric interpretation, SHORT, 1, 0 = WhiteIsZero
                        273, 4, 1, struct.calcsize(tiff_header_struct),  # StripOffsets, LONG, 1, len of header
                        277, 3, 1, cpp,  # SamplesPerPixel, LONG, 1, component per pixel (cpp)
                        278, 4, 1, height,  # RowsPerStrip, LONG, 1, height
                        279, 4, 1, len(stream),  # StripByteCounts, LONG, 1, size of image
                        292, 4, 1, 4 if encodedByteAlign else 0,  # TIFFTAG_GROUP3OPTIONS, LONG, 1, 0..7 (flags)
                        317, 3, 1, predictor,  # Predictor, SHORT, 1, predictor = 1 (no prediction) or 2 (=left)
                        0  # --- IFD ends here; offset of the next IFD, 4 0-bytes b/c no next IFD
                        )
        return Image.open(BytesIO(tiff_header + stream))


    def pil_image_to_jpeg(img:Image, jpegQuality=95):
        '''Converts any PIL image to a JPEG-compressed PIL image with the given quality. Any transparency, if present, is lost.
        '''
        # if img.mode in ['LA','1']: img = img.convert('L')
        # if img.mode in ['RGBA','P']: img = img.convert('RGB')
        icc_profile = img.info.get('icc_profile')
        bs = BytesIO()
        img.save(bs, format='JPEG', quality=jpegQuality)
        bs.seek(0)
        img_new = Image.open(bs)
        img_new.info['icc_profile'] = icc_profile
        return img_new

    def pil_image_invert_cmyk_colors_if_needed(image:Image):
        '''CMYK (more precisely, YCCK) images are stored inverted in PDF if JPEG's adobe_transform tag is 2.
        This function checks the tag and inverts the image back if necessary.
        '''
        if image.mode == 'CMYK' and get_any_key(image.info, 'adobe_transform') == 2:
            msg("Fixing Adobe CMYK")
            image = ImageChops.invert(image)
        return image

    def pil_image_to_srgb(image:Image, intent = None):
        '''The correct way to convert CMYK to RGB using ImageCms.profileToProfile().
        The intent argument is the rendering intent. It can be one of:

        * '/Perceptual'
        * '/RelativeColorimetric'
        * '/Saturation'
        * '/AbsoluteColorimetric'.

        If it is None then '/Perceptual' is used.
        '''
        intents = {'/Perceptual': ImageCms.Intent.PERCEPTUAL,
                   '/RelativeColorimetric': ImageCms.Intent.RELATIVE_COLORIMETRIC,
                   '/Saturation': ImageCms.Intent.SATURATION,
                   '/AbsoluteColorimetric': ImageCms.Intent.ABSOLUTE_COLORIMETRIC }
        renderingIntent = intents.get(intent, ImageCms.Intent.PERCEPTUAL)
        icc_profile = image.info.get('icc_profile')
        if icc_profile == None and image.mode == 'CMYK':
            icc_profile = CMYK_DEFAULT_ICC_PROFILE
        if icc_profile != None:
            inputProfile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
            outputProfile = ImageCms.createProfile("sRGB")
            return ImageCms.profileToProfile(image, inputProfile, outputProfile, renderingIntent = renderingIntent, outputMode="RGB")
        else:
            warn('low quality conversion to RGB')
            return image.convert('RGB')

# ============================================================================= jbig2_compress()

def jbig2_compress(bitonal_images:dict):
    '''
    Compress bitonal images with the JBIG2 codec
    '''
    msg(f'processing {len(bitonal_images)} bitonal images')

    # Prepare the tmpDir
    tmpDir = '_jbig2'
    if not os.path.isdir(tmpDir):
        os.makedirs(tmpDir)
    os.chdir(tmpDir)

    images = list(bitonal_images.values())
    # Dump images to tmpDir
    tif_names = []
    for n, image in enumerate(images):
        pil_image, encoded = PdfImage.decode(image)
        if pil_image == None:
            sys.exit(f'failed to decode image: {image}')
        tif_name = f'{n:04d}.tif'
        pil_image.save(tif_name)
        tif_names.append(tif_name)

    result = subprocess.run(['jbig2', '-p', '-s', '-t', '0.97999'] + tif_names, capture_output=True)
    if result.returncode:
        print('ERROR: running jbig2 encoder failed:')
        print(result.stdout.decode('utf-8'))
        print(result.stderr.decode('utf-8'))
        sys.exit(1)

    try:
        globals = IndirectPdfDict(stream = py23_diffs.convert_load(open('output.sym','rb').read()))
    except:
        globals = None
    for n, image in enumerate(images):
        image.Filter = PdfName.JBIG2Decode
        image.DecodeParms = PdfDict(JBIG2Globals = globals) if globals else None
        image.stream = py23_diffs.convert_load(open(f'output.{n:04d}','rb').read())

    os.chdir('..')

    import shutil
    shutil.rmtree('_jbig2')

# ============================================================================= main()

if __name__ == '__main__':

    helpMessage='''
    pdfimage.py -- image library for pdfrw

    Usage:
    
    1) Help: pdfimage.py [-h]

    Images -> PDF:
    pdfimage.py [-o=output.pdf] [-dpi=dpi] image1.png [image2.jpg ..]
    
    Output: output.pdf | image1.pdf

    2) PDF -> images:
    pdfimage.py file.pdf
    
    Output: file.img1.png, file.img2.jpg ...

    3) PDF -> PDF (compression):
    pdfimage.py [-jpeg[=Q(90)]|jbig2] file.pdf
    
    Output: file-[jpeg|jbig2].pdf
    '''

    ap = argparse.ArgumentParser()

    ap.add_argument('inputPaths', nargs='+', metavar='FILE', help='input files: images or PDF')
    ap.add_argument('-output', '-o', type=str, metavar='PATH', help='output PDF file path')
    ap.add_argument('-first', '-f', type=int, metavar='N', default=1, help='first pageNo to process (def = 1)')
    ap.add_argument('-last', '-l', type=int, metavar='N', default=-1, help='last pageNo to process (def = last page)')
    ap.add_argument('-dpi', type=int, metavar='N', help='set resolution of input images to DPI')

    ap.add_argument('-bitonal', action='store_true', help='convert color/gray images to bitonal')
    ap.add_argument('-jbig2', action='store_true', help='compress bitonal images with JBIG2 (lossless)')
    ap.add_argument('-dict', type=int, metavar='N', help='pages per JBIG2 symbol dictionary')

    ap.add_argument('-gray', action='store_true', help='convert color images to grayscale')

    ap.add_argument('-zip', action='store_true', help='compress color/gray images with JPEG')
    ap.add_argument('-jpeg', action='store_true', help='compress color/gray images with JPEG')
    ap.add_argument('-quality', '-q', type=int, default=90, metavar='Q', help='JPEG compression quality; Q=0..100 (def=90)')
    ap.add_argument('-colorspace', '-cs', type=str, metavar='S', choices=['rgb'], help='convert images to color space; S = rgb (def=rgb)')

    ap.add_argument('-intent', type=str, metavar='I', choices = ['absolute', 'relative', 'perceptual', 'saturation', 'native', 'none'],
                    default = 'perceptual',
                    help='rendering intent; I = absolute|relative|perceptual|saturation|native|none (def=\'perceptual\')')

    ap.add_argument('-bicubic', type=float, metavar='F', help='resample color/gray images with bicubic interpolation')
    ap.add_argument('-upsample', action='store_true', help='upsample color/gray images x2 using Bayesian algorithm')
    ap.add_argument('-predictors', action='store_true', help='fixes incorrect PNG predictor values in images\' DecodeParms dicts')
    ap.add_argument('-alpha', type=int, metavar='A', default=1, help='alpha for the upsampling algo, higher=sharper; A=1..10, def=1')
    ap.add_argument('-bounds', type=str, metavar='B', default='local', choices=['softmax','local','none'], help='bounds for the upsampling algo, B=softmax|local|none, def=softmax')

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
        pdf = PdfReader(options.inputPaths[0])
        N = len(pdf.pages)
        eprint(f"[PAGES]: {N}")
        if options.last == -1: options.last = N

        # JBIG2 compression
        if options.jbig2:
            cache = set()
            bitonal_images = {}
            pageCount = 0
            for pageNo in range(options.first, options.last+1):
                print(f'[{pageNo}]')
                page = pdf.pages[pageNo-1]
                bitonal_images = bitonal_images | {id(obj):obj for name, obj in PdfObjects(page, cache=cache) 
                            if isinstance(obj, PdfDict) and obj.Subtype == PdfName.Image 
                            and (obj.BitsPerComponent == '1' or obj.ImageMask == PdfObject('true'))}
                pageCount += 1
                if pageCount == options.dict:
                    jbig2_compress(bitonal_images) 
                    pageCount = 0
                    bitonal_images = {}
            
            if len(bitonal_images) > 0:
                jbig2_compress(bitonal_images) 

            pdfOutPath = fileBase + f'-jbig2' + fileExt
            print(LINE_DOUBLE)
            print(f'Writing output to {pdfOutPath}')
            PdfWriter(pdfOutPath, trailer=pdf, compress=True).write()

            sys.exit()

        # Iterate over pages
        cache = set()
        for pageNo in range(options.first, options.last+1):

            page = pdf.pages[pageNo-1]
            images = {id(obj):obj for name, obj in PdfObjects(page, cache=cache) 
                        if isinstance(obj, PdfDict) and obj.Subtype == PdfName.Image 
                        and name not in [PdfName.Mask, PdfName.SMask]}

            # print(page)
            print(f'Page {pageNo}, read {len(images)} images')
            print(f'Page.Resources.ColorSpace:', page.Resources.ColorSpace if page.Resources != None else None)
            for n,idx in enumerate(images):
                obj = images[idx]
                print(LINE_SINGLE)
                print(f'[{n+1}] Image ID = {idx}')
                pprint(obj)
                print(LINE_SINGLE)

                # ---------- Modify images ----------
                if options.zip or options.jpeg or options.upsample or options.bicubic or options.colorspace \
                        or options.bitonal or options.predictors:
                    pageArg = page if options.colorspace else None
                    PdfImage.modify_image_xobject(obj, pageArg, options)
                    print(LINE_SINGLE)
                    print("RESULT:")
                    pprint(obj)

                # ---------- Extract images ----------
                else:

                    # image = PdfImage.decode(obj,page, adjustColors = False, applyMasks = True, applyIntent = options.applyIntent)
                    image, encoded = PdfImage.decode(obj,page, invertCMYK = True,
                                                        applyMasks = True, applyColorSpace = True, intent = options.intent)
                    if image == None: warn('failed to extract image; continuing'); continue
                    if encoded != None:
                        format, stream = encoded
                        if format == 'JPEG':
                            ext = '.jpg'
                        elif format == 'JPEG2000':
                            ext = '.jp2'
                        else:
                            sys.exit(f'unknown encoded format {format}')
                        outputPath = fileBase  +f'.page{pageNo}.image{n+1}' + ext
                        print(f'Extracting original image stream --> {outputPath}')
                        open(outputPath,'wb').write(stream)
                        continue

                    dpi = options.dpi if options.dpi != None else image.info.get('dpi') if image.info.get('dpi') != None else (72,72)
                    icc_profile = image.info.get('icc_profile')

                    # Save images
                    formats = {'TIFF':'.tif', 'PNG':'.png', 'JPEG':'.jpg', 'JPEG2000':'.jp2'}
                    print('Saving image.format:', image.format)
                    ext = formats[image.format] if image.format in formats else '.tif'
                    tiff_compression = 'group4' if image.mode == '1' else 'tiff_lzw'
                    outputPath = fileBase  +f'.page{pageNo}.image{n+1}' + ext
                    print(f'Extracting --> {outputPath}')
                    if image.format in ['PNG', 'JPEG', 'JPEG2000']:
                        image.save(outputPath, dpi=dpi, icc_profile=icc_profile, quality='keep', info=image.info)
                    else:
                        image.save(outputPath, dpi=dpi, icc_profile=icc_profile, compression=tiff_compression)

        if options.zip or options.jpeg or options.upsample or options.bicubic or options.colorspace \
                or options.bitonal or options.predictors:
            suffix = ''
            if options.upsample: suffix += f'-upsample-{options.alpha}'
            if options.bicubic: suffix += '-bicubic'
            if options.colorspace: suffix += f'-{options.colorspace}'
            if options.zip: suffix += f'-zip'
            if options.jpeg: suffix += f'-jpeg-{options.quality}'
            if options.bitonal: suffix += '-bitonal'
            if options.predictors: suffix += '-predictors'
            assert suffix != ''
            pdfOutPath = fileBase + suffix + fileExt
            print(LINE_DOUBLE)
            print(f'Writing output to {pdfOutPath}')
            PdfWriter(pdfOutPath, trailer=pdf, compress=True).write()

    # ---------- Input is images: convert to PDF ----------
    else:

        pdfPath = options.output or options.inputPaths[0]+".pdf"

        pdf = PdfWriter(pdfPath,compress=True)
        N = len(options.inputPaths)
        for n,imagePath in enumerate(options.inputPaths):
            image = Image.open(imagePath)
            dpi_orig = image.info.get('dpi')
            print(f'[{n+1}/{N}] {image.size} {dpi_orig} {image.format} {image.mode} {imagePath}')
            dpi = (options.dpi, options.dpi) if options.dpi != None else image.info.get('dpi') if image.info.get('dpi') != None else (72,72)
            image.info['dpi'] = dpi
            page, dpi_actual = PdfImage.pil_image_to_pdf_page(image)
            pdf.addPage(page)
            print(f'[{n+1}/{N}] {image.mode} {image.size} @ {dpi_actual} dpi')

        pdf.write()
        print(f'Output written to {pdfPath}')
