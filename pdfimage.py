#!/usr/bin/env python3

# Write this man about the bug in PIL Image.paste()!
# https://sudonull.com/post/129265-Alpha_composite-optimization-history-in-Pillow-20
# https://habr.com/ru/post/98743/
# https://github.com/homm/
# https://twitter.com/wouldntfix

import struct, zlib, base64, sys
from io import BytesIO
import numpy as np

from PIL import Image, TiffImagePlugin, ImageChops, ImageCms
Image.MAX_IMAGE_PIXELS = None

from pdfrw import PdfObject, PdfName, PdfArray, PdfDict, IndirectPdfDict, py23_diffs
from pdfrwx.common import err, msg, warn, eprint, get_key, get_any_key
from pdfrwx.pdffilter import PdfFilter

from pprint import pprint

# ========================================================================== class PdfColorSpace

class PdfColorSpace(PdfDict):

    def __init__(self, cs = None, **kwargs):
        '''A utility class for parsing PDF color spaces.

        Usage:
        ```
        cs = PdfColorSpace(mode = 'mode', cpp = cpp...)
        or
        cs = PdfColorSpace(colorSpace),
        ```
        where colorSpace is either a ```PdfName``` or ```PdfArray```.
        The second form parses colorSpace and sets the following attributes:
        ```
        self.name:str
        self.mode:str
        self.cpp:int
        self.icc_profile:bytes
        self.palette:bytes
        self.palette_cs:PdfColorSpace
        ```
        The first three are always set, the rest are set depending on the particular color space.

        '''
        super().__init__(**kwargs)
        if cs == None: return

        # The name
        self.name = cs[0] if isinstance(cs,PdfArray) else cs

        # A map from color self.name to (self.mode, self.cpp)
        map = {
            # Simple color spaces
            '/DeviceGray': ['L',1], '/CalGray': ['L',1],
            '/DeviceRGB': ['RGB',3], '/CalRGB': ['RGB',3],
            '/DeviceCMYK': ['CMYK',4],
            '/Lab': ['LAB',3],
            # Complex color spaces
            '/Indexed': ['P',1],
            '/ICCBased': [None,None],
            '/Separation': ['SEP',1], '/DeviceN': ['DEVN',None]
        }

        if self.name not in map:
            raise ValueError(f'invalid colorspace: {cs}')
        self.mode, self.cpp = map[self.name]

        if self.name in ['/CalGray','/CalRGB', '/Lab']:
            self.icc_profile = PdfColorSpace.CreateProfile(cs)

        if self.name == '/Indexed':

            self.name, base, hival, pal = cs

            self.palette_cs = PdfColorSpace(base)

            size = (int(hival) + 1) * self.palette_cs.cpp
            # pal is either a PdfDict, with palette in its stream, or a PdfString
            pal = py23_diffs.convert_store(PdfFilter.uncompress(pal).stream) if isinstance(pal,PdfDict) else pal.to_bytes()
            while len(pal) > size and pal[-1] in b'\n\r':
                pal = pal[:-1]
            if len(pal) != size:
                raise ValueError(f'palette size mismatch: expected {size}, got {len(pal)}')

            self.palette = pal

        if self.name == '/ICCBased':
            self.cpp = int(cs[1].N)
            self.icc_profile = py23_diffs.convert_store(PdfFilter.uncompress(cs[1]).stream)
            cpp2mode = {1:'L', 3:'RGB', 4:'CMYK'}
            self.mode = cpp2mode[self.cpp] if self.cpp in cpp2mode else 'ICC'

        if self.name == '/DeviceN':
            self.cpp = len(cs[1])


    def CreateProfile(cs:PdfArray):
        '''Creates an ICC profile for a given color space, when it's one of ['/CalGray', ..], ['/CalRGB', ..], ['/Lab', ..]
        '''
        if not isinstance(cs,PdfArray): return None
        name,val = cs
        if name not in ['/CalGray','/CalRGB', 'Lab'] or val.WhitePoint == None: return None

        # Calculate the CIR xyY values of the white point
        X,Y,Z = [float(v) for v in val.WhitePoint]
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
                gamma = float(val.Gamma) if val.Gamma != None else 1
                transferFunction = lc.cmsBuildGamma(ctx,gamma)
                profile = lc.cmsCreateGrayProfile(white, transferFunction)
            if name == '/CalRGB':
                primaries = [float(v) for v in val.Matrix] if val.Matrix != None else [1,0,0,0,1,0,0,0,1]
                primaries = [primaries[3*i,3*i+3] for i in range(3)]
                primaries = [[X/(X+Y+Z),Y/(X+Y+Z),Y] for X,Y,Z in primaries]
                gamma = [float(v) for v in val.Gamma] if val.Gamma != None else [1,1,1]
                transferFunction = [lc.cmsBuildGamma(ctx, g) for g in gamma]
                profile = lc.cmsCreateRGBProfile(white, primaries, transferFunction)
            with BytesIO() as bs:
                lc.cmsSaveProfileToStream(profile, bs)
                icc_profile = bs.getvalue()
            msg(f'LittleCMS: processed {cs}')
        except:
            # ImageCms needs correlated temperature to create a Lab profile; use the (approximate) McCamy's formula
            n = (x - 0.3320) / (0.1858 - y)
            CCT = 449 * n*n*n + 3525 * n*n + 6823.3 * n + 5520.33
            if name == '/Lab':
                icc_profile = ImageCms.createProfile('LAB', colorTemp = CCT)
            if name == '/CalRGB':
                return None
            msg(f'ImageCms: processed {cs}')

        if val.BlackPoint != None: warn(f'/BlackPoint ignored: {cs}')
        if val.Range != None: warn(f'/Range ignored: {cs}')

        return icc_profile

# ========================================================================== class PdfImage

class PdfImage:

    # -------------------------------------------------------------------- encode()

    def encode(image:Image, quality = 'keep'):
        '''
        Converts a PIL image to a PDF image XObject.
        Returns a tuple (xobj, dpi), where xobj is the created PDF image XObject
        and dpi is a tuple (xdpi, ydpi) of the image's resolution. Processing depends on image.mode/format:
        
        * JPEG (image.format is 'JPEG') images are stored "as-is" using the /DCTDecode filter (no transcoding is done)

        * Bitonal (image.mode is '1') images are compressed using the /CCITTFaxDecode filter with Group4 encoding.

        * All other images are compressed using the /FlateDecode filter.

        For images with transparency, the alpha-channel is stored xobj.SMask.
        For palette-based/indexed (image.mode == 'P') images, the palette is used to create an /Indexed color space.
        The ICC color profile (image.info['icc_profile']), if present, is used to set up the corresponding
        /ICCBased color space.
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
            xobj, dpi = PdfImage.encode(image, quality)
            # msg('Image encoding ended')
            if 'icc_profile' in alpha.info: del alpha.info['icc_profile'] # alpha should not have icc_profile
            # msg('Mask encoding started')
            xobj_alpha, _ = PdfImage.encode(alpha, quality)
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
            stream = ImageUtils.PIL_TIFF_to_PDF_stream(image)
            filter = 'CCITTFaxDecode'
            # ImageMask = PdfObject('true')
            DecodeParms = PdfDict(K = -1, Columns = w, Rows = h, BlackIs1 = PdfObject('true'))
            # DecodeParms = PdfDict(K = -1, Columns = w, Rows = h)

        else: # PNG and others

            # msg('Color --> /FlateDecode')
            # ToDo: a) use optional predictors; b) feed the PNG IDAT chunk without re-encoding
            stream = zlib.compress(image.tobytes())
            filter = 'FlateDecode'

        # Re-compress streams as JPEG if requested; do not recompress palette-based images
        if quality != 'keep' and filter in ['FlateDecode', 'DCTDecode'] and image.mode != 'P':
            with BytesIO() as bs:
                image.save(bs, format='JPEG', quality=quality)
                stream_compressed = bs.getvalue()
                if stream[-2:] != b'\xff\xd9': stream += b'\xff\xd9' # fix a bug found in some jpeg files
            size, size_compressed = len(stream), len(stream_compressed)
            # size gains of < 10% do not justify loss of quality in re-compression
            if size_compressed * 10 < size * 9:
                # msg(f'Re-compressed stream as JPEG with quality = {quality}; ' + print_size_change(size, size_compressed))
                stream = stream_compressed
                filter = 'DCTDecode'
            else:
                pass
                # msg('JPEG re-compressions skipped; ' + print_size_change(size, size_compressed))

        # PDF expects inverted CMYK JPEGs
        if image.mode == 'CMYK' and filter == 'DCTDecode':
            Decode = [1,0,1,0,1,0,1,0]
            msg(f'CMYK JPEG; inserting /Decode = {Decode}')

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

    def decode(obj:IndirectPdfDict, pdfPage:PdfDict = None, adjustColors = True, applyMasks = True, applyIntent = False):
        '''Returns a PIL Image if obj is an image, fails otherwise.

        If applyMasks is True, the soft mask (SMask) of the image, if present, becomes the alpha-channel of the
        resulting image. At this stage, the conversion from the original non-RGB image.mode to
        RGB occurs, which becomes RGBA after the insertion of the alpha-mask. The conversion
        to RGB utilizes image's icc_profile.

        if fixAdobeCMYK is True, the function checks if the image is an Adobe CMYK image
        and, if so, fixes it by calling ImageUtils.pil_image_invert_cmyk_colors_if_needed(). 

        [EXTEND THE DESCRIPTION]
        '''

        assert obj.Subtype == '/Image'

        if obj.stream == None: return None

        # Remove compression for all filters except the image-specific ones
        obj = PdfFilter.uncompress(obj)

        # pprint(obj)

        stream = py23_diffs.convert_store(obj.stream) if isinstance(obj.stream,str) else obj.stream

        filter, parm = obj.Filter, obj.DecodeParms

        width, height = int(obj.Width), int(obj.Height)
        bpc, cs = PdfImage.get_image_specs(obj)
        msg(f'image specs: {bpc}, {cs}')

        img = None

        if filter == None:

            pass

        elif filter in ['/CCITTFaxDecode', '/CCF']: # --> TIFF

            # msg('/CCITTFaxDecode --> TIFF')
            if get_key(parm, '/EncodedByteAlign') == 'true':
                warn(f'/EncodedByteAlign == true is not supported')
                return None
            K = get_key(parm, '/K', '0')
            BlackIs1 = get_key(parm, '/BlackIs1', 'false')
            tiff_compression = 3 if int(K) >= 0 else 4
            tiff_photometric = 1 if BlackIs1 == 'true' else 0           # WHY IS IT BACKWARDS???
            img = ImageUtils.PDF_stream_to_PIL_TIFF(width, height, 1, 1, tiff_compression, tiff_photometric, stream)

        # def JBIG2Decode(stream):
        # # JBIG2 decoding can be done by system-calling jbig2dec; but let's wait for JBIG2 support in PIL instead
        #     if filter == '/JBIG2Decode':
        #         header = b'\x97\x4A\x42\x32\x0D\x0A\x1A\x0A\x01\x00\x00\x00\x01' # 1-page, sequential order; see JBIG2 Specs
        #         return Image.open(BytesIO(header + data))

        elif filter in ['/DCTDecode', '/DCT']: # --> JPEG

            # msg('/DCTDecode --> JPEG')
            if (ct := get_any_key(parm, '/ColorTransform')) is not None: warn(f'/DecodeParms: /ColorTransform {ct} ignored')
            if (ct := get_any_key(parm, '/Decode')) is not None: warn(f'/DecodeParms: /Decode {ct} ignored')
            img = Image.open(BytesIO(stream))

        elif filter == '/JPXDecode': # --> JPEG 2000

            # msg('/JPXDecode --> JPEG2000')
            warn(f'/JPXDecode implementation is buggy at the moment, decoding is not attempted')
            # img = Image.open(BytesIO(stream))
            return None

        else:
            warn(f'unsupported stream filter: {filter}')
            return None

        # -------------------------------------------------------------------------------------

        # RAW image streams
        indexedDecodeChecked = False
        if img == None:

            # Fix a bug with some of the /Deflate-encoded streams ending with \n
            if len(stream) == width * height * cs.cpp * bpc / 8 + 1 and stream[-1] == b'\n':
                stream = stream[:-1]
                warn(f'shortening stream by 1 byte to {len(stream)}')


            # PIL has poor or no support for bpc != 8 so unpack these to bpc == 8
            # Here, array.dtype is 'uint8' or 'uint16'
            array = PdfFilter.unpack_pixels(stream, width, height, cs.cpp, bpc)
            
            if cs.mode == 'P':
                # /Indexed mode
                if bpc == 16: warn('16-bit palette color indices are not supported'); return None
                else: array = array >> (8-bpc) # reverse the up-scaling done by unpack_pixels() above

                if obj.Decode != None:
                    d1, d2 = [int(float(x)) for x in obj.Decode]
                    iMax = 2 ** bpc - 1
                    if [d1,d2] != [0,iMax]:
                        # msg(f'applying /Decode: {obj.Decode} in /Indexed mode')
                        array = d1 + (array * (d2 - d1)) // iMax
                    indexedDecodeChecked = True

                stream = array.tobytes()

                # CMYK palettes are reduced as PIL doesn't support them
                if cs.palette_cs.mode == 'CMYK':
                    # msg('reducing /Indexed CMYK --> /DeviceCMYK')
                    palette_map = dict([(i//4, cs.palette[i:i+4]) for i in range(0,len(cs.palette),4)])
                    stream = b''.join(palette_map[i] for i in stream)
                    cs = PdfColorSpace('/DeviceCMYK')

            else:
                # All other modes
                if bpc == 16:
                    warn('16 --> 8 image bit depth reduction')
                    array = (array // 256).astype('uint8')
                    bpc = 8
                if bpc != 1:
                    stream = array.tobytes()

            if cs.mode not in ['1','L','P','RGB','CMYK']:
                warn(f'unsupported color space: {cs.name}, mode: {cs.mode}')
                return None
            img = Image.frombytes(cs.mode, (width,height), stream, 'raw')

        # Insert palette
        if cs.palette != None:
            # img.putpalette(ImagePalette.ImagePalette('RGB', palette))
            img.putpalette(cs.palette)

        # Adjust colors: process the /Decode attribute and CMYK JPEGs (need to be processed together)
        if adjustColors and not indexedDecodeChecked \
                and cs.name not in ['/Separation', '/DeviceN', None] \
                and cs.palette == None:

            decode = [float(x) for x in obj.Decode] if obj.Decode != None else None

            # This assumes that /DCTDecode in combination with 4-component /ICCBased or /DeviceN
            # color spaces uses inverted CMYK colors, just like with /DeviceCMYK; probably so;
            # For all such color spaces, cs.mode will be 'CMYK'
            cmyk_jpeg = img.mode == 'CMYK' and img.format == 'JPEG'

            if cs.palette != None: err('cs.palette != None; this case needs more testing')
            iMax = 2 ** bpc - 1
            vMax = iMax if cs.palette != None else 1
            # print("DECODE:", decode, cs)
            decode_is_trivial = decode == None or decode == [c for _ in range(len(decode)//2) for c in (0,vMax)]
            decode_is_inverted = decode != None and decode == [c for _ in range(len(decode)//2) for c in (vMax,0)]

            # print("CHECKS:", decode_is_trivial, decode_is_inverted)

            do_nothing = decode_is_trivial and not cmyk_jpeg or decode_is_inverted and cmyk_jpeg

            if not do_nothing:
                if cmyk_jpeg:
                    # msg('inverting CMYK JPEG')
                    img = ImageChops.invert(img)
                if not decode_is_trivial:
                    # msg(f'applying /Decode {decode} ')
                    if bpc == 1:
                        img = ImageChops.invert(img)
                    else:
                        components = img.split()
                        components_new = []
                        for n,c in enumerate(components):
                            d1,d2 = decode[n*2: n*2 + 2]
                            a = np.array(c).astype('float')
                            a = (iMax * d1 + a * (d2 - d1))/vMax
                            a = np.clip(a, 0, 255).astype('uint8')
                            components_new.append(Image.fromarray(a, mode='L'))
                        img = Image.merge(img.mode, components_new)

        # icc_profile from default color spaces
        default_icc_profile = None
        if pdfPage != None:
            try:
                default_cs_names = {'L':'/DefaultGray', 'RGB':'/DefaultRGB', 'CMYK':'/DefaultCMYK'}
                default_cs_name = default_cs_names[img.mode]
                default_cs = pdfPage.Resources.ColorSpace[default_cs_name]
                default_icc_profile = PdfImage.decode(default_cs[1])
            except:
                pass

        # Insert icc_profile. Clear the profile first (!) as there can be bogus profiles in the image data streams
        img.info['icc_profile'] = None
        if cs.icc_profile != None:
            # msg("inserting icc_profile from the /ICCBased color space")
            img.info['icc_profile'] = cs.icc_profile
        elif cs.palette_cs != None and cs.palette_cs.icc_profile != None:
            # msg("inserting icc_profile from the /Indexed /ICCBased color space")
            img.info['icc_profile'] = cs.palette_cs.icc_profile
        elif default_icc_profile != None:
            # msg(f"inserting icc_profile from the page's default color space: {default_cs_name}")
            img.info['icc_profile'] = default_icc_profile

        # Remember icc_profile
        icc_profile = img.info.get('icc_profile')

        # Masks
        obj_mask = obj.Mask if obj.Mask != None else obj.SMask if obj.SMask != None else None
        if applyMasks and obj_mask != None:

            # print("Mask -->", obj_mask)
            # msg('Mask decoding started')
            mask = PdfImage.decode(obj_mask)
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
                intent = obj.Intent if applyIntent else None
                # msg(f"Converting {img.mode} to sRGB with rendering intent: {intent}")
                img = ImageUtils.pil_image_to_srgb(img, intent)
                if 'icc_profile' in img.info: del img.info['icc_profile']

            # msg("Inserting alpha: RGB + Mask --> RGBA")
            img.putalpha(mask)
            img.format = 'PNG'

        return img

    # -------------------------------------------------------------------- get_image_specs()

    def get_image_specs(obj:PdfDict):
        '''For a PDF image XObject, returns a tuple (BitsPerComponent:int, ColorSpace:PdfColorSpace).
        '''
        if obj.Subtype != '/Image':
            raise ValueError(f"Not an image: {obj}")

        if obj.ImageMask == PdfObject('true'):
            return 1, PdfColorSpace(mode = '1', cpp = 1)

        if obj.Filter == PdfName.JPXDecode:
            return None, PdfColorSpace(mode = 'JPX')
        
        bpc = int(obj.BitsPerComponent)

        # This is wrong if the CS is /Indexed (yes, there are indexed bitonal images)
        # if bpc == 1:
        #     return 1, PdfColorSpace(mode = '1', cpp = 1)        

        return bpc, PdfColorSpace(obj.ColorSpace)

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

    def pil_image_to_pdf_page(image:Image):
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

    # -------------------------------------------------------------------- compress_image_xobject()

    def compress_image_xobject(image_obj:IndirectPdfDict, pdfPage:PdfDict, options:PdfDict):

        # A lambda for printing out size changes in the form old_size --> new size (+-change%)
        print_size_change = lambda size_old, size_new: \
            f'Size: {size_old} --> {size_new} ({((size_new - size_old)*100)//size_old:+d}%)'

        if options.forceRGB:
            # This produces RGBA if there are masks
            msg('Processing image and masks together as an image with transparency')
            msg('Image decoding started')
            img = PdfImage.decode(image_obj, pdfPage, adjustColors = True, applyMasks = True, applyIntent = options.applyIntent)
            msg('Image decoding ended')
        else:
            # Decode the image and store the masks (if any) separately
            msg('Processing image and masks separately')
            msg('Image decoding started')
            img = PdfImage.decode(image_obj, pdfPage, adjustColors = True, applyMasks = False, applyIntent = options.applyIntent)
            msg('Image decoding ended')

        if img == None: warn(f'Failed to decode image'); return
        cs = PdfColorSpace(image_obj.ColorSpace)

        obj_masks = [image_obj.Mask, image_obj.SMask]

        # Force CMYK --> RGB, if requested (unless CMYK is actually /DeviceN)
        if options.forceRGB and img.mode == 'CMYK' and cs.name != '/DeviceN':
            intent = image_obj.Intent if options.applyIntent else None
            msg(f'Forcing {img.mode} --> RGB with rendering intent: {intent}')
            img = ImageUtils.pil_image_to_srgb(img, intent)
            if 'icc_profile' in img.info: del img.info['icc_profile']

        # if options.gray and img.mode not in ['L', '1'] and cs.name not in ['/Separation','/DeviceN']:
        if options.gray and img.mode not in ['L', '1']:
            msg(f'Conversion {img.mode} --> L')
            img = img.convert('L')
            if 'icc_profile' in img.info: del img.info['icc_profile']

        # Resizing
        if options.resize:
            f = options.resize
            msg(f'Resizing image by factor {f}')
            w,h = img.size
            img = img.resize((int((w+0.5/f)*f), int((h+0.5/f)*f)))
            if not options.forceRGB:
                obj_masks_downsampled = []
                for obj_mask in obj_masks:
                    if obj_mask == None: obj_masks_downsampled.append(None); continue
                    msg('Mask decoding started')
                    mask = PdfImage.decode(obj_mask)
                    msg('Mask decoding ended')
                    msg(f'Mask mode: {mask.mode}')
                    msg(f'Downsampling mask')
                    mask = mask.resize(((w+1)//2, (h+1)//2))
                    msg(f'Mask mode: {mask.mode}')
                    Matte = obj_mask.Matte
                    msg('Mask encoding started')
                    obj_mask,_ = PdfImage.encode(mask, options.compressionQuality if mask.mode == 'L' else 'keep')
                    msg('Mask encoding ended')
                    obj_mask.Matte = Matte
                    obj_masks_downsampled.append(obj_mask)
                obj_masks = obj_masks_downsampled

        # if options.bitonal and img.mode != '1' and cs.name not in ['/Separation','/DeviceN']:
        if options.bitonal and img.mode != '1':
            msg(f'Conversion {img.mode} --> 1')
            img = img.convert('1', dither=Image.Dither.NONE)
            if 'icc_profile' in img.info: del img.info['icc_profile']


        # Convert PIL image to a PDF Image XObject and compress it with jpegQuality (if it's != 'keep')
        msg('Image encoding started')
        image_obj_new,_ = PdfImage.encode(img, options.compressionQuality)
        msg('Image encoding ended')

        # Correct /ColorSpace and /Decode for /DeviceN and /Separation color spaces
        # These were decoded as RGB, CMYK or L and so will produce an incorrect color space in the step above
        if cs.name in ['/Separation','/DeviceN']:
            image_obj_new.ColorSpace = image_obj.ColorSpace
            image_obj_new.Decode = image_obj.Decode

        # Put masks back in
        if not options.forceRGB:
            image_obj_new.Mask, image_obj_new.SMask = obj_masks
        
        # Preserve /Intent
        image_obj_new.Intent = image_obj.Intent

        # Preserve /ImageMask
        image_obj_new.ImageMask = image_obj.ImageMask

        # Calculate the before/after sizes and decide if we want to go ahead with the compression
        size_old, size_new = PdfImage.size(image_obj), PdfImage.size(image_obj_new)
        msg(print_size_change(size_old, size_new))
        if not (size_new * 10 < size_old * 9): msg('Skipping') ; return

        # Copy the the result
        image_obj.clear()
        for k,v in image_obj_new.items():
            image_obj[k] = v
        image_obj.stream = image_obj_new.stream



# ========================================================================== class ImageUtils

class ImageUtils:

    def PIL_TIFF_to_PDF_stream(image:Image):
        '''
        Convert a bitonal TIFF PIL Image (image.mode == '1') to a /CCITTFaxDecode-encoded PDF stream
        '''

        # Make sure Pillow produces single-strip TIFF; this works with Pillow versions < 8.3.0 or >=8.4.0  
        # See: https://github.com/python-pillow/Pillow/pull/5744
        # For Pillow versions from 8.3.0 to 8.3.9 see:
        # https://gitlab.mister-muffin.de/josch/img2pdf/commit/6eec05c11c7e1cb2f2ea21aa502ebd5f88c5828b
        # https://gitlab.mister-muffin.de/josch/img2pdf/issues/46

        TiffImagePlugin.STRIP_SIZE = 2 ** 31

        with BytesIO() as bs:
            image.save(bs, format="TIFF", compression="group4")
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

    def PDF_stream_to_PIL_TIFF(width:int, height:int, bpc:int, cpp:int, tiff_compression:int, tiff_photometric:int, stream:bytes, predictor = 1):
        '''Wraps /CCITTFaxDecode PDF stream in the TIFF format and creates a PIL image out of it.
        '''
        tiff_header_struct = '<' + '2s' + 'H' + 'L' + 'H' + 'HHLL' * 10 + 'L'
        tiff_header = struct.pack(tiff_header_struct,
                        b'II',  # Byte order indication: Little indian
                        42,  # Version number (always 42)
                        8,  # Offset to first IFD
                        8,  # --- IFD starts here; the number of tags in IFD
                        256, 4, 1, width,  # ImageWidth, LONG, 1, width
                        257, 4, 1, height,  # ImageLength, LONG, 1, length
                        258, 3, 1, bpc,  # BitsPerSample, SHORT, 1, 1; this is the default, so omit?
                        259, 3, 1, tiff_compression,  # Compression, SHORT, 1, 4 = CCITT Group 4 fax encoding, 8 - Deflate (ZIP)
                        262, 3, 1, tiff_photometric,  # Photometric interpretation, SHORT, 1, 0 = WhiteIsZero
                        273, 4, 1, struct.calcsize(tiff_header_struct),  # StripOffsets, LONG, 1, len of header
                        277, 3, 1, cpp,  # SamplesPerPixel, LONG, 1, component per pixel (cpp)
                        278, 4, 1, height,  # RowsPerStrip, LONG, 1, height
                        279, 4, 1, len(stream),  # StripByteCounts, LONG, 1, size of image
                        258, 3, 1, predictor,  # Predictor, SHORT, 1, predictor = 1 (no prediction) or 2 (=left)
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
        intents = {'/Perceptual': 0, '/RelativeColorimetric': 1, '/Saturation': 2, '/AbsoluteColorimetric': 3 }
        renderingIntent = intents[intent] if intent in intents else 0
        icc_profile = image.info.get('icc_profile')
        if icc_profile != None:
            inputProfile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
            outputProfile = ImageCms.createProfile("sRGB")
            return ImageCms.profileToProfile(image, inputProfile, outputProfile, renderingIntent = renderingIntent, outputMode="RGB")
        else:
            warn('low quality conversion to RGB')
            return image.convert('RGB')

# ============================================================================= main()

if __name__ == '__main__':

    helpMessage='''
    pdfimage.py -- image library for pdfrw

    Usage:
    
    Help: pdfimage.py [-h]

    Images -> PDF: pdfimage.py [-o=output.pdf] [-dpi=dpi] image1.png [image2.jpg ..]
    Output: output.pdf | image1.pdf

    PDF -> images: pdfimage.py file.pdf
    Output: file.img1.png, file.img2.jpg ...
    '''

    pdfPath = None
    imagePaths = []
    dpi = None

    options = PdfDict(
        dpi = None,
        compress = False,
        compressionQuality = 90,
        resize = None,
        gray = False,
        bitonal = False,
        forceRGB = False,
        applyIntent = False
    )

    firstPage,lastPage = 1,-1

    LINE_SINGLE = '-'*64
    LINE_DOUBLE = '='*64

    import re,os
    from pdfrw import PdfReader,PdfWriter
    from pdfobjects import PdfObjects

    if len(sys.argv) == 1: print(helpMessage); sys.exit() 
    for arg in sys.argv[1:]:
        if arg[0] == '-':
            key,value = (arg, None) if re.search('=',arg) == None else re.split('=',arg)
            if key == '-h': print(helpMessage); sys.exit()
            if key == '-o': pdfPath = value
            if key == '-f': firstPage = int(value)
            if key == '-l': lastPage = int(value)
            if key == '-compress':
                options.compress = True
                options.compressionQuality='keep' if value == 'keep' else int(value) if value != None else 90
            if key == '-resize':
                options.resize = float(value)
            if key == '-gray':
                options.gray = True
            if key == '-bitonal':
                options.bitonal = True
            if key == '-rgb':
                options.forceRGB = True
            if key == '-intent':
                options.applyIntent = True
            if key == '-dpi':
                try: options.dpi = float(value); dpi = (options.dpi,options.dpi)
                except: err(f'invalid dpi: {dpi}')
        else:
            imagePaths.append(arg)

    if len(imagePaths) == 0: err("No PDF or images files given, nothing to do")

    fileBase, fileExt = os.path.splitext(imagePaths[0])

    # ---------- Input is PDF: compress and extract images ----------
    if fileExt.lower() == '.pdf':

        pdf = PdfReader(imagePaths[0])
        N = len(pdf.pages)
        eprint(f"[PAGES]: {N}")
        if lastPage == -1: lastPage = N

        # Iterate over pages
        for pageNo in range(firstPage,lastPage+1):

            page = pdf.pages[pageNo-1]
            images = PdfObjects()
            images.read(page, PdfObjects.imageFilter)

            print(f'Page {pageNo}, read {len(images)} images')
            print(f'Page.Resources.ColorSpace:', page.Resources.ColorSpace if page.Resources != None else None)

            for n,id in enumerate(images):
                obj = images[id]
                print(LINE_SINGLE)
                print(f'[{n+1}] Image ID = {id}')
                pprint(obj)
                print(LINE_SINGLE)

                # ---------- Compress images ----------
                if options.compress:
                    pageArg = page if options.forceRGB else None
                    PdfImage.compress_image_xobject(obj, pageArg, options)
                    print(LINE_SINGLE)
                    print("RESULT:")
                    pprint(obj)

                # ---------- Extract images ----------
                else:

                    image = PdfImage.decode(obj,page, adjustColors = True, applyMasks = True, applyIntent = options.applyIntent)
                    if image == None: warn('failed to extract image; continuing'); continue
                    print(image.format, image.mode)
                    if image == None: warn(f'failed to decode image'); continue
                    dpi = options.dpi if options.dpi != None else image.info.get('dpi') if image.info.get('dpi') != None else (72,72)
                    icc_profile = image.info.get('icc_profile')

                    # Save images
                    formats = {'TIFF':'.tif', 'PNG':'.png', 'JPEG':'.jpg', 'JPEG2000':'.jp2'}
                    ext = formats[image.format] if image.format in formats else '.tif'
                    tiff_compression = 'group4' if image.mode == '1' else 'tiff_lzw'
                    outputPath = fileBase  +f'.page{pageNo}.image{n+1}' + ext
                    print(f'Extracting --> {outputPath}')
                    if image.format in ['PNG', 'JPEG', 'JPEG2000']:
                        image.save(outputPath, dpi=dpi, icc_profile=icc_profile, quality='keep')
                    else:
                        image.save(outputPath, dpi=dpi, icc_profile=icc_profile, compression=tiff_compression)

        if options.compress:
            pdfOutPath = fileBase + '-compressed' + fileExt
            print(LINE_DOUBLE)
            print(f'Writing output to {pdfOutPath}')
            PdfWriter(pdfOutPath, trailer=pdf, compress=True).write()

    # ---------- Input is images: convert to PDF ----------
    else:

        if pdfPath == None: pdfPath = imagePaths[0]+".pdf"

        pdf = PdfWriter(pdfPath,compress=True)
        N = len(imagePaths)
        for n,imagePath in enumerate(imagePaths):
            image = Image.open(imagePath)
            dpi_orig = image.info.get('dpi')
            print(f'[{n+1}/{N}] {image.size} {dpi_orig} {image.format} {image.mode} {imagePath}')
            dpi = dpi if dpi != None else image.info.get('dpi') if image.info.get('dpi') != None else (72,72)
            image.info['dpi'] = dpi
            page, dpi_actual = PdfImage.pil_image_to_pdf_page(image)
            pdf.addPage(page)
            print(f'[{n+1}/{N}] {image.mode} {image.size} @ {dpi_actual} dpi')

        pdf.write()
        print(f'Output written to {pdfPath}')

