#!/usr/bin/env python3

import numpy as np
import potrace

from pdfrw import PdfDict, IndirectPdfDict, PdfName, PdfObject

from pdfrwx.common import err
from pdfrwx.pdfimage import PdfImage

# ========================================================================== class PdfVectorImage

class PdfVectorImage:
    '''An auxiliary class that contains static functions for bitonal image processing
    '''

    def vectorize(obj:PdfDict, alpha = 1, upsample=False, smooth=0, vectorize=True):
        '''Converts PDF image xobject to vectorized PDF form xobject
        The mask may be optionally upsampled and/or smoothed; this may (or may not) increase
        the quality of the resulting vectorized form depending on the particular mask.
        The value of alpha = 0.93 preserves serifs a bit better; values > 1 make for more rounded curves.
        '''
        if obj == None or obj.Subtype != PdfName.Image: return False
        image = PdfImage.decode(obj)
        if image == None or image.mode != '1': return False

        mask = np.array(image)
        mask = np.flipud(mask) # flip upside down since images use inverted coordinate system
        mask = np.invert(mask) # black is 1 for masks

        h,w = mask.shape

        mask = np.where(mask, 1, 0).astype(np.uint8) # Convert boolean to uint8
        noncorner = None
        if upsample: mask, noncorner = PdfVectorImage.scale2x(mask, preserveCorners=True)
        if smooth != 0: mask = PdfVectorImage.smooth(mask, noncorner, smooth > 0)

        if not vectorize:

            mask = mask > 0
            mask = np.invert(mask)
            mask = np.flipud(mask)
            from PIL import Image
            image = Image.frombytes(mode='1', size = mask.shape[::-1], data = np.packbits(mask, axis=1))
            result = PdfImage.encode(image)
            if result == None: err('internal error')
            obj_new,dpi = result
 
            obj.clear()
            obj.stream = obj_new.stream
            obj.Type = obj_new.Type
            obj.Subtype = obj_new.Subtype
            obj.Width = obj_new.Width
            obj.Height = obj_new.Height
            # obj.BitsPerComponent = obj_new.BitsPerComponent
            obj.Filter = obj_new.Filter
            obj.DecodeParms = obj_new.DecodeParms
            obj.Decode = obj_new.Decode
            # obj.ColorSpace = obj_new.ColorSpace
            obj.ImageMask = PdfObject('true')

        else:

            # This allows to better preserve corners on elongated glyphs (e.g. horizontal bars)
            e = 1 - abs(w-h)/(w+h)
            alphamax = (1 if e >= 0.25 else 4*e)*alpha 

            trace = potrace.Bitmap(mask).trace(turnpolicy=potrace.TURNPOLICY_BLACK, alphamax=alphamax)

            stream  = f'q\n{1/w:f} 0 0 {1/h:f} 0 0 cm\n'
            stream += '\n'.join(PdfVectorImage.potrace_curve_to_pdf_stream(curve, 0.5 if upsample else 1) \
                                    for curve in trace.curves_tree) + 'f*\n'
            stream += 'Q\n'

            # Modify obj
            obj.clear()
            obj.Type = PdfName.XObject
            obj.Subtype = PdfName.Form
            obj.BBox = [0,0,1,1]
            obj.stream = stream

        return True

    def potrace_curve_to_pdf_stream(curve, scaleFactor = 1):
        '''Translates a single curve from the output of Potrace into PDF stream.
        '''
        z = scaleFactor
        p = lambda x: y(x[0])+' '+y(x[1])
        y = lambda x: f'{round(x*z*100)/100:f}'.rstrip('0').rstrip('.')
        d = f'{p(curve.start_point)} m\n'
        for s in curve.segments:
            if s.is_corner: d += f'{p(s.c)} l {p(s.end_point)} l\n'
            else: d += f'{p(s.c1)} {p(s.c2)} {p(s.end_point)} c\n'
        d += "h\n"
        d += ' '.join(PdfVectorImage.potrace_curve_to_pdf_stream(child, scaleFactor) for child in curve.children)
        return d

    def scale2x(image:np.array, preserveCorners = True):
        '''Image upsampling using the scale2x algorithm: https://www.scale2x.it/algorithm
        The image is an arbitrary 2D numpy array (any dtype). Given a WxH image, returns a tuple (X, NONCORNER)
        where X is a 2Wx2H array of the same dtype and NONCORNER is a boolean array of the same dimensions
        where each pixel which is not at the corner  of a shape is marked as true.
        '''
        h,w = image.shape
        hzeros = np.zeros(w)
        vzeros = np.zeros(h).reshape(h,1)

        T = np.vstack((hzeros,image[:-1,:]))
        B = np.vstack((image[1:,:],hzeros))
        L = np.hstack((vzeros,image[:,:-1]))
        R = np.hstack((image[:,1:],vzeros))

        TL = np.vstack((hzeros,L[:-1,:]))
        BL = np.vstack((L[1:,:],hzeros))
        TR = np.vstack((hzeros,R[:-1,:]))
        BR = np.vstack((R[1:,:],hzeros))

        edge = (T!=B) & (L!=R)

        NONEDGE = np.logical_not(edge)
        
        UL_NONCORNER = np.logical_or(NONEDGE, np.logical_or(T!=TR, L!=BL))
        UR_NONCORNER = np.logical_or(NONEDGE, np.logical_or(T!=TL, R!=BR))
        LL_NONCORNER = np.logical_or(NONEDGE, np.logical_or(B!=BR, L!=TL))
        LR_NONCORNER = np.logical_or(NONEDGE, np.logical_or(B!=BL, R!=TR))


        if preserveCorners:
            UL = np.where(edge, np.where(np.logical_and(T==L, UL_NONCORNER),L,image),image)
            UR = np.where(edge, np.where(np.logical_and(T==R, UR_NONCORNER),R,image),image)
            LL = np.where(edge, np.where(np.logical_and(B==L, LL_NONCORNER),L,image),image)
            LR = np.where(edge, np.where(np.logical_and(B==R, LR_NONCORNER),R,image),image)
        else:
            UL = np.where(edge, np.where(T==L,L,image),image)
            UR = np.where(edge, np.where(T==R,R,image),image)
            LL = np.where(edge, np.where(B==L,L,image),image)
            LR = np.where(edge, np.where(B==R,R,image),image)


        X = np.empty(image.size * 4, dtype=image.dtype).reshape(image.shape[0]*2,image.shape[1]*2)
        X[0::2,0::2] = UL
        X[0::2,1::2] = UR
        X[1::2,0::2] = LL
        X[1::2,1::2] = LR

        NONCORNER = np.empty(image.size * 4, dtype=image.dtype).reshape(image.shape[0]*2,image.shape[1]*2)
        NONCORNER[0::2,0::2] = UL_NONCORNER
        NONCORNER[0::2,1::2] = UR_NONCORNER
        NONCORNER[1::2,0::2] = LL_NONCORNER
        NONCORNER[1::2,1::2] = LR_NONCORNER

        return X, NONCORNER

    def scale3x(image:np.array):
        '''The scale3x algorithm
        '''
        h,w = image.shape
        hzeros = np.zeros(w)
        vzeros = np.zeros(h).reshape(h,1)

        T = np.vstack((hzeros,image[:-1,:]))
        B = np.vstack((image[1:,:],hzeros))
        L = np.hstack((vzeros,image[:,:-1]))
        R = np.hstack((image[:,1:],vzeros))

        # TL = np.vstack((hzeros,L[:-1,:]))
        # BL = np.vstack((L[1:,:],hzeros))
        # TR = np.vstack((hzeros,R[:-1,:]))
        # BR = np.vstack((R[1:,:],hzeros))

        edge = (T!=B) & (L!=R)

        UL = np.where(edge, np.where(T==L,L,image),image)
        UR = np.where(edge, np.where(T==R,R,image),image)
        LL = np.where(edge, np.where(B==L,L,image),image)
        LR = np.where(edge, np.where(B==R,R,image),image)

        X = np.empty(image.size * 9, dtype=image.dtype).reshape(image.shape[0]*3,image.shape[1]*3)

        X[0::3,0::3] = UL
        X[0::3,1::3] = image
        X[0::3,2::3] = UR

        X[1::3,0::3] = image
        X[1::3,1::3] = image
        X[1::3,2::3] = image

        X[2::3,0::3] = LL
        X[2::3,1::3] = image
        X[2::3,2::3] = LR

        return X

    def smooth(mask:np.array, noncorner = None, thicken = True):
        '''Mask smoothing algorithm, where mask is 2D array of 1s & 0s (mask.dtype should be np.uint8).
        Pixel Pi is set to 1 (black) if Ni(1)*2 + Ni(2) >= 7, where Ni(k) is the number
        of 1-pixels (black) that are at distance k from the i-th pixel, and where the distance is Manhattan/Checkerboard:
        left,right,top,bottom-adjacent pixels are at distance 1, diagonal adjacent pixels are at distance 2, etc.
        '''
        if mask.dtype != np.uint8: err(f'expected mask.dtype == uint8, got: {mask.dtype}')
        h,w = mask.shape
        hzeros = np.zeros(w)
        vzeros = np.zeros(h).reshape(h,1)
        T = np.vstack((hzeros,mask[:-1,:]))
        B = np.vstack((mask[1:,:],hzeros))
        L = np.hstack((vzeros,mask[:,:-1]))
        R = np.hstack((mask[:,1:],vzeros))
        TL = np.hstack((vzeros,T[:,:-1]))
        TR = np.hstack((T[:,1:],vzeros))
        BL = np.hstack((vzeros,B[:,:-1]))
        BR = np.hstack((B[:,1:],vzeros))

        neighbors = (T+B+L+R)*2 + TL+TR+BL+BR

        sum = 7 if thicken else 5
        value = 1 if thicken else 0

        if noncorner is not None:
            result = np.where(np.logical_and(neighbors == sum, noncorner), value, mask)            
        else:
            result = np.where(neighbors >= 7, 1, mask) if thicken else np.where(neighbors <= 7, 0, mask)

        return result

