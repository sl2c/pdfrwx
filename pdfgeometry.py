#!/usr/bin/env python3

def prec(self, precision = 3):
    '''
    Returns a list of strings representing CTM values with 3 decimal points accuracy.
    '''
    p = lambda x: f'{round(x*10**precision)/10**precision:f}'.rstrip('0').rstrip('.')
    return [p(a) for a in self]

# ================================================== class CTM

class CTM(list):
    '''
    A CTM matrix
    '''

    def __init__(self, ctm = [1,0,0,1,0,0]):
        '''
        Creates CTM matrix from a list of floats/strings: [a,b,c,d,e,f]. See PDF Ref. sec. 4.2.3
        '''
        if not isinstance(ctm,list) or len(ctm) != 6: raise ValueError(f'invalid CTM matrix: {ctm}')
        try: ctm = [float(x) for x in ctm]
        except: raise ValueError(f'invalid CTM matrix: {ctm}')
        self += ctm

    def equal(self, ctm: 'CTM'):
        '''
        True if self is equal to ctm
        '''
        return all(self[i]==ctm[i] for i in range(6))

    def multiply(self, ctm: 'CTM'):
        '''
        Returns self * ctm
        '''
        a,b = self, ctm
        return CTM([a[0]*b[0] + a[2]*b[1], a[1]*b[0] + a[3]*b[1],
                a[0]*b[2] + a[2]*b[3], a[1]*b[2] + a[3]*b[3],
                a[0]*b[4] + a[2]*b[5] + a[4], a[1]*b[4] + a[3]*b[5] + a[5]])

    def invert(self):
        '''
        Returns an inverse of self
        '''
        a,b,c,d,e,f = self
        det = a*d-b*c
        if det == 0: raise ValueError(f'a degenerate CTM matrix has no inverse: {self}')
        return CTM([d/det, -b/det, -c/det, a/det, (c*f-d*e)/det, (b*e-a*f)/det])

# ================================================== class VEC

class VEC(list):
    '''
    A 2D vector
    '''

    def __init__(self, vec = [0,0]):
        '''
        Creates a 2D vector from a list of floats/strings: [x,y]
        '''
        if not isinstance(vec,list) or len(vec) != 2: raise ValueError(f'invalid vector: {vec}')
        try: vec = [float(x) for x in vec]
        except: raise ValueError(f'invalid vector: {vec}')
        self += vec

    def equal(self, vec: 'VEC'):
        '''
        True if self is equal to vec
        '''
        return all(self[i]==vec[i] for i in range(2))

    def transform(self, ctm: CTM):
        '''
        Returns self transformed by ctm
        '''
        v,m = self,ctm
        return VEC([v[0]*m[0] + v[1]*m[2] + m[4], v[0]*m[1] + v[1]*m[3] + m[5]])

# ================================================== class BOX

class BOX(list):
    '''
    A box: [xmin, ymin, xmax, ymax]
    '''

    def __init__(self, box = [0,0,0,0]):
        '''
        Creates a box from a list of floats/string: [xmin, ymin, xmax, ymax]
        '''
        if not isinstance(box,list) or len(box) != 4: raise ValueError(f'invalid box: {box}')
        try: box = [float(x) for x in box]
        except: raise ValueError(f'invalid box: {box}')
        self += box

    def equal(self, box: 'BOX'):
        '''
        True if self is equal to box
        '''
        return all(self[i] == box[i] for i in range(4))

    def enlarge(self, box: 'BOX'):
        '''
        Returns a minimal BOX such that it contains both self and box
        '''
        return BOX([min(self[0],box[0]), min(self[1],box[1]), max(self[2],box[2]), max(self[3],box[3])])

    def transformFrom(self, box: 'BOX'):
        '''
        Returns a ctm such that box.transform(ctm) == self
        '''
        ctm1 = CTM([1,0,0,1,self[0],self[1]])
        ctm2 = CTM([(self[2]-self[0])/(box[2]-box[0]),0,0,(self[3]-self[1])/(box[3]-box[1]),0,0])
        ctm3 = CTM([1,0,0,1,-box[0],-box[1]])
        return ctm1.multiply(ctm2).multiply(ctm3)

    def transform(self, ctm: CTM):
        '''
        Returns a minimal box such that it contains self transformed by the ctm, i.e. ctm * self
        '''
        xmin,ymin,xmax,ymax = self

        v1 = VEC([xmin,ymin])
        v2 = VEC([xmin,ymax])
        v3 = VEC([xmax,ymin])
        v4 = VEC([xmax,ymax])

        v1 = v1.transform(ctm)
        v2 = v2.transform(ctm)
        v3 = v3.transform(ctm)
        v4 = v4.transform(ctm)

        xmin = min(min(v1[0],v2[0]),min(v3[0],v4[0]))
        xmax = max(max(v1[0],v2[0]),max(v3[0],v4[0]))
        ymin = min(min(v1[1],v2[1]),min(v3[1],v4[1]))
        ymax = max(max(v1[1],v2[1]),max(v3[1],v4[1]))

        return BOX([xmin,ymin,xmax,ymax])

    def scale(self, scale = 1.0):
        '''
        Returns a box which is equal to self scaled with self's center as a fixed point
        '''
        x = (self[0]+self[2])/2
        y = (self[1]+self[3])/2
        ctm1 = CTM([1,0,0,1,-x,-y])
        ctm2 = CTM([scale,0,0,scale,0,0])
        ctm3 = CTM([1,0,0,1,x,y])
        return self.transform(ctm1).transform(ctm2).transform(ctm3)
        







# ========================================================

class PdfMatrix:

    def multiply(matrix1:list, matrix2:list):
        '''
        Returns a product of two affine matrices: matrix1 * matrix2
        '''
        a,b = matrix1, matrix2
        try:
            return [
                a[0]*b[0] + a[2]*b[1],
                a[1]*b[0] + a[3]*b[1],
                a[0]*b[2] + a[2]*b[3],
                a[1]*b[2] + a[3]*b[3],
                a[0]*b[4] + a[2]*b[5] + a[4],
                a[1]*b[4] + a[3]*b[5] + a[5]
            ]
        except:
            return None
        
    def transform(matrix:list, vector:list):
        '''
        Returns a vector transformed by the matrix: matrix * vector
        '''
        m,v = matrix, vector
        try:
            return [
                m[0]*v[0] + m[2]*v[1] + m[4],
                m[1]*v[0] + m[3]*v[1] + m[5]
            ]
        except:
            return None