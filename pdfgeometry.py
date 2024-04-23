#!/usr/bin/env python3

from math import sqrt, ceil

# def prec(self, precision = 3):
#     '''
#     Returns a list of strings representing matrix values with 3 decimal points accuracy.
#     '''
#     p = lambda x: f'{round(x*10**precision)/10**precision:f}'.rstrip('0').rstrip('.')
#     return [p(a) for a in self]

# ================================================== class VEC

class VEC(list):
    '''
    A 2D vector
    '''

    def __init__(self, vec:list = [0,0]):
        '''
        Creates a 2D vector from a list of ints/floats/strings: [x,y]
        '''
        try: super().__init__([float(x) for x in vec]) ; assert len(vec) == 2
        except: raise ValueError(f'invalid vector: {vec}') from None

    def __neg__(self):
        '''Negation: -self'''
        return VEC([-x for x in self])
                
    def __add__(self, vec: 'VEC'):
        '''Addition: self + vec'''
        if not isinstance(vec, VEC): raise TypeError(f'cannot add {self} and {vec}')
        return VEC([self[0]+vec[0],self[1]+vec[1]])

    def __sub__(self, vec: 'VEC'):
        '''Subtraction: self - vec'''
        if not isinstance(vec, VEC): raise TypeError(f'cannot subtract {vec} from {self}')
        return VEC([self[0]-vec[0],self[1]-vec[1]])

    def __mul__(self, x):
        '''Multiplication: self * x where x can be an int, a float, or a VEC, the latter case producing a dot product'''
        if isinstance(x, int) or isinstance(x, float):
            return VEC([self[0]*x, self[1]*x])
        elif isinstance(x, VEC):
            return float(self[0]*x[0] + self[1]*x[1])
        else:
            raise TypeError(f'cannot multiply {self} and {x}')
        
    def copy(self):
        '''A copy'''
        return VEC(self)

        
# ================================================== class CTM

class MAT(list):
    '''
    A transformation matrix
    '''

    def __init__(self, mat:list = [1,0,0,1,0,0]):
        '''
        Creates a matrix from a list of ints/floats/strings: [a,b,c,d,e,f]. See PDF Ref. sec. 4.2.3
        '''
        try: super().__init__([float(x) for x in mat]) ; assert len(mat) == 6
        except: raise ValueError(f'invalid matrix: {mat}') from None
            
    def __mul__(self, x):
        '''Multiplication: self * x -- a transformation of x, where x is a VEC, a MAT, or a BOX'''
        m = self
        if isinstance(x, VEC):
            return VEC([m[0]*x[0] + m[2]*x[1] + m[4], m[1]*x[0] + m[3]*x[1] + m[5]])
        elif isinstance(x, MAT):
            return MAT([m[0]*x[0] + m[2]*x[1], m[1]*x[0] + m[3]*x[1],
                    m[0]*x[2] + m[2]*x[3], m[1]*x[2] + m[3]*x[3],
                    m[0]*x[4] + m[2]*x[5] + m[4], m[1]*x[4] + m[3]*x[5] + m[5]])
        elif isinstance(x, BOX):
            ll, ur = x.ll(), x.ur()
            ul, lr = VEC([ll[0], ur[1]]), VEC([ur[0], ll[1]])
            return BOX(list(self * ll) + list(self * ur) + list(self * ul) + list(self * lr))
        else:
            raise TypeError(f'cannot multiply {self} and {x}')
        
    def __lt__(self, m:'MAT') -> bool:
        '''A partial order, self < m, for matrices that describe text boxes,
        in which case the partial order is interpreted as text precedence order.'''

        order1, order2 = False, False

        # The coords of the center of the left edge of the m-text box as seen in the self-box's ref. frame
        x,y = self.inv() * m * VEC([0,0.5])
        if (y < 1 and x > 0.5) or y < 0: order1 = True

        # # The coords of the center of the left edge of the self-text box as seen in the m-box's ref. frame
        x,y = m.inv() * self * VEC([0,0.5])
        if (y < 1 and x > 0.5) or y < 0: order2 = True

        # The pair is ordered if both members of the pair think it's ordered
        return order1 and not order2
        # return order1

    def spacer(self, prev:'MAT') -> str:
        '''
        Infer a spacer between two text chunks interpreting self and prev as their boxes.
        '''
        # A typical space width for a proportional font is 0.25-0.33, for fixed-width-font - 0.67
        SPACE_WIDTH = 0.5

        # This is the min char spacing that is considered a space
        THRESHOLD = 0.125

        # Previous text box
        ll = prev * VEC([0,0])
        ul = prev * VEC([0,1])
        lr = prev * VEC([1,0])

        # The space unit is determined by the aspect ratio of the previous text box
        UNIT = SPACE_WIDTH * sqrt((ul - ll) * (ul - ll)) / sqrt((lr - ll) * (lr - ll))

        # The center of the left edge of the new text box:
        midPoint = self * VEC([0,0.5])

        # The midPoint coords in the previous 'box system of coordinates'
        x,y = prev.inv() * midPoint

        if not 0 < y < 1: spacer = '\n' * int(round(abs(y-0.5)))
        elif x - 1 < THRESHOLD * UNIT: spacer = ''
        else: spacer = ' ' * int(ceil((x - 1 - THRESHOLD * UNIT)/UNIT))

        return spacer


    def copy(self):
        '''A copy'''
        return MAT(self)

    def det(self):
        '''Determinant'''
        return float(self[0]*self[3]-self[1]*self[2])

    def inv(self):
        '''Inversion'''
        a,b,c,d,e,f = self
        det = self.det()
        if det == 0: raise ValueError(f'det = 0: {self}')
        return MAT([d/det, -b/det, -c/det, a/det, (c*f-d*e)/det, (b*e-a*f)/det])

# ================================================== class BOX

class BOX(list):
    '''
    The box class represents a bounding box of a set of points.
    '''

    def __init__(self, points:list):
        '''Creates a minimal box that contains all points from the list.
        The format: points = [x1, y1, x2, y2, ...]'''
        try:
            assert len(points) % 2 == 0
            xx = [float(x) for x in points[::2]]; yy = [float(y) for y in points[1::2]]
            xmin, xmax = min(x for x in xx), max(x for x in xx)
            ymin, ymax = min(y for y in yy), max(y for y in yy)
            super().__init__([xmin, ymin, xmax, ymax])
        except:
            raise ValueError(f'invalid points list: {points}') from None
    
    def __contains__(self, x) -> bool: 
        '''Inside: x in self'''
        if isinstance(x, VEC):
            return (self[0] <= x[0] <= self[2]) and (self[1] <= x[1] <= self[3])
        elif isinstance(x, BOX):
            return x.ll() in self and x.ur() in self
        else:
            raise TypeError(f'cannot check if {x} is inside {self}')

    def __add__(self, box: 'BOX') -> 'BOX':
        '''Add: self + box, which is a minimal BOX such that it contains both self and box'''
        if box == None: return self.copy()
        if not isinstance(box, BOX): raise TypeError(f'cannot add {self} and {box}')
        return BOX(list(self) + list(box))

    def __mul__(self, scale:float) -> 'BOX':
        '''Multiply: self * scale, produces a scaled version of self with self's center as a fixed point'''
        center, diag = (self.ll() + self.ur())*0.5, (self.ur() - self.ll())*0.5
        return BOX(list(center - diag*scale) + list(center + diag*scale))

    def copy(self):
        '''A copy'''
        return BOX(self)

    def w(self):
        '''Box width'''
        return self[2] - self[0]

    def h(self):
        '''Box height'''
        return self[3] - self[1]
    
    def ll(self) -> VEC:
        '''Lower-left point'''
        return VEC(self[:2])

    def ur(self) -> VEC:
        '''Upper-right point'''
        return VEC(self[2:])

    def transformFrom(self, box: 'BOX'):
        '''Returns a mat such that mat * box == self'''
        s1x,s1y = self.ll; s2x,s2y = self.ur
        b1x,b1y = box.ll; b2x,b2y = box.ur
        return MAT([1,0,0,1,s1x,s1y]) * MAT([(s2x-s1x)/(b2x-b1x),0,0,(s2y-s1y)/(b2y-b1y),0,0]) * MAT([1,0,0,1,-b1x,-b1y])

# # ========================================================

# class PdfMatrix:

#     def multiply(matrix1:list, matrix2:list):
#         '''
#         Returns a product of two affine matrices: matrix1 * matrix2
#         '''
#         a,b = matrix1, matrix2
#         try:
#             return [
#                 a[0]*b[0] + a[2]*b[1],
#                 a[1]*b[0] + a[3]*b[1],
#                 a[0]*b[2] + a[2]*b[3],
#                 a[1]*b[2] + a[3]*b[3],
#                 a[0]*b[4] + a[2]*b[5] + a[4],
#                 a[1]*b[4] + a[3]*b[5] + a[5]
#             ]
#         except:
#             return None
        
#     def transform(matrix:list, vector:list):
#         '''
#         Returns a vector transformed by the matrix: matrix * vector
#         '''
#         m,v = matrix, vector
#         try:
#             return [
#                 m[0]*v[0] + m[2]*v[1] + m[4],
#                 m[1]*v[0] + m[3]*v[1] + m[5]
#             ]
#         except:
#             return None