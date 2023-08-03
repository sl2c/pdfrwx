#!/usr/bin/env python3


class PdfBezier:

    def bezier_to_relative(self, bezier: list):
        '''Converts Bezier curve from standard to relative coordinate representation'''
        P0,P1,P2,P3 = bezier
        return [(P0+P3)/2, (P3-P0)/2, (P1-P0+P2-P3)/2, (P0-P1+P2-P3)/2]

    def bezier_from_relative(self, bezier_relative: list):
        '''Converts Bezier curve from relative to standard coordinate representation'''
        Q,R,B,F = bezier_relative
        return [Q-R, Q-R+B-F, Q+R+B+F, Q+R]

    def bezier_from_list(self, c: list):
        '''Creates a list of complex numbers from the list of real coordinate pairs'''
        return [complex(c[i],c[i+1]) for i in range(0,len(c),2)]

    def bezier_to_list(self, bezier: list):
        '''Creates a list of real coordinate pairs from the list of complex numbers'''
        nested = [[z.real, z.imag] for z in bezier]
        return [x for xx in nested for x in xx]

    def bezier_area(self, bezier:list):
        '''Returns area swept by the Bezier curve'''
        X = lambda a,b: a.real*b.imag-a.imag*b.real # cross-product
        Q,R,B,F = self.bezier_to_relative(bezier)
        return X(Q+1.2*B,R)+ 0.3*X(B,F)

    def bezier_scale(self, bezier:list, area:float):
        '''Returns the scaling factor needed to achieve given area under the Bezier curve'''
        X = lambda a,b: a.real*b.imag-a.imag*b.real # cross-product
        Q,R,B,F = self.bezier_to_relative(bezier)
        if X(B,F) == 0: return (area - X(Q,R))/(1.2 * X(B,R)) if X(B,R) != 0 else None
        D = (1.2*X(B,R))**2 - 1.2 * X(B,F) * (X(Q,R) - area) # the discriminant of the quadratic equation
        if D < 0: return None
        x1 = (-1.2 * X(B,R) + sqrt(D)) / (0.6 * X(B,F))
        x2 = (-1.2 * X(B,R) - sqrt(D)) / (0.6 * X(B,F))
        return x1 if abs(x1-1) <= abs(x2-1) else x2 # return the root that's closer to 1

    def enlarge_box(self, box:list, coords:list):
        for i in range(0,len(coords),2):
            x,y = coords[i:i+2]
            box[0] = min(box[0],x)
            box[1] = min(box[1],y)
            box[2] = max(box[2],x)
            box[3] = max(box[3],y)
        return max(box[2] - box[0],0), max(box[3] - box[1],0)


    def decimate_bezier(self, tree:list):
        '''Merge pairs of Bezier segments into single Bezier segments whenever this does not introduce noticeable artifacts.
        '''
        if len(tree) == 0: return []

        path_construction_commands = ['m','l','c','v','y','re','h','W','W*']
        path_construction_commands_that_move_cursor = ['m','l','c','v','y']
        path_painting_commands = ['s','S','f','F','f*','B','B*','b','b*','n']
        # path_commands = path_construction_commands + path_painting_commands
        p = lambda x: f'{round(x*1000000)/1000000:f}'.rstrip('0').rstrip('.')

        LINE_PRECISION = 0.1
        BEZIER_PRECISION = 0.1

        x,y = None, None    # current starting coordinates
        x0,y0 = None, None  # starting coordinates at the start of the replacement curve
        x1,y1 = None, None  # first control point at the start of the replacement curve (for Bezier)
        x2,y2 = None, None  # current ending coordinates; coptied to x,y at the end of each loop

        out = []
        inside = False
        total_area = 0
        box_none = [1000000,1000000,-1000000,-1000000]
        total_box = box_none.copy()
        leaf_replacement = None
        mode = None
        leaf_count, replacement_count = 0,0

        i = 0
        while i < len(tree):
            # print('----- ', i)
            leaf = tree[i]
            # print(leaf)
            cmd,args = leaf[0],leaf[1]

            leaf_count += 1

            if not inside:
                if cmd in ['m','re']: # a graphics block can actually start with a re command as well!
                    x,y = [float(a) for a in args[:2]]
                    inside = True
                out.append(leaf)
                # print(f'LEAF INSERTED (NOT INSIDE): {leaf}')
                i += 1
            else:
                if cmd in path_construction_commands:
                    args = [float(a) for a in args]

                    if mode == None and cmd in ['c','l']:
                        leaf_replacement = leaf
                        mode = cmd
                        total_area = 0
                        total_box = box_none.copy()
                        x0,y0 = x,y
                        # print('CMD: ', cmd)
                        if cmd == 'c': x1,y1 = args[0:2]
                        else: x1,y1 = None, None
                        # print(f'[{i}] REPLACEMENT INITIALIZED: {leaf_replacement}')

                    # print('MODE: ', mode)
                    
                    if cmd in ['c','l']:
                        new_list = [x,y] + args if cmd == 'c' else [x,y] + [x,y] + args + args
                        new_curve = self.bezier_from_list(new_list)
                        total_area += self.bezier_area(new_curve)

                        total_box_width, total_box_height = self.enlarge_box(total_box, new_list)
                        total_box_area = total_box_width * total_box_height

                        x2,y2 = args[-2:]
                        if mode == 'c':
                            if cmd == 'c':
                                replacement_list = [x0,y0] + [x1,y1] + args[-4:]
                            else:
                                replacement_list = [x0,y0] + [x1,y1] + [x,y] + [x2,y2]
                        else:
                            replacement_list = [x0,y0] + [x0,y0] + [x2,y2] + [x2,y2]

                        # print('REPLACEMENT LIST: ', replacement_list)
                        replacement_curve = self.bezier_from_list(replacement_list)
                        replacement_area = self.bezier_area(replacement_curve)

                        replacement_box = box_none.copy()
                        replacement_box_width, replacement_box_height = self.enlarge_box(replacement_box, replacement_list)
                        replacement_box_area = replacement_box_width * replacement_box_height

                        try:
                            boxes_epsilon = abs(replacement_box_area - total_box_area)/(replacement_box_area + total_box_area)
                        except:
                            boxes_epsilon = 0
                        
                        # print('BOXES: ', replacement_box_width, replacement_box_height, ' ??? ',total_box_width, total_box_height)

                    if mode == 'l' and cmd == 'l' \
                        and abs(replacement_area - total_area) <= LINE_PRECISION \
                        and abs(replacement_box_width - total_box_width) <= LINE_PRECISION \
                        and abs(replacement_box_height - total_box_height) <= LINE_PRECISION \
                        and boxes_epsilon < 0.5:

                        if leaf_replacement != None: replacement_count += 1

                        leaf_replacement = leaf
                        # print(f'[{i}] REPLACEMENT UPDATED: {leaf_replacement}')
                        i += 1
                        x,y = args[-2:]
                        continue

                    # if mode == 'c' and cmd in ['c','l'] and abs(replacement_area - total_area) <= BEZIER_PRECISION:
                    if mode == 'c' and cmd in ['c'] \
                        and abs(replacement_area - total_area) <= BEZIER_PRECISION \
                        and abs(replacement_box_width - total_box_width) <= BEZIER_PRECISION \
                        and abs(replacement_box_height - total_box_height) <= BEZIER_PRECISION \
                        and boxes_epsilon < 0.5:

                        if leaf_replacement != None: replacement_count += 1

                        scale = self.bezier_scale(replacement_curve, total_area)
                        # print(scale)
                        if scale != None and abs(scale-1)<=5:
                            Q,R,B,F = self.bezier_to_relative(replacement_curve)
                            B,F = scale*B,scale*F
                            replacement_curve = self.bezier_from_relative([Q,R,B,F])
                            replacement_list = self.bezier_to_list(replacement_curve)
                        leaf_replacement = ['c', [p(a) for a in replacement_list[2:]]]
                        # print(f'[{i}] REPLACEMENT UPDATED: {leaf_replacement}')
                        i += 1
                        x,y = args[-2:]
                        continue

                    total_area = 0
                    total_box = box_none.copy()
                    mode = None

                    if leaf_replacement != None:
                        out.append(leaf_replacement)
                        # print(f'REPLACEMENT INSERTED[{i}]: {leaf_replacement}')
                        leaf_replacement = None
                        continue

                    out.append(leaf)
                    # print(f'LEAF INSERTED (PATH CONSTRUCT): {leaf}')

                elif cmd in path_painting_commands:
                    inside = False
                    x,y = None,None
                    if leaf_replacement != None:
                        out.append(leaf_replacement)
                        # print(f'REPLACEMENT INSERTED[{i}]: {leaf_replacement}')
                        total_area = 0
                        total_box = box_none.copy()
                        leaf_replacement = None
                        mode = None
                    out.append(leaf)
                    # print(f'LEAF INSERTED (PATH PAINT): {leaf}')
                else:
                    raise ValueError(f'unexpected command inside path: {leaf}')

                # increment leaf index
                i += 1
                if cmd in path_construction_commands_that_move_cursor:
                    x,y = args[-2:]


        print(f'Leaves: {leaf_count}, replaced: {replacement_count}')
        return out

