# CCITT Group4 Decoder

class Group4Decoder(object):
    '''
    An implementation of the CCITT Group 4 (T.6) decoder. See:

    ITU-T Recommendation T.6: FACSIMILE CODING SCHEMES AND CODING
    CONTROL FUNCTIONS FOR GROUP 4 FACSIMILE APPARATUS
    '''

    EOFB = '000000000001000000000001'

    MODES_ENCODE = {
        'PASS' : '0001',
        'HOR' : '001',
        0   : '1',
        1   : '011',
        2   : '000011',
        3   : '0000011',
        -1  : '010',
        -2  : '000010',
        -3  : '0000010',
        'EXT' : '0000001',
    }
    MODES_DECODE = {v:k for k,v in MODES_ENCODE.items()}

    TERMINALS_WHITE_ENCODE = [
        '00110101', '000111', '0111', '1000', '1011', '1100','1110','1111',
        '10011','10100','00111','01000','001000','000011','110100','110101',
        '101010','101011','0100111','0001100','0001000','0010111','0000011','0000100',
        '0101000','0101011','0010011','0100100','0011000','00000010','00000011','00011010',
        '00011011','00010010','00010011','00010100','00010101','00010110','00010111','00101000',
        '00101001','00101010','00101011','00101100','00101101','00000100','00000101','00001010',
        '00001011','01010010','01010011','01010100','01010101','00100100','00100101','01011000',
        '01011001','01011010','01011011','01001010','01001011','00110010','00110011','00110100'
    ]
    TERMINALS_WHITE_DECODE = {v:k for k,v in enumerate(TERMINALS_WHITE_ENCODE)}

    TERMINALS_BLACK_ENCODE = [
        '0000110111','010','11','10','011','0011','0010','00011',
        '000101','000100','0000100','0000101','0000111','00000100','00000111','000011000',
        '0000010111','0000011000','0000001000','00001100111','00001101000','00001101100','00000110111','00000101000',
        '00000010111','00000011000','000011001010','000011001011','000011001100','000011001101','000001101000','000001101001',
        '000001101010','000001101011','000011010010','000011010011','000011010100','000011010101','000011010110','000011010111',
        '000001101100','000001101101','000011011010','000011011011','000001010100','000001010101','000001010110','000001010111',
        '000001100100','000001100101','000001010010','000001010011','000000100100','000000110111','000000111000','000000100111',
        '000000101000','000001011000','000001011001','000000101011','000000101100','000001011010','000001100110','000001100111'
    ]
    TERMINALS_BLACK_DECODE = {v:k for k,v in enumerate(TERMINALS_BLACK_ENCODE)}

    MAKEUP_LOW_WHITE_ENCODE = [
        '11011','10010','010111','0110111','00110110','00110111','01100100','01100101',
        '01101000','01100111','011001100','011001101','011010010','011010011','011010100','011010101',
        '011010110','011010111','011011000','011011001','011011010','011011011','010011000','010011001',
        '010011010','011000','010011011'
    ]
    MAKEUP_LOW_WHITE_DECODE = {v:(k+1)*64 for k,v in enumerate(MAKEUP_LOW_WHITE_ENCODE)}

    MAKEUP_LOW_BLACK_ENCODE = [
        '0000001111','000011001000','000011001001','000001011011','000000110011','000000110100','000000110101','0000001101100',
        '0000001101101','0000001001010','0000001001011','0000001001100','0000001001101','0000001110010','0000001110011','0000001110100',
        '0000001110101','0000001110110','0000001110111','0000001010010','0000001010011','0000001010100','0000001010101','0000001011010',
        '0000001011011','0000001100100','0000001100101'
    ]
    MAKEUP_LOW_BLACK_DECODE = {v:(k+1)*64 for k,v in enumerate(MAKEUP_LOW_BLACK_ENCODE)}

    MAKEUP_HIGH_ENCODE = [
        '00000001000','00000001100','00000001001','000000010010','000000010011','000000010100','000000010101','000000010110',
        '000000010111','000000011100','000000011101','000000011110','000000011111'
    ]
    MAKEUP_HIGH_DECODE = {v:(k + 28)*64 for k,v in enumerate(MAKEUP_HIGH_ENCODE)}

    MAKEUP_WHITE_DECODE = MAKEUP_LOW_WHITE_DECODE | MAKEUP_HIGH_DECODE
    MAKEUP_BLACK_DECODE = MAKEUP_LOW_BLACK_DECODE | MAKEUP_HIGH_DECODE

    # ---------------------------------------------------------------------------------------- decode()

    def decode(self, data:bytes, Columns:int, EncodedByteAlign:bool = False):
        '''
        Decodes a CCITT Group 4 (T.6) encoded bytes stream. Returns decoded
        bitonal image pixel data as a bytes stream, which consists of a sequence of lines,
        each line consisting of a sequence of bits, contiguously packed,
        with ends of lines padded with 0-bits to whole bytes, if necessary.
        '''

        def dump(outBits:str, Columns:int, a0:int, line:int, message:str):
            nBytes = (Columns + 7 ) // 8
            if a0 < nBytes * 8:
                outBits += '0' * (nBytes * 8 - a0)
            from PIL import Image, ImageChops
            pil = Image.frombytes('1',(Columns, line+1), toBytes(outBits))
            ImageChops.invert(pil).save('dump.tif')
            raise ValueError(message + '\n' + 'salvaged parts of the image written to dump.tif')

        MODES = self.MODES_DECODE
        WHITE, BLACK = 0, 1

        toBytes = lambda bits: b''.join(int(bits[i:i+8],2).to_bytes(1,'big') for i in range(0,len(bits),8))

        # Bit streams
        inBits = ''.join(f'{d:08b}' for d in data)
        inPos = 0
        outBits = ''
        peek = lambda i: inBits[inPos:inPos+i]
        getBit = lambda color: '0' if color == WHITE else '1'

        b = []
        a = []
        a0 = -1
        color = WHITE
        line = 0

        while True:

            b1 = next((b1 for n,b1 in enumerate(b) if b1 > a0 and n%2 == color), Columns)
            b2 = next((b2 for n,b2 in enumerate(b) if b2 >= b1 and n%2 == color^1), Columns)

            if a0 == -1: a0 = 0

            l = None
            for i in range(1,8):

                l = MODES.get(peek(i), None)
                
                if l is not None:

                    inPos += i

                    if l == 'PASS':
                        # Pass mode
                        outBits += getBit(color)*(b2-a0)
                        a0 = b2
                    elif l == 'HOR':
                        # Horizontal mode
                        M01, inPos = self.get_run_length(inBits, inPos, color)
                        M12, inPos = self.get_run_length(inBits, inPos, color^1)
                        outBits += getBit(color)*M01 + getBit(color^1)*M12
                        a1, a2 = a0 + M01, a0 + M01 + M12
                        a.append(a1); a.append(a2)
                        a0 += M01 + M12
                    elif isinstance(l, int):
                        # Vertical mode (flips color)
                        outBits += getBit(color)*(b1 + l - a0)
                        a0 = b1 + l
                        if a0 < Columns:
                            a.append(a0)
                        color ^= 1
                    elif l == 'EXT':
                        # Extensions, incl. uncompressed mode: implement later when sample files are available
                        l = peek(3)
                        raise ValueError(f"Extension code not implemented: E{l:03b}")
                    
                    break
                    
            if l is None:

                if peek(24) != self.EOFB:
                    dump(outBits, Columns, a0, line, f'unrecognized bits at line = {line}, a0 = {a0}: {peek(24)}')

                if res := len(outBits) % 8:
                    outBits += '0'*(8-res)
                return toBytes(outBits)

            if a0 > Columns:
                dump(outBits, Columns, a0, line, f'extra bits at line = {line}, a0 = {a0}')
            
            if a0 == Columns:
                a0 = -1
                color = WHITE
                b = a
                a = []
                line += 1
                if EncodedByteAlign:
                    if res := inPos % 8:
                        inPos += 8-res
                if res := len(outBits) % 8:
                    outBits += '0'*(8-res)
                

    # ---------------------------------------------------------------------------------------- get_run_length()

    def get_run_length(self, inBits:str, inPos:int, color:int):
        '''
        '''
        WHITE, BLACK = 0, 1
        MAKEUP = self.MAKEUP_WHITE_DECODE if color == WHITE else self.MAKEUP_BLACK_DECODE
        TERMINAL = self.TERMINALS_WHITE_DECODE if color == WHITE else self.TERMINALS_BLACK_DECODE

        peek = lambda i: inBits[pos:pos+i]

        bits = 0
        pos = inPos
        while True:
            
            l = None
            for i in range(2, 14):

                codeword = peek(i)

                l = TERMINAL.get(codeword, None)
                if l is not None:
                    pos += i
                    bits += l
                    return bits, pos
                
                l = MAKEUP.get(codeword, None)
                if l is not None:
                    pos += i
                    bits += l
                    break

            if l is None:
                if bits == 0:
                    raise ValueError(f'failed to get {"white" if color == WHITE else "black"} run length: {peek(24)}')
                return bits, pos
