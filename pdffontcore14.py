#!/usr/bin/env python3

import struct, zlib, base64

from pdfrw import PdfName, IndirectPdfDict, PdfDict

from pdfrwx.common import err, warn, msg

# =========================================================================== class PdfCore14Fonts

class PdfFontCore14:
        
   # ascent/descent font parameters
    ASCENT_DESCENT = {
        '/Courier': (629, -157),
        '/Courier-Bold': (626, -142),
        '/Courier-BoldOblique': (626, -142),
        '/Courier-Oblique': (629, -157),
        '/Helvetica': (718, -207),
        '/Helvetica-Bold': (718, -207),
        '/Helvetica-BoldOblique': (718, -207),
        '/Helvetica-Oblique': (718, -207),
        '/Times-Roman': (683, -217),
        '/Times-Bold': (676, -205),
        '/Times-BoldItalic': (699, -205),
        '/Times-Italic': (683, -205),
        '/Symbol': (0, 0),
        '/ZapfDingbats': (0, 0)
        }

   # A map from aliases to stanard names
    CORE14_FONTNAMES_ALIASES = {
        '/CourierNew':'/Courier', '/CourierNew,Italic':'/Courier-Oblique',
        '/CourierNew,Bold':'/Courier-Bold', '/CourierNew,BoldItalic':'/Courier-BoldOblique',
        '/Arial':'/Helvetica', '/Arial,Italic':'/Helvetica-Oblique',
        '/Arial,Bold':'/Helvetica-Bold', '/Arial,BoldItalic':'/Helvetica-BoldOblique',
        '/TimesNewRoman':'/Times-Roman', '/TimesNewRoman,Italic':'/Times-Italic',
        '/TimesNewRoman,Bold':'/Times-Bold', '/TimesNewRoman,BoldItalic':'/Times-BoldItalic',
        '/Symbol':'/Symbol', '/ZapfDingbats':'/ZapfDingbats'
    }

    def standard_fontname(fontname:PdfName):
        '''
        If the fontname is a standard core14 font name returns fontname. Otherwise, if
        it is an alias for a standard core14 font name, returns the corresponding standard core 14 font name.
        If none of the above, returns None. The standard core14 font names and their aliases are in
        PdfFontUtils.CORE14_FONTNAMES_ALIASES; see PDF Reference 1.7 Sec. 5.5.1; Sec. H.3 Implementation notes
        for Sec. 5.5.1.
        '''
        aliases = PdfFontCore14.CORE14_FONTNAMES_ALIASES
        return fontname if fontname in aliases.values() else aliases[fontname] if fontname in aliases else None
        
    def make_core14_font_dict(fontname:PdfName):
        '''
        Returns a core14 font dictionary for the specified font name or None if the specified
        font name is neither a standard core14 font name nor an alias for a standard core14 font name.
        See help for PdfFontCore14.standard_fontname() for more info
        '''
        if PdfFontCore14.standard_fontname(fontname) == None: return None
        return IndirectPdfDict(
            Type = PdfName.Font,
            Subtype = PdfName.Type1,
            BaseFont = PdfName(fontname[1:]),
        )

    def built_in_encoding(fontname:PdfName):
        '''
        If fontname is the name of one of Core 14 fonts
        (see PdfFontCore14.CORE14_FONTNAMES_ALIASES for Core 14 font names and aliases),
        returns the name of the built-in
        encoding of that font, otherwise returns None. Examples: '/Times' --> '/WinAnsiEncoding',
        '/Symbol' --> '/SymbolEncoding', '/ZapfDingbats' --> '/ZapfDingbatsEncoding'.
        '''
        standard_fontname = PdfFontCore14.standard_fontname(fontname)
        if standard_fontname == None: return None
        return '/ZapfDingbatsEncoding' if standard_fontname == '/ZapfDingbats' \
            else '/SymbolEncoding' if standard_fontname == '/Symbol' \
            else '/WinAnsiEncoding'

    def make_name2width(fontname:str):
        '''
        '''
        standard_fontname = PdfFontCore14.standard_fontname(fontname)
        if standard_fontname == None: return None

        # Creates the names vector
        if standard_fontname == '/Symbol':
            names ='''Alpha Beta Chi Delta Epsilon Eta Euro Gamma Ifraktur Iota Kappa Lambda Mu Nu Omega Omicron
                    Phi Pi Psi Rfraktur Rho Sigma Tau Theta Upsilon Upsilon1 Xi Zeta aleph alpha ampersand angle
                    angleleft angleright apple approxequal arrowboth arrowdblboth arrowdbldown arrowdblleft arrowdblright arrowdblup arrowdown arrowhorizex arrowleft arrowright arrowup arrowvertex
                    asteriskmath bar beta braceex braceleft braceleftbt braceleftmid bracelefttp braceright bracerightbt bracerightmid bracerighttp bracketleft bracketleftbt bracketleftex bracketlefttp
                    bracketright bracketrightbt bracketrightex bracketrighttp bullet carriagereturn chi circlemultiply circleplus club colon comma congruent copyrightsans copyrightserif degree
                    delta diamond divide dotmath eight element ellipsis emptyset epsilon equal equivalence eta exclam existential five florin
                    four fraction gamma gradient greater greaterequal heart infinity integral integralbt integralex integraltp intersection iota kappa lambda
                    less lessequal logicaland logicalnot logicalor lozenge minus minute mu multiply nine notelement notequal notsubset nu numbersign
                    omega omega1 omicron one parenleft parenleftbt parenleftex parenlefttp parenright parenrightbt parenrightex parenrighttp partialdiff percent period perpendicular
                    phi phi1 pi plus plusminus product propersubset propersuperset proportional psi question radical radicalex reflexsubset reflexsuperset registersans
                    registerserif rho second semicolon seven sigma sigma1 similar six slash space spade suchthat summation tau therefore
                    theta theta1 three trademarksans trademarkserif two underscore union universal upsilon weierstrass xi zero zeta'''
        elif standard_fontname == '/ZapfDingbats':
            names ='''a1 a10 a100 a101 a102 a103 a104 a105 a106 a107 a108 a109 a11 a110 a111 a112 a117 a118 a119 a12 a120
                    a121 a122 a123 a124 a125 a126 a127 a128 a129 a13 a130 a131 a132 a133 a134 a135 a136 a137 a138 a139
                    a14 a140 a141 a142 a143 a144 a145 a146 a147 a148 a149 a15 a150 a151 a152 a153 a154 a155 a156 a157
                    a158 a159 a16 a160 a161 a162 a163 a164 a165 a166 a167 a168 a169 a17 a170 a171 a172 a173 a174 a175
                    a176 a177 a178 a179 a18 a180 a181 a182 a183 a184 a185 a186 a187 a188 a189 a19 a190 a191 a192 a193
                    a194 a195 a196 a197 a198 a199 a2 a20 a200 a201 a202 a203 a204 a205 a206 a21 a22 a23 a24 a25 a26 a27
                    a28 a29 a3 a30 a31 a32 a33 a34 a35 a36 a37 a38 a39 a4 a40 a41 a42 a43 a44 a45 a46 a47 a48 a49 a5 a50
                    a51 a52 a53 a54 a55 a56 a57 a58 a59 a6 a60 a61 a62 a63 a64 a65 a66 a67 a68 a69 a7 a70 a71 a72 a73
                    a74 a75 a76 a77 a78 a79 a8 a81 a82 a83 a84 a85 a86 a87 a88 a89 a9 a90 a91 a92 a93 a94 a95 a96 a97
                    a98 a99 space'''
        else:
            names = '''A AE Aacute Acircumflex Adieresis Agrave Aring Atilde B C Ccedilla D E Eacute Ecircumflex Edieresis
                    Egrave Eth Euro F G H I Iacute Icircumflex Idieresis Igrave J K L Lslash M
                    N Ntilde O OE Oacute Ocircumflex Odieresis Ograve Oslash Otilde P Q R S Scaron T
                    Thorn U Uacute Ucircumflex Udieresis Ugrave V W X Y Yacute Ydieresis Z Zcaron a aacute
                    acircumflex acute adieresis ae agrave ampersand aring asciicircum asciitilde asterisk at atilde b backslash bar braceleft
                    braceright bracketleft bracketright breve brokenbar bullet c caron ccedilla cedilla cent circumflex colon comma copyright currency
                    d dagger daggerdbl degree dieresis divide dollar dotaccent dotlessi e eacute ecircumflex edieresis egrave eight ellipsis
                    emdash endash equal eth exclam exclamdown f fi five fl florin four fraction g germandbls grave
                    greater guillemotleft guillemotright guilsinglleft guilsinglright h hungarumlaut hyphen i iacute icircumflex idieresis igrave j k l
                    less logicalnot lslash m macron minus mu multiply n nine ntilde numbersign o oacute ocircumflex odieresis
                    oe ogonek ograve one onehalf onequarter onesuperior ordfeminine ordmasculine oslash otilde p paragraph parenleft parenright percent
                    period periodcentered perthousand plus plusminus q question questiondown quotedbl quotedblbase quotedblleft quotedblright quoteleft quoteright quotesinglbase quotesingle
                    r registered ring s scaron section semicolon seven six slash space sterling t thorn three threequarters
                    threesuperior tilde trademark two twosuperior u uacute ucircumflex udieresis ugrave underscore v w x y yacute
                    ydieresis yen z zcaron zero'''
                            
        names = names.split()
        names = [PdfName(name) for name in names]

        # Create the widths vector
        if standard_fontname in ['/Courier','/Courier-Bold','/Courier-BoldOblique','/Courier-Oblique']:
            widthsArray = [600]*229
        else:
            # The widths vectors are store zlib- and base64-compressed to save space and prevent accidental mods
            widthsDict = {
                '/Helvetica' : b'eJxVUbENwyAQ9POpPABSdmAJu6IhI6ShTpPOA6RP7wHSeoPsYCmtJXceAClt7h+MHJ0wz/F3z8lm5M2Mf5gVx7MzkVszk61I4J1x3Ekvt7xhVZhRulUbq5968lRdI+AEFOAzoIIjfYynB3+Vt3SiCyDzAuorJexJ+rGDNWt2wPkJ1md+5/R9G9I5vbGqSaiS3DQvSZVVqsuuoVn2jKhSs+DeNwt3pdMf3XXCJErsPffQ3+lWcp3FjQdxwht8UUaKwsscxRvqVVPJy2zJbfdaPYNkyFNq3oS/kTLkJN8fS7RbrA==',
                '/Helvetica-Bold' : b'eJxtkKERwzAMRSOJZQBfhvASDTLJCiXGJWUZILw8A5Rmk94ZloUZFXWASt+OW9D7JxD5/ycpnCRz+qO1KbHnKD0ncoe0kzhq92Re6SVrNfFqbmQj8o0pW6NqWikqmtjLDKLnwIEWedhEnTPQopqggc7FW0rleDcfGDftBvRdoRbpXhnUWChwI9HdbYOSKrzqcb9XatlGTmY4I8rXgmQrVBll1PSVLvUFTM0pR3cISNrEJ70hTOhetPBuFx1/wmZicqWCkuv3Ibus/15pPPYfQchXzQ==',
                '/Times-Roman' : b'eJxlkCEOwlAMhl9bhyZz3ODJBc8BdgUMp+AAeCRkB8AikLMk6AU8cmrqiWWKv+02FsiXt/1b27/t41r2XM8ojZp3EzUljhqjYuLgGWPtj0c0nwh2Xjki10mjmiqjoIpLqmQB/eIVJTlDJcrCg97AOkJvPRcn4WSh5w7KOVLBuf8fXJUkjTSY3iOAo+6iFeFidV7lUX9mMzAD58jIMJtlDtkz7Fb03XJLkZa0HiJ3dZNN6EMvjbkkm+k0ba1b3eDb6X2CZB11737U5lnIEzr+9f1+2a4fxRyQKg==',
                '/Times-Bold' : b'eJxNkDEOwjAMRWNnQT1AJm5ATtCNgQUpV2DpWViY2LuxsHKAnqIzB4iExJQpE992G6onFNf+/rbh2WeeN4ztXZmp8OA739G1UXwnNf8SLWpZ6iuiRj6CkYet+2YWuqkoCf6Y4Y94T7xD9EAUKbg33UASEF9oQiS/opnKH9WJx50S95oPUKwU7JWhsQoQtXW4p0bWVZZa0v4VyQTuoQjYDUrc0/8dDGye9P3ylyI461fkvbj57KqruLq3TlQO7WqbNuGKZP+pOeHuKrFOEs8kN1DkxjK37WC3/gCycoQ+',
                '/Times-Italic' : b'eJxVkDEOwjAMReN4gxGUjRtkZu8BcgWWnKIH6M6I1LEIVg7QUyDEGZg6ZUCd+HbS0upLiWM//8SxkWsbV2qh5+L8pCQr9jCrB+Ot5yrT/JD6LKEjJfT53DnlwUfpyxICCuBaSrxBfLMHRBfNO9rRGWqggPhEvdwsvLzBjPZbHBKoYDvNOxCTEtdcwz1XGtQmPpl7vltr6ldWt1BPznYgnMwuZKEXwsuD7oMdyNOejiW/FTeuzGhG/qhL7riW2fUfzZsaTBF0xpSdyJlRp1dXeAZ+Ifare/H//ylz9w8xy4v0',
                '/Times-BoldItalic' : b'eJxlkLsNg0AMhs++jgFYIjcBFRnAG0Q0TJEB6NPfAGkzAFNQZwCkSFRXUeW3HR5S9Ik7nx+/bTjHF+c/prNNRc9Y0bCjnp77eEcmgMZ0orf6BHqzNzLyrc6hYgjU0AP6hRtuYLXmr8ObbkAU2B2NsPQr5ln581Mo9CDRSvhrZGyUOMdZVRHRqYUTXkkrwtN7W8z04BerP8AMmKijGrMJd9zitSs4mFzsXnihRBUdMajFa1jDGmffy/IuR9S6DdhC/J+6EvZe9XRV6InuQIl38Bo56+lbevUXENqI9A==',
                '/Symbol' : b'eJxdT6FOA0EU3JlXU+SJM9Xr8LVFVjSpX4EoEt+AwuBJ7X1A3QlMDR/QTVBnz2BIMGxyySUkxTC7oQkhk9037+17s/PYsWHHG250Jwa2WIm1dsfOnNi3OZvZjJdcq+edz7zlIzc259YuzGGJpdX0k2pSMVhvvTSCfZwZRnekR8KbTvoTV3goOMdX7uj1l+OA2p3Ea6vxhMSB3p0wMkp1jgO9EOTSY8xwezSMpTowomKbkZ2VWunWe5CSd19c5LnSHzkVazOXzn8/ya7kYseppouKNo4FLV5+NWL2KN6gVh6yNyll94N4xMGuNb/FaJXdZ6/aLnJhPT6VpR+pg2TA',
                '/ZapfDingbats' : b'eJxtkL1LglEUxt/7/BbHPvzIJpegrTahD8SIoMWloTmcbKylIRpFGnLKIQWjcKkhlDCXoCiIHHSosT8gpBpqeN+KoGNr8oNz7n2ewznnXjo6V0VPSijBHkWVFRgbXKmpFUJqECPKKeH/aGKQyupAtThQrbHDEhUC17ZYosqhUWeBHHOm7HPEseuyySIdsmqxroAGa9yRspqqPmhT0ov5sGXZIlmuyfRr6ehLvvtxB5Y/Nauavu11vt4Z4fFvgyhxxpkmSYRlkuqStluKsp71oFfzx3RhXgjPTr5udKsTXTLDlHszPWLKmVqqW8dR3RPTJMP2ZzmbZ7PZNsIMKe6a3q6LuGp/G6/n9VxeaZd3YWPeyBgFV1DFRX8BrA5Wog=='
            }
            widthsDict['/Helvetica-Oblique'] = widthsDict['/Helvetica']
            widthsDict['/Helvetica-BoldOblique'] = widthsDict['/Helvetica-Bold']
            d = zlib.decompress(base64.b64decode(widthsDict[standard_fontname]))
            widthsArray = struct.unpack('>' + 'H'*(len(d)//2), d)

        # Return the names and widths vectors zipped up into a map
        return dict(zip(names,widthsArray))

