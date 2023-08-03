#!/usr/bin/env python3

import sys, os
from io import BytesIO
from PIL import Image, ImageCms

if __name__ == '__main__':

    helpMessage='''
    icc.py -- Convert an image to a specified color profile

    Usage:

    icc.py [-h|--help] -- get help    
    icc.py image.jpg -- extracts an ICC profile from the image; result is written to image.icc
    icc.py image.jpg (profile.icc|sRGB|LAB|XYZ) [mode] -- convert the image to the specified ICC profile

    Unless mode is specified, the the mode of the output is the same as that of the input (e.g. RGB->RGB)

    '''

    imagePath = sys.argv[1] if len(sys.argv) >= 2 else None
    profilePath = sys.argv[2] if len(sys.argv) >= 3 else None
    outputMode = sys.argv[3] if len(sys.argv) >= 4 else None

    if imagePath in [None, '-h', '--help']:
        print(helpMessage); sys.exit()

    image = Image.open(imagePath)

    # Input profile
    icc_profile = image.info.get('icc_profile')
    inputProfile = ImageCms.getOpenProfile(BytesIO(icc_profile)) if icc_profile != None else None

    # Extract profile
    if profilePath == None:
        if inputProfile == None:
            sys.exit('Image has no associated ICC profile')
        print(f'[Profile] {ImageCms.getProfileDescription(inputProfile).strip()}')
        inputProfilePath = os.path.splitext(imagePath)[0]+'.icc'
        with open(inputProfilePath, 'wb') as f:
            print(f'[Output] {inputProfilePath}')
            f.write(icc_profile)

    # Convert to profile
    else:
        # Input profile
        if inputProfile == None:
            if image.mode == 'RGB':
                print('Assuming input profile to be sRGB')
                inputProfile = ImageCms.createProfile("sRGB")
            else:
                sys.exit('Cannot determine image\'s ICC profile')
        print(f'[Input Profile] {ImageCms.getProfileDescription(inputProfile).strip()}')

        # Output profile
        outputProfile = ImageCms.createProfile(profilePath) if profilePath in ['sRGB', 'LAB', 'XYZ'] \
            else ImageCms.getOpenProfile(profilePath)
        print(f'[Output Profile] {ImageCms.getProfileDescription(outputProfile).strip()}')

        # Convert image
        if outputMode == None: outputMode = image.mode
        print(f'[Output Mode] {outputMode}')
        try:
            outputImage = ImageCms.profileToProfile(image, inputProfile, outputProfile, outputMode=outputMode)
        except ImageCms.PyCMSError:
            sys.exit(f'error: output mode {outputMode} not supported by the output profile')

        # Save the result
        outputImagePath = '-transformed'.join(os.path.splitext(imagePath))
        print(f'[Output] {outputImagePath}')
        outputImage.save(outputImagePath,icc_profile = outputProfile.tobytes())
        
