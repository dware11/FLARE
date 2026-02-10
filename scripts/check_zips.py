import zipfile
paths = [r'D:\FLARE_DATA\ct_brain\raw_zips\qure.headct.study\CQ500-CT-102.zip', r'D:\FLARE_DATA\ct_brain\raw_zips\qure.headct.study\CQ500-CT-103.zip']
for p in paths:
    print('\nChecking', p)
    try:
        print('is_zipfile:', zipfile.is_zipfile(p))
        with zipfile.ZipFile(p, 'r') as zf:
            bad = zf.testzip()
            print('testzip result (first bad file or None):', bad)
            print('filecount:', len(zf.namelist()))
    except Exception as e:
        print('EXCEPTION:', type(e).__name__, e)
