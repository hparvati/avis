from ginilib.FaceUtil import faces
from ginilib.LPRS import vehicles

#UC1 Detect and Add human faces and track the records
'''
fc=faces()
fc.add_User()
fc.detect()
'''

#UC2 Detect and add Number plates
vh=vehicles()
vh.detect_plate()


