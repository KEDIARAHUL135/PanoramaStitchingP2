import os
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

      

def ReadImage(ImageFolderPath):
    Images = []									# Input Images will be stored in this list.

	# Checking if path is of folder.
    if os.path.isdir(ImageFolderPath):                              # If path is of a folder contaning images.
        ImageNames = os.listdir(ImageFolderPath)
        ImageNames_Split = [[int(os.path.splitext(os.path.basename(ImageName))[0]), ImageName] for ImageName in ImageNames]
        ImageNames_Split = sorted(ImageNames_Split, key=lambda x:x[0])
        ImageNames_Sorted = [ImageNames_Split[i][1] for i in range(len(ImageNames_Split))]
        
        for i in range(len(ImageNames_Sorted)):                     # Getting all image's name present inside the folder.
            ImageName = ImageNames_Sorted[i]
            InputImage = cv2.imread(ImageFolderPath + "/" + ImageName)  # Reading images one by one.
            
            # Checking if image is read
            if InputImage is None:
                print("Not able to read image: {}".format(ImageName))
                extt(0)

            Images.append(InputImage)                               # Storing images.
            
    else:                                       # If it is not folder(Invalid Path).
        print("\nEnter valid Image Folder Path.\n")
        
    if len(Images) < 2:
        print("\nNot enough images found. Please provide 2 or more images.\n")
        extt(1)
    
    return Images

    
def FindMatches(BaseImage, SecImage):
    # Using SIFT to find the keypoints and decriptors in the images
    Sift = cv2.SIFT_create()
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)

    # Using Brute Force matcher to find matches.
    BF_Matcher = cv2.BFMatcher()
    InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)

    # Applytng ratio test and filtering out the good matches.
    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])

    return GoodMatches, BaseImage_kp, SecImage_kp



def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    # If less than 4 matches found, extt the code.
    if len(Matches) < 4:
        print("\nNot enough matches found between the images.\n")
        extt(0)

    # Storing coordinates of points corresponding to the matches found in both the images
    BaseImage_pts = []
    SecImage_pts = []
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match[0].querytdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)

    # Changing the datatype to "float32" for finding homography
    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

    # Finding the homography matrix(transformation matrix).
    (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

    return HomographyMatrix, Status

    
def GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape):
    # Reading the size of the image
    (Height, Width) = Sec_ImageShape
    
    # Taking the matrix of initial coordinates of the corners of the secondary image
    # Stored in the following format: [[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]]
    # Where (xt, yt) is the coordinate of the i th corner of the image. 
    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])
    
    # Finding the final coordinates of the corners of the image after transformation.
    # NOTE: Here, the coordinates of the corners of the frame may go out of the 
    # frame(negative values). We will correct this afterwards by updating the 
    # homography matrix accordingly.
    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)

    [x, y, c] = FinalMatrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    # Finding the dimentions of the stitched image frame and the "Correction" factor
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)
    
    # Again correcting New_Width and New_Height
    # Helpful when secondary image is overlaped on the left hand side of the Base image.
    if New_Width < Base_ImageShape[1] + Correction[0]:
        New_Width = Base_ImageShape[1] + Correction[0]
    if New_Height < Base_ImageShape[0] + Correction[1]:
        New_Height = Base_ImageShape[0] + Correction[1]

    # Finding the coordinates of the corners of the image if they all were within the frame.
    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    OldInitialPoints = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    NewFinalPonts = np.float32(np.array([x, y]).transpose())

    # Updating the homography matrix. Done so that now the secondary image completely 
    # lies inside the frame
    HomographyMatrix = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPonts)
    
    return [New_Height, New_Width], Correction, HomographyMatrix



def StitchImages(BaseImage, SecImage):
    # Finding matches between the 2 images and their keypoints
    Matches, BaseImage_kp, SecImage_kp = FindMatches(BaseImage, SecImage)
    
    # Finding homography matrix.
    HomographyMatrix, Status = FindHomography(Matches, BaseImage_kp, SecImage_kp)
    
    # Finding size of new frame of stitched images and updating the homography matrix 
    NewFrameSize, Correction, HomographyMatrix = GetNewFrameSizeAndMatrix(HomographyMatrix, SecImage.shape[:2], BaseImage.shape[:2])

    # Finally placing the images upon one another.
    StitchedImage = cv2.warpPerspective(SecImage, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
    StitchedImage[Correction[1]:Correction[1]+BaseImage.shape[0], Correction[0]:Correction[0]+BaseImage.shape[1]] = BaseImage

    return StitchedImage


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def ConvertPt(pt):
    global center, f

    xt = ( f * math.tan( (pt.x - center.x) / f ) ) + center.x
    yt = ( (pt.y - center.y) / math.cos( (pt.x - center.x) / f ) ) + center.y

    return Point(xt, yt)


def ProjectOntoCylinder(InitialImage):
    global w, h, center, f
    h, w = InitialImage.shape[:2]
    center = Point(w // 2, h // 2)
    f = 1000
    
    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)

    for j in range(h):
        for i in range(w):
            ti_pt = Point(i, j)
            ii_pt = ConvertPt(ti_pt)

            ii_tl_pt = Point(int(ii_pt.x), int(ii_pt.y))
            print((ti_pt.x, ti_pt.y), (ii_tl_pt.x, ii_tl_pt.y))
            
            if ii_tl_pt.x < 0 or \
               ii_tl_pt.x > w - 2 or \
               ii_tl_pt.y < 0 or \
               ii_tl_pt.y > h - 2:
                continue

            # Bilinear interpolation
            dx = ii_pt.x - ii_tl_pt.x
            dy = ii_pt.y - ii_tl_pt.y

            weight_tl = (1.0 - dx) * (1.0 - dy)
            weight_tr = (dx)       * (1.0 - dy)
            weight_bl = (1.0 - dx) * (dy)
            weight_br = (dx)       * (dy)

            for k in range(InitialImage.shape[2]):
                color_val = ( weight_tl * InitialImage[ii_tl_pt.y][ii_tl_pt.x][k] ) + \
                            ( weight_tr * InitialImage[ii_tl_pt.y][ii_tl_pt.x + 1][k] ) + \
                            ( weight_bl * InitialImage[ii_tl_pt.y + 1][ii_tl_pt.x][k] ) + \
                            ( weight_br * InitialImage[ii_tl_pt.y + 1][ii_tl_pt.x + 1][k] )

                TransformedImage[ti_pt.y][ti_pt.x][k] = int(color_val)
            
            #print(i, j)
        print("\n\n\n\n")
        
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(InitialImage, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(TransformedImage, cv2.COLOR_BGR2RGB))
    plt.show()

    return TransformedImage


if __name__ == "__main__":
    # Reading images.
    Images = ReadImage("InputImages/Field")
    
    Cyl_Images = []
    for Image in Images:
        Cyl_Images.append(ProjectOntoCylinder(Image))

    BaseImage = Cyl_Images[0]
    for i in range(1, len(Cyl_Images)):
        StitchedImage = StitchImages(BaseImage, Cyl_Images[i])

        plt.imshow(cv2.cvtColor(StitchedImage, cv2.COLOR_BGR2RGB))
        plt.show()
        
        BaseImage = StitchedImage.copy()    
