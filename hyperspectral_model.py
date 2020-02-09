import numpy as np
import easygui, os, colour
from utils import *

base_path   = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(base_path, "Source Data")
output_path = os.path.join(base_path, "Output Data")

np.set_printoptions(suppress=True)

R1 = 3
R2 = 1
if R1 == 0:
    pass
elif R1 == 1:
    pass
elif R1 == 2:
    pass
elif R1 == 3:
    filename = easygui.fileopenbox("讀取24色塊RGB txt", default="*.txt", filetypes=["*.txt"])
    CameraRGB = load(filename).astype(np.uint8)

CameraRGB_3D = np.reshape(np.transpose(CameraRGB), (24, 1, 3))
A_lin = rgb2lin(CameraRGB_3D)

d65 = np.array([0.950470, 1.0000, 1.088830])
XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
B_lin = colour.chromatic_adaptation_VonKries(A_lin, d65, XYZ_wr).astype(np.uint8)
CameraRGB = np.transpose(np.reshape(lin2rgb(B_lin), (24, 3)).astype(np.double))
CameraXYZ = np.round(np.transpose(colour.sRGB_to_XYZ(CameraRGB.T / 255.0, illuminant=colour.XYZ_to_xy(d65)) * 100), 4)

filename = easygui.fileopenbox("讀取24色塊頻譜 txt", default="*.txt", filetypes=["*.txt"])
color_Rspectrum = load(filename).astype(np.double)
filename = easygui.fileopenbox("讀取光源頻譜 txt", default="*.txt", filetypes=["*.txt"])
light = load(filename).astype(np.double)
CMF = load(os.path.join(source_path, "CMF.txt")).astype(np.double)
light_k = round(100 / np.sum(np.matmul(CMF[:,1], light[:,0])), 4)
spectrumXYZ = light_k * np.matmul((color_Rspectrum * light * np.ones((1, 24))).T, CMF).T
