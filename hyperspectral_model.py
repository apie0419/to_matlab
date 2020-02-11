import numpy as np
import easygui, os, colour
from utils        import *
from numpy.linalg import inv, pinv

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
d50 = np.array([0.96429568, 1.00000000, 0.82510460])
B_lin = colour.chromatic_adaptation_VonKries(A_lin, d65, d50).astype(np.uint8) # 結果有誤差
CameraRGB = np.transpose(np.reshape(lin2rgb(B_lin), (24, 3)).astype(np.double))
CameraXYZ = np.round(np.transpose(colour.sRGB_to_XYZ(CameraRGB.T / 255.0, illuminant=colour.XYZ_to_xy(d65)) * 100), 4)

filename = easygui.fileopenbox("讀取24色塊頻譜 txt", default="*.txt", filetypes=["*.txt"])
color_Rspectrum = load(filename).astype(np.double)
filename = easygui.fileopenbox("讀取光源頻譜 txt", default="*.txt", filetypes=["*.txt"])
light = load(filename).astype(np.double)
CMF = load(os.path.join(source_path, "CMF.txt")).astype(np.double)
light_k = round(100 / np.sum(np.matmul(CMF[:,1], light[:,0])), 4)
spectrumXYZ = light_k * np.matmul((color_Rspectrum * light * np.ones((1, 24))).T, CMF).T

Ma = np.array([
    [0.40024, 0.70760, -0.08081],
    [-0.22603, 1.16532, 0.04570],
    [0, 0, 0.91822]
])

White_light = light_k * (light.T @ CMF).T / 100
CameraXYZ = inv(Ma) @ np.round(np.diag(np.diag((Ma @ White_light) / (Ma @ d65))), 4) @ Ma @ CameraXYZ
CameraXYZ_2 = np.power(CameraXYZ, 2)
CameraXYZ_3 = np.power(CameraXYZ, 3)

extend = np.array([
    np.ones(24),
    CameraXYZ[0], CameraXYZ[1], CameraXYZ[2],
    CameraXYZ[0, :] * CameraXYZ[1, :],
    CameraXYZ[1, :] * CameraXYZ[2, :],
    CameraXYZ[2, :] * CameraXYZ[0, :],
    CameraXYZ_2[0], CameraXYZ_2[1], CameraXYZ_2[2],
    CameraXYZ[0, :] * CameraXYZ[1, :] * CameraXYZ[2, :],
    CameraXYZ_3[0], CameraXYZ_3[1], CameraXYZ_3[2]
])

C = spectrumXYZ @ pinv(extend)
CorrectXYZ = C @ extend

Lab1 = colour.XYZ_to_Lab(CorrectXYZ.T, illuminant=colour.XYZ_to_xy(d65)) / 100
LabRGB11 = (colour.XYZ_to_sRGB(CameraXYZ.T / 100) * 255).astype(np.uint8)
Lab2 = colour.XYZ_to_Lab(spectrumXYZ.T / 100)
LabRGB21 = (colour.XYZ_to_sRGB(spectrumXYZ.T / 100) * 255).astype(np.uint8)
C_CIE76= np.sqrt(np.sum((Lab1 - Lab2) ** 2, 1))
C_AvgCIE76 = np.mean(C_CIE76)

RMSE_XYZ = np.sqrt(np.sum(CorrectXYZ - spectrumXYZ) ** 2 / 3)
RMSE_allXYZ = np.sum(RMSE_XYZ / 24)

pca(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32).T) # 不一樣
# print (coeff)

# coeff, score = pca(color_Rspectrum.T)
# EV = coeff[:, :11]
# alpha = score[:, :11]