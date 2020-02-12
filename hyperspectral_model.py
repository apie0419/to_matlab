import numpy as np
import matplotlib.pyplot as plt
import easygui, os, colour, numpy.matlib
from utils        import *
from numpy.linalg import inv, pinv


base_path   = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(base_path, "Source Data")
output_path = os.path.join(base_path, "Output Data")

np.set_printoptions(suppress=True)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

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

C_CIE2000 = colour.delta_E_CIE2000(Lab1, Lab2).T
C_AvgCIE2000 = np.mean(C_CIE2000)

i = 24
RMSE_XYZ = np.sqrt(np.sum(CorrectXYZ[:, :i] - spectrumXYZ[:, :i]) ** 2 / 3)
RMSE_allXYZ = np.sum(RMSE_XYZ / 24)

coeff, scores, latent, explained = pca(color_Rspectrum.T)
EV = coeff[:, :12]
alpha = scores[:, :12].T

fig = plt.figure(num=f"主成分比重: {np.sum(explained[:12])}%")

for i in range(12):
    ax = fig.add_subplot(3, 4, i+1)
    ax.plot(coeff[:, i])
    ax.set(title=f"第{i+1}主成分({round(explained[i], 4)}%)")
    ax.grid()

plt.subplots_adjust(hspace=0.5, wspace=0.2)
plt.show()

spec_extend = np.array([
    spectrumXYZ[0], spectrumXYZ[1], spectrumXYZ[2],
    spectrumXYZ[0, :] * spectrumXYZ[1, :],
    spectrumXYZ[1, :] * spectrumXYZ[2, :],
    spectrumXYZ[2, :] * spectrumXYZ[0, :],
    spectrumXYZ[0, :] * spectrumXYZ[1, :] * spectrumXYZ[2, :]
])

M = alpha @ pinv(spec_extend)

correct_extend = np.array([
    CorrectXYZ[0], CorrectXYZ[1], CorrectXYZ[2],
    CorrectXYZ[0, :] * CorrectXYZ[1, :],
    CorrectXYZ[1, :] * CorrectXYZ[2, :],
    CorrectXYZ[2, :] * CorrectXYZ[0, :],
    CorrectXYZ[0, :] * CorrectXYZ[1, :] * spectrumXYZ[2, :]
])

simulate_spectrum = EV @ M @ correct_extend

simulate_spectrumXYZ = light_k * ((simulate_spectrum * (light @ np.ones((1, 24)))).T @ CMF).T

i = 24

RMSE_spectrum = np.sqrt(np.sum((simulate_spectrum[:,:i] - color_Rspectrum[:, :i]) ** 2) / 401)
RMSE_allspectrum = np.sum(RMSE_spectrum / 24)

Lab1 = colour.XYZ_to_Lab(spectrumXYZ.T / 100)
Lab2 = colour.XYZ_to_Lab(simulate_spectrumXYZ.T / 100)

LabRGB1 = (colour.XYZ_to_sRGB((inv(Ma) @ np.diag(np.diag((Ma @ d65) / (Ma @ White_light))) @ Ma @ spectrumXYZ).T / 100) * 255).astype(np.uint8)
LabRGB2 = (colour.XYZ_to_sRGB((inv(Ma) @ np.diag(np.diag((Ma @ d65) / (Ma @ White_light))) @ Ma @ simulate_spectrumXYZ).T / 100) * 255).astype(np.uint8)
Total_CIE76 = np.sqrt(np.sum((Lab1 - Lab2) ** 2, 1))
Total_AvgCIE76 = np.mean(Total_CIE76)
Total_CIE2000 = colour.delta_E_CIE2000(Lab1, Lab2).T
Total_AvgCIE2000 = np.mean(Total_CIE2000)

R3 = 2

if R3 == 0:
    
    for i in range(24):
        fig, ax = plt.subplots(num=f"Figure {i+1}")
        x = list(range(len(simulate_spectrum[:, i])))
        ax.plot(x, simulate_spectrum[:, i], label="模擬頻譜")
        ax.plot(x, color_Rspectrum[:, i], label="量測頻譜")
        ax.grid()
        ax.legend()
        plt.show()

elif R3 == 1:
    
    fig, ax = plt.subplots(nrows=1, ncols=2, num="Position")
    ax[0].plot(simulate_spectrum)
    ax[0].set_title("模擬頻譜")
    ax[0].set_xlabel("wavelength(nm)", fontsize=12)
    ax[0].set_ylabel("Reflectivity(a.u.)", fontsize=12)
    ax[0].grid()

    ax[1].plot(color_Rspectrum)
    ax[1].set_title("量測頻譜")
    ax[1].set_xlabel("wavelength(nm)", fontsize=12)
    ax[1].set_ylabel("Reflectivity(a.u.)", fontsize=12)
    ax[1].grid()

    plt.show()

elif R3 == 2:
    
    L1 = np.tile(np.reshape(LabRGB2, (24, 1, 3)), [1, 5, 1])
    L2 = np.tile(np.ones((24, 1, 3)) * 255, [1, 1])
    L3 = np.tile(np.reshape(LabRGB2, (24, 1, 3)), [1, 5, 1])

if not os.path.exists(output_path):
    os.mkdir(output_path)

np.savetxt(os.path.join(output_path, "C.txt"), C, delimiter="\t")
np.savetxt(os.path.join(output_path, "M.txt"), M, delimiter="\t")
np.savetxt(os.path.join(output_path, "EV.txt"), EV, delimiter="\t")
np.savetxt(os.path.join(output_path, "light.txt"), light, delimiter="\t")
np.savetxt(os.path.join(output_path, "White_light.txt"), White_light, delimiter="\t")
