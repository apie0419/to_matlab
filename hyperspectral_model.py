import numpy as np
import matplotlib.pyplot as plt
import easygui, os, colour
from PIL          import Image
from utils        import *
from numpy.linalg import inv, pinv


base_path   = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(base_path, "Source Data")
output_path = os.path.join(base_path, "Output Data")

np.set_printoptions(suppress=True)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

R1_choices = ['單色塊影像(可圈選範圍)','單色塊影像','已儲存的txt數據']

R1_choice = easygui.buttonbox(title="24色塊影像類型", choices=R1_choices)
if R1_choice == None:
    exit()

R1 = R1_choices.index(R1_choice)
R2 = 1

if R1 == 0:
    filename = easygui.fileopenbox("選擇圖檔中有多種色塊", default="*.*", multiple=True)
    CameraRGB = np.zeros((3, len(filename)))

    for i in range(len(filename)):
        color_picture = np.array(Image.open(filename[i]), dtype=np.uint8)
        position = imrect(filename[i])

        CameraRGB[:, i] = np.median(np.reshape(color_picture[position[0]:position[1], position[2]:position[3], :], (-1, 3)), axis=0)


elif R1 == 1:
    R2_choices = ['取平均','取中位數']
    easygui.msgbox(title="告知說明", msg='選擇處理方式：不知道要選甚麼就選"取平均"，若色塊顏色不均勻則選"取中位數"')
    R2_choice = easygui.buttonbox(title="RGB處理方式", choices=R2_choices)
    if R2_choice == None:
        R2 = 0
    else:
        R2 = R2_choices.index(R2_choice)

    filename = easygui.fileopenbox("選擇色塊影像", default="*.*", multiple=True)
    CameraRGB = np.zeros((3, len(filename)))
    for i in range(len(filename)):
        f = filename[i]
        color_picture = np.array(Image.open(f), dtype=np.uint8)
        if R2 == 0:
            CameraRGB[:, i] = np.mean(np.reshape(color_picture, (-1, 3)), axis=0)
        elif R2 == 1:
            CameraRGB[:, i] = np.median(np.reshape(color_picture, (-1, 3)), axis=0)

elif R1 == 2:
    filename = easygui.fileopenbox("讀取24色塊RGB txt", default="*.txt", filetypes=["*.txt"])
    CameraRGB = np.loadtxt(filename, delimiter="\t").astype(np.uint8)


CameraRGB_3D = np.reshape(CameraRGB.T, (24, 1, 3))
A_lin = rgb2lin(CameraRGB_3D)

d65 = np.array([0.95047, 1.00000000, 1.08883])
d50 = np.array([0.96422, 1.00000, 0.82521])
B_lin = colour.chromatic_adaptation_VonKries(A_lin, d65, d50, transform="Bradford").astype(np.uint8) # 結果有誤差
CameraRGB = np.reshape(lin2rgb(B_lin), (-1, 3)).astype(np.double).T
CameraXYZ = np.round(np.transpose(colour.sRGB_to_XYZ(CameraRGB.T / 255.0, illuminant=colour.XYZ_to_xy(d65)) * 100), 4)

filename = easygui.fileopenbox("讀取24色塊頻譜 txt", default="*.txt", filetypes=["*.txt"])
color_Rspectrum = np.loadtxt(filename, delimiter="\t").astype(np.double)
filename = easygui.fileopenbox("讀取光源頻譜 txt", default="*.txt", filetypes=["*.txt"])
light = np.reshape(np.loadtxt(filename, delimiter="\n").astype(np.double), (-1, 1))
CMF = np.loadtxt(os.path.join(source_path, "CMF.txt"), delimiter="\t").astype(np.double)
light_k = round(100 / np.sum(np.matmul(CMF[:,1], light[:,0])), 4)
spectrumXYZ = light_k * np.matmul((color_Rspectrum * light * np.ones((1, 24))).T, CMF).T

Ma = np.array([
    [0.40024, 0.70760, -0.08081],
    [-0.2263, 1.16532, 0.04570],
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

coeff, scores, latent, explained = pca(color_Rspectrum.T, centered=False)
EV = coeff[:, :12]
alpha = scores[:, :12].T

fig = plt.figure(num=f"主成分比重: {np.sum(explained[:12])}%")

for i in range(12):
    ax = fig.add_subplot(3, 4, i+1)
    ax.plot(coeff[:, i])
    percent = round(explained[i], 4)
    ax.set(title=f"第{i+1}主成分({percent}%)")
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

R3_choices = ['各色塊頻譜比較','全頻譜比較','色差圖示比較']

R3_choice = easygui.buttonbox(title="頻譜選取", choices=R3_choices)
if R3_choice == None:
    R3 = -1
else:
    R3 = R3_choices.index(R3_choice)

if R3 == 0:
    
    for i in range(24):
        fig, ax = plt.subplots(num=f"Figure {i+1}")
        x = list(range(380, 380+len(simulate_spectrum[:, i])))
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
    ax[0].set_xlim(380, 780)
    ax[0].grid()

    ax[1].plot(color_Rspectrum)
    ax[1].set_title("量測頻譜")
    ax[1].set_xlabel("wavelength(nm)", fontsize=12)
    ax[1].set_ylabel("Reflectivity(a.u.)", fontsize=12)
    ax[1].set_xlim(380, 780)
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
