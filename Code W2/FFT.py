import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import fft

fs, wave_data = wavfile.read("/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Audio/Alarm01.wav")
fft_data = abs(fft(wave_data[:,0]))
fft_data = fft_data/2**15
num_frame = len(wave_data)   
n_channel = int(wave_data.size/ num_frame)  

n0=int(np.ceil(num_frame/2))
fft_data1=np.concatenate([fft_data[n0:num_frame],fft_data[0:n0]])

freq=np.concatenate([range(n0-num_frame,0),range(0,n0)])*fs/num_frame

plt.plot(freq,fft_data1)
plt.xlim([-1000,1000])    # 限制頻率的顯示範圍
plt.show()  # 如後圖
