from scipy.signal import butter, lfilter, resample,iirnotch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


fs=300
#b, a = iirnotch(2 * w0/fs, Q)

band=[8.0, 30.0]
B, A = butter(5, np.array(band) / (fs / 2), btype='bandpass')
freq, h = signal.freqz(B, A, fs=fs)



fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
ax[0].set_title("Frequency Response")
ax[0].set_ylabel("Amplitude (dB)", color='blue')
ax[0].set_xlim([0, 50])
ax[0].set_ylim([-25, 10])
ax[0].grid()
# ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
# ax[1].set_ylabel("Angle (degrees)", color='green')
# ax[1].set_xlabel("Frequency (Hz)")
# ax[1].set_xlim([0, 50])
# ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
# ax[1].set_ylim([-90, 90])
# ax[1].grid()
plt.show()