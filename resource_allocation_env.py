import numpy as np
import matplotlib.pyplot as plt

# 参数：真实频率与采样率
f_true = 9.0   # Hz（高于 Nyquist=fs/2=5 Hz）
fs = 10.0      # Hz
T = 2.0        # 显示 2 秒

# 高分辨率“连续”曲线（近似连续时间）
fs_hi = 2000.0
t_hi = np.linspace(0, T, int(fs_hi*T), endpoint=False)
x_hi = np.sin(2*np.pi*f_true*t_hi)

# 采样点（离散时间）
t_s = np.arange(0, T, 1/fs)
x_s = np.sin(2*np.pi*f_true*t_s)

# 计算混叠后的等效低频（折叠到 [-fs/2, fs/2]）
def alias_freq(f, fs):
    return abs(((f + fs/2) % fs) - fs/2)

f_alias = alias_freq(f_true, fs)  # 这里应得到 1 Hz
x_alias_cont = np.sin(2*np.pi*f_alias*t_hi)

# ========== 图1：时域 ==========
plt.figure(figsize=(8,4))
plt.plot(t_hi, x_hi, color='#1f77b4', alpha=0.35, label=f'原信号 {f_true:.0f} Hz')
plt.scatter(t_s, x_s, color='red', zorder=3, label=f'采样点（fs={fs:.0f} Hz）')
plt.plot(t_hi, x_alias_cont, 'k--', label=f'混叠等效 {f_alias:.0f} Hz')
plt.title('时域：高频采样不足产生混叠为低频')
plt.xlabel('时间 (s)'); plt.ylabel('幅度'); plt.legend(); plt.tight_layout()

# ========== 图2：频域 ==========
# 原信号（高采样率近似连续）的幅度谱
X_hi = np.fft.rfft(x_hi)
F_hi = np.fft.rfftfreq(x_hi.size, d=1/fs_hi)
mag_hi = np.abs(X_hi)
mag_hi = mag_hi / (mag_hi.max() + 1e-12)  # 归一化到 [0,1]

# 采样信号（fs=10 Hz）的幅度谱
X_s = np.fft.rfft(x_s)
F_s = np.fft.rfftfreq(x_s.size, d=1/fs)
mag_s = np.abs(X_s)
mag_s = mag_s / (mag_s.max() + 1e-12)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(F_hi, mag_hi, color='#1f77b4')
plt.axvline(f_true, color='orange', linestyle='--', label=f'{f_true:.0f} Hz')
plt.title('frequency Domain (High-Res Signal)')
plt.xlabel('Frequency (Hz)'); plt.ylabel('Normalized Amplitude'); plt.xlim(0, 20); plt.legend()

plt.subplot(1,2,2)
plt.stem(F_s, mag_s, basefmt=" ", linefmt='C3-', markerfmt='C3o')
plt.axvline(fs/2, color='gray', linestyle='--', label=f'Nyquist {fs/2:.0f} Hz')
plt.axvline(f_alias, color='k', linestyle='--', label=f'alias {f_alias:.0f} Hz')
plt.title('frequency Domain (Sampled Signal)')
plt.xlabel('Frequency (Hz)'); plt.ylabel('Normalized Amplitude'); plt.xlim(0, fs/2); plt.legend()
plt.tight_layout()

plt.show()