from python_speech_features import *
from matplotlib import pyplot as plt

# 对音频信号处理程序
# 本程序主要有四个函数，它们分别是：
#    audio2frame:将音频转换成帧矩阵
#    deframesignal:对每一帧做一个消除关联的变换
#    spectrum_magnitude:计算每一帧傅立叶变换以后的幅度
#    spectrum_power:计算每一帧傅立叶变换以后的功率谱
#    log_spectrum_power:计算每一帧傅立叶变换以后的对数功率谱
#    pre_emphasis:对原始信号进行预加重处理
import numpy
import math
from scipy.fftpack import dct
import scipy.io.wavfile as wavfile

def audio2frame(signal, frame_length, frame_step, winfunc=lambda x: numpy.ones((x,))):
    '''将音频信号转化为帧。
	参数含义：
	signal:原始音频型号
	frame_length:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
	frame_step:相邻帧的间隔（同上定义）
	winfunc:lambda函数，用于生成一个向量
    '''
    signal_length = len(signal)  # 信号总长度
    frame_length = int(round(frame_length))  # 以帧帧时间长度
    frame_step = int(round(frame_step))  # 相邻帧之间的步长
    if signal_length <= frame_length:  # 若信号长度小于一个帧的长度，则帧数定义为1
        frames_num = 1
    else:  # 否则，计算帧的总长度
        frames_num = 1 + int(math.ceil((1.0 * signal_length - frame_length) / frame_step))
    pad_length = int((frames_num - 1) * frame_step + frame_length)  # 所有帧加起来总的铺平后的长度
    zeros = numpy.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = numpy.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
    indices = numpy.tile(numpy.arange(0, frame_length), (frames_num, 1)) + numpy.tile(
        numpy.arange(0, frames_num * frame_step, frame_step),
        (frame_length, 1)).T  # 相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices = numpy.array(indices, dtype=numpy.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    win = numpy.tile(winfunc(frame_length), (frames_num, 1))  # window窗函数，这里默认取1
    return frames * win  # 返回帧信号矩阵


def deframesignal(frames, signal_length, frame_length, frame_step, winfunc=lambda x: numpy.ones((x,))):
    '''定义函数对原信号的每一帧进行变换，应该是为了消除关联性
    参数定义：
    frames:audio2frame函数返回的帧矩阵
    signal_length:信号长度
    frame_length:帧长度
    frame_step:帧间隔
    winfunc:对每一帧加window函数进行分析，默认此处不加window
    '''
    # 对参数进行取整操作
    signal_length = round(signal_length)  # 信号的长度
    frame_length = round(frame_length)  # 帧的长度
    frames_num = numpy.shape(frames)[0]  # 帧的总数
    assert numpy.shape(frames)[1] == frame_length, '"frames"矩阵大小不正确，它的列数应该等于一帧长度'  # 判断frames维度
    indices = numpy.tile(numpy.arange(0, frame_length), (frames_num, 1)) + numpy.tile(
        numpy.arange(0, frames_num * frame_step, frame_step),
        (frame_length, 1)).T  # 相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices = numpy.array(indices, dtype=numpy.int32)
    pad_length = (frames_num - 1) * frame_step + frame_length  # 铺平后的所有信号
    if signal_length <= 0:
        signal_length = pad_length
    recalc_signal = numpy.zeros((pad_length,))  # 调整后的信号
    window_correction = numpy.zeros((pad_length, 1))  # 窗关联
    win = winfunc(frame_length)
    for i in range(0, frames_num):
        window_correction[indices[i, :]] = window_correction[indices[i, :]] + win + 1e-15  # 表示信号的重叠程度
        recalc_signal[indices[i, :]] = recalc_signal[indices[i, :]] + frames[i, :]  # 原信号加上重叠程度构成调整后的信号
    recalc_signal = recalc_signal / window_correction  # 新的调整后的信号等于调整信号处以每处的重叠程度
    return recalc_signal[0:signal_length]  # 返回该新的调整信号


def spectrum_magnitude(frames, NFFT):
    '''计算每一帧经过FFY变幻以后的频谱的幅度，若frames的大小为N*L,则返回矩阵的大小为N*NFFT
    参数说明：
    frames:即audio2frame函数中的返回值矩阵，帧矩阵
    NFFT:FFT变换的数组大小,如果帧长度小于NFFT，则帧的其余部分用0填充铺满
    '''
    complex_spectrum = numpy.fft.rfft(frames, NFFT)  # 对frames进行FFT变换
    return numpy.absolute(complex_spectrum)  # 返回频谱的幅度值


def spectrum_power(frames, NFFT):
    '''计算每一帧傅立叶变换以后的功率谱
    参数说明：
    frames:audio2frame函数计算出来的帧矩阵
    NFFT:FFT的大小
    '''
    return 1.0 / NFFT * numpy.square(spectrum_magnitude(frames, NFFT))  # 功率谱等于每一点的幅度平方/NFFT


def log_spectrum_power(frames, NFFT, norm=1):
    '''计算每一帧的功率谱的对数形式
    参数说明：
    frames:帧矩阵，即audio2frame返回的矩阵
    NFFT：FFT变换的大小
    norm:范数，即归一化系数
    '''
    spec_power = spectrum_power(frames, NFFT)
    spec_power[spec_power < 1e-30] = 1e-30  # 为了防止出现功率谱等于0，因为0无法取对数
    log_spec_power = 10 * numpy.log10(spec_power)
    if norm:
        return log_spec_power - numpy.max(log_spec_power)
    else:
        return log_spec_power


def pre_emphasis(signal, coefficient=0.95):
    '''对信号进行预加重
    参数含义：
    signal:原始信号
    coefficient:加重系数，默认为0.95
    '''
    return numpy.append(signal[0], signal[1:] - coefficient * signal[:-1])




def calcMFCC_delta_delta(signal, samplerate=16000, win_length=0.025, win_step=0.01, cep_num=13, filters_num=26,
                         NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97, cep_lifter=22,
                         appendEnergy=True):
    '''计算13个MFCC+13个一阶微分系数+13个加速系数,一共39个系数
    '''
    feat = calcMFCC(signal, samplerate, win_length, win_step, cep_num, filters_num, NFFT, low_freq, high_freq,
                    pre_emphasis_coeff, cep_lifter, appendEnergy)  # 首先获取13个一般MFCC系数
    result1 = derivate(feat)
    result2 = derivate(result1)
    result3 = numpy.concatenate((feat, result1), axis=1)
    result = numpy.concatenate((result3, result2), axis=1)
    return result


def calcMFCC_delta(signal, samplerate=16000, win_length=0.025, win_step=0.01, cep_num=13, filters_num=26, NFFT=512,
                   low_freq=0, high_freq=None, pre_emphasis_coeff=0.97, cep_lifter=22, appendEnergy=True):
    '''计算13个MFCC+13个一阶微分系数
    '''
    feat = calcMFCC(signal, samplerate, win_length, win_step, cep_num, filters_num, NFFT, low_freq, high_freq,
                    pre_emphasis_coeff, cep_lifter, appendEnergy)  # 首先获取13个一般MFCC系数
    result = derivate(feat)  # 调用derivate函数
    result = numpy.concatenate((feat, result), axis=1)
    return result


def derivate(feat, big_theta=2, cep_num=13):
    '''计算一阶系数或者加速系数的一般变换公式
    参数说明:
    feat:MFCC数组或者一阶系数数组
    big_theta:公式中的大theta，默认取2
    '''
    result = numpy.zeros(feat.shape)  # 结果
    denominator = 0  # 分母
    for theta in numpy.linspace(1, big_theta, big_theta):
        denominator = denominator + theta ** 2
    denominator = denominator * 2  # 计算得到分母的值
    for row in numpy.linspace(0, feat.shape[0] - 1, feat.shape[0]):
        tmp = numpy.zeros((cep_num,))
        numerator = numpy.zeros((cep_num,))  # 分子
        for t in numpy.linspace(1, cep_num, cep_num):
            t = int(t)
            a = 0
            b = 0
            s = 0
            for theta in numpy.linspace(1, big_theta, big_theta):
                if (t + theta) > cep_num:
                    a = 0
                else:
                    a = feat[int(row)][int(t + theta - 1)]
                if (t - theta) < 1:
                    b = 0
                else:
                    b = feat[int(row)][int(t - theta - 1)]
                s += theta * (a - b)
            numerator[int(t - 1)] = s
        tmp = numerator * 1.0 / denominator
        result[int(row)] = tmp
    return result


def calcMFCC(signal, samplerate=16000, win_length=0.025, win_step=0.01, cep_num=13, filters_num=26, NFFT=512,
             low_freq=0, high_freq=None, pre_emphasis_coeff=0.97, cep_lifter=22, appendEnergy=True):
    '''计算13个MFCC系数
    参数含义：
    signal:原始音频信号，一般为.wav格式文件
    samplerate:抽样频率，这里默认为16KHz
    win_length:窗长度，默认即一帧为25ms
    win_step:窗间隔，默认情况下即相邻帧开始时刻之间相隔10ms
    cep_num:倒谱系数的个数，默认为13
    filters_num:滤波器的个数，默认为26
    NFFT:傅立叶变换大小，默认为512
    low_freq:最低频率，默认为0
    high_freq:最高频率
    pre_emphasis_coeff:预加重系数，默认为0.97
    cep_lifter:倒谱的升个数？？
    appendEnergy:是否加上能量，默认加
    '''

    feat, energy = fbank(signal, samplerate, win_length, win_step, filters_num, NFFT, low_freq, high_freq,
                         pre_emphasis_coeff)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :cep_num]  # 进行离散余弦变换,只取前13个系数
    feat = lifter(feat, cep_lifter)
    if appendEnergy:
        feat[:, 0] = numpy.log(energy)  # 只取2-13个系数，第一个用能量的对数来代替
    return feat


def fbank(signal, samplerate=16000, win_length=0.025, win_step=0.01, filters_num=26, NFFT=512, low_freq=0,
          high_freq=None, pre_emphasis_coeff=0.97):
    '''计算音频信号的MFCC
    参数说明：
    samplerate:采样频率
    win_length:窗长度
    win_step:窗间隔
    filters_num:梅尔滤波器个数
    NFFT:FFT大小
    low_freq:最低频率
    high_freq:最高频率
    pre_emphasis_coeff:预加重系数
    '''

    high_freq = high_freq or samplerate / 2  # 计算音频样本的最大频率
    signal = pre_emphasis(signal, pre_emphasis_coeff)  # 对原始信号进行预加重处理
    frames = audio2frame(signal, win_length * samplerate, win_step * samplerate)  # 得到帧数组
    spec_power = spectrum_power(frames, NFFT)  # 得到每一帧FFT以后的能量谱
    energy = numpy.sum(spec_power, 1)  # 对每一帧的能量谱进行求和
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)  # 对能量为0的地方调整为eps，这样便于进行对数处理
    fb = get_filter_banks(filters_num, NFFT, samplerate, low_freq, high_freq)  # 获得每一个滤波器的频率宽度
    feat = numpy.dot(spec_power, fb.T)  # 对滤波器和能量谱进行点乘
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)  # 同样不能出现0
    return feat, energy


def log_fbank(signal, samplerate=16000, win_length=0.025, win_step=0.01, filters_num=26, NFFT=512, low_freq=0,
              high_freq=None, pre_emphasis_coeff=0.97):
    '''计算对数值
    参数含义：同上
    '''
    feat, energy = fbank(signal, samplerate, win_length, win_step, filters_num, NFFT, low_freq, high_freq,
                         pre_emphasis_coeff)
    return numpy.log(feat)


def ssc(signal, samplerate=16000, win_length=0.025, win_step=0.01, filters_num=26, NFFT=512, low_freq=0, high_freq=None,
        pre_emphasis_coeff=0.97):
    '''
    待补充
    '''
    high_freq = high_freq or samplerate / 2
    signal = pre_emphasis(signal, pre_emphasis_coeff)
    frames = audio2frame(signal, win_length * samplerate, win_step * samplerate)
    spec_power = spectrum_power(frames, NFFT)
    spec_power = numpy.where(spec_power == 0, numpy.finfo(float).eps, spec_power)  # 能量谱
    fb = get_filter_banks(filters_num, NFFT, samplerate, low_freq, high_freq)
    feat = numpy.dot(spec_power, fb.T)  # 计算能量
    R = numpy.tile(numpy.linspace(1, samplerate / 2, numpy.size(spec_power, 1)), (numpy.size(spec_power, 0), 1))
    return numpy.dot(spec_power * R, fb.T) / feat


def hz2mel(hz):
    '''把频率hz转化为梅尔频率
    参数说明：
    hz:频率
    '''
    return 2595 * numpy.log10(1 + hz / 700.0)


def mel2hz(mel):
    '''把梅尔频率转化为hz
    参数说明：
    mel:梅尔频率
    '''
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filter_banks(filters_num=20, NFFT=512, samplerate=16000, low_freq=0, high_freq=None):
    '''计算梅尔三角间距滤波器，该滤波器在第一个频率和第三个频率处为0，在第二个频率处为1
    参数说明：
    filers_num:滤波器个数
    NFFT:FFT大小
    samplerate:采样频率
    low_freq:最低频率
    high_freq:最高频率
    '''
    # 首先，将频率hz转化为梅尔频率，因为人耳分辨声音的大小与频率并非线性正比，所以化为梅尔频率再线性分隔
    low_mel = hz2mel(low_freq)
    high_mel = hz2mel(high_freq)
    # 需要在low_mel和high_mel之间等间距插入filters_num个点，一共filters_num+2个点
    mel_points = numpy.linspace(low_mel, high_mel, filters_num + 2)
    # 再将梅尔频率转化为hz频率，并且找到对应的hz位置
    hz_points = mel2hz(mel_points)
    # 我们现在需要知道这些hz_points对应到fft中的位置
    bin = numpy.floor((NFFT + 1) * hz_points / samplerate)
    # 接下来建立滤波器的表达式了，每个滤波器在第一个点处和第三个点处均为0，中间为三角形形状
    print(filters_num, NFFT / 2 + 1)
    fbank = numpy.zeros([filters_num, int(NFFT / 2 + 1)])

    for j in range(0, filters_num):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


def lifter(cepstra, L=22):
    '''升倒谱函数
    参数说明：
    cepstra:MFCC系数
    L：升系数，默认为22
    '''
    if L > 0:
        nframes, ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L / 2) * numpy.sin(numpy.pi * n / L)
        return lift * cepstra
    else:
        return cepstra

if __name__=='__main__':
    (rate, sig) = wavfile.read("../dataset/merge/AD/AD_F_030807.wav")
    mfcc_feat = calcMFCC_delta_delta(sig, rate)
    print(mfcc_feat)
    plt.plot(mfcc_feat)
    plt.show()
    print(mfcc_feat.shape)