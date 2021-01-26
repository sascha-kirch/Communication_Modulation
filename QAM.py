# Import functions and libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy as scy
import threading,time
import multiprocessing
import sys
import bitarray
import cmath

from scipy.fftpack import fft
from numpy import pi
from numpy import sqrt
from numpy import sin
from numpy import cos
from numpy import exp
from numpy import zeros
from numpy import r_
from  scipy.io.wavfile import read as wavread

# DAC - Maps an two bit input to a certain amplitude provided
def TwoBitToAmplitudeMapper(bit1:bool, bit2:bool, amplitude1, amplitude2):
    if(~bit1 & ~bit2):
        return np.array(-amplitude1)
    elif(~bit1 & bit2):
        return np.array(-amplitude2)
    elif(bit1 & ~bit2):
        return np.array(amplitude1)
    elif(bit1 & bit2):
        return np.array(amplitude2)
    else:
        return 0

# Used for symbol creation. Returns a decimal number from a 4 bit input
def GetQam16Symbol(bit1:bool, bit2:bool, bit3:bool, bit4:bool):
    if(~bit1 & ~bit2 & ~bit3 & ~bit4):
        return 0
    elif(~bit1 & ~bit2 & ~bit3 & bit4):
        return 1
    elif(~bit1 & ~bit2 & bit3 & ~bit4):
        return 2
    elif(~bit1 & ~bit2 & bit3 & bit4):
        return 3
    elif(~bit1 & bit2 & ~bit3 & ~bit4):
        return 4
    elif(~bit1 & bit2 & ~bit3 & bit4):
        return 5
    elif(~bit1 & bit2 & bit3 & ~bit4):
        return 6
    elif(~bit1 & bit2 & bit3 & bit4):
        return 7
    elif(bit1 & ~bit2 & ~bit3 & ~bit4):
        return 8
    elif(bit1 & ~bit2 & ~bit3 & bit4):
        return 9
    elif(bit1 & ~bit2 & bit3 & ~bit4):
        return 10
    elif(bit1 & ~bit2 & bit3 & bit4):
        return 11
    elif(bit1 & bit2 & ~bit3 & ~bit4):
        return 12
    elif(bit1 & bit2 & ~bit3 & bit4):
        return 13
    elif(bit1 & bit2 & bit3 & ~bit4):
        return 14
    elif(bit1 & bit2 & bit3 & bit4):
        return 15
    else:
        return -1

# Maps a given symbol to a complex signal. Optionally, noise and phase offset can be added.
def Qam16SymbolMapper(symbols:int, amplitude_I1, amplitude_Q1, amplitude_I2, amplitude_Q2, noise1=0, noise2=0, noise3=0, noise4=0, phaseOffset1 = 0, phaseOffset2 = 0):
    if(symbols == 0):#0000
        return sqrt(amplitude_I1**2 + amplitude_Q1**2)*(cos(np.deg2rad(225) + phaseOffset1)+ 1j *sin(np.deg2rad(225) + phaseOffset2)) + (noise1 + 1j*noise2)
    elif(symbols == 1):#0001
        return sqrt(amplitude_I2**2 + amplitude_Q1**2 )*(cos(np.deg2rad(198) + phaseOffset1)+ 1j *sin(np.deg2rad(198) + phaseOffset2)) + (noise3 + 1j*noise2)
    elif(symbols == 2):#0010
        return sqrt(amplitude_I1**2 + amplitude_Q2**2)*(cos(np.deg2rad(251) + phaseOffset1)+ 1j *sin(np.deg2rad(251) + phaseOffset2)) + (noise1 + 1j*noise4)
    elif(symbols == 3):#0011
        return sqrt(amplitude_I2**2 + amplitude_Q2**2)*(cos(np.deg2rad(225) + phaseOffset1)+ 1j *sin(np.deg2rad(225) + phaseOffset2)) + (noise3 + 1j*noise4)
    elif(symbols == 4):#0100
        return sqrt(amplitude_I1**2 + amplitude_Q1**2)*(cos(np.deg2rad(315) + phaseOffset1)+ 1j *sin(np.deg2rad(315) + phaseOffset2)) + (noise1 + 1j*noise2)
    elif(symbols == 5):#0101
        return sqrt(amplitude_I1**2 + amplitude_Q2**2)*(cos(np.deg2rad(288) + phaseOffset1)+ 1j *sin(np.deg2rad(288) + phaseOffset2)) + (noise1 + 1j*noise4)
    elif(symbols == 6):#0110
        return sqrt(amplitude_I2**2 + amplitude_Q1**2)*(cos(np.deg2rad(342) + phaseOffset1)+ 1j *sin(np.deg2rad(342) + phaseOffset2)) + (noise2 + 1j*noise3)
    elif(symbols == 7):#0111
        return sqrt(amplitude_I2**2 + amplitude_Q2**2)*(cos(np.deg2rad(315) + phaseOffset1)+ 1j *sin(np.deg2rad(315) + phaseOffset2)) + (noise3 + 1j*noise4)
    elif(symbols == 8):#1000
        return sqrt(amplitude_I1**2 + amplitude_Q1**2)*(cos(np.deg2rad(135) + phaseOffset1)+ 1j *sin(np.deg2rad(135) + phaseOffset2)) + (noise1 + 1j*noise2)
    elif(symbols == 9):#1001
        return sqrt(amplitude_I1**2 + amplitude_Q2**2)*(cos(np.deg2rad(108) + phaseOffset1)+ 1j *sin(np.deg2rad(108) + phaseOffset2)) + (noise1 + 1j*noise4)
    elif(symbols == 10):#1010
        return sqrt(amplitude_I2**2 + amplitude_Q1**2)*(cos(np.deg2rad(162) + phaseOffset1)+ 1j *sin(np.deg2rad(162) + phaseOffset2)) + (noise3 + 1j*noise2)
    elif(symbols == 11):#1011
        return sqrt(amplitude_I2**2 + amplitude_Q2**2)*(cos(np.deg2rad(135) + phaseOffset1)+ 1j *sin(np.deg2rad(135) + phaseOffset2)) + (noise3 + 1j*noise4)
    elif(symbols == 12):#1100
        return  sqrt(amplitude_I1**2 + amplitude_Q1**2)*(cos(np.deg2rad(45) + phaseOffset1)+ 1j *sin(np.deg2rad(45) + phaseOffset2)) + (noise1 + 1j*noise2)
    elif(symbols == 13):#1101
        return sqrt(amplitude_I2**2 + amplitude_Q1**2)*(cos(np.deg2rad(18) + phaseOffset1)+ 1j *sin(np.deg2rad(15) + phaseOffset2)) + (noise2 + 1j*noise3)
    elif(symbols == 14):#1110
        return sqrt(amplitude_I1**2 + amplitude_Q2**2)*(cos(np.deg2rad(71) + phaseOffset1)+ 1j *sin(np.deg2rad(75) + phaseOffset2)) + (noise1 + 1j*noise4)
    elif(symbols == 15):#1111
        return sqrt(amplitude_I2**2 + amplitude_Q2**2)*(cos(np.deg2rad(45) + phaseOffset1)+ 1j *sin(np.deg2rad(45) + phaseOffset2)) + (noise3 + 1j*noise4)
    else:
        return complex(0)

#-------------------------------------#
#---------- Configuration ------------#
#-------------------------------------#
fs = 44100                  # sampling rate
baud = 900                  # symbol rate
Nbits = 4000                # number of bits
f0 = 1800                   # carrier Frequency
Ns = int(fs/baud)           # number of Samples per Symbol
N = Nbits * Ns              # Total Number of Samples
t = r_[0.0:N]/fs            # time points
f = r_[0:N/2.0]/N*fs        # Frequency Points

# Limit for representation of time domain signals for better visibility. 
symbolsToShow = 20
timeDomainVisibleLimit = np.minimum(Nbits/baud,symbolsToShow/baud)     

#----------------------------------------#
#---------- QAM16 Modulation ------------#
#----------------------------------------#

#Input of the modulator
inputBits = np.random.randn(Nbits,1) > 0 
inputSignal = (np.tile(inputBits,(1,Ns))).ravel()

#Only calculate when dividable by 4
if(inputBits.size%4 == 0): 
    
    # amplitudes of I_signal and Q_signal (absolut values)
    amplitude1_I_signal = 0.25
    amplitude2_I_signal = 0.75
    amplitude1_Q_signal = 0.25
    amplitude2_Q_signal = 0.75

    #carrier signals used for modulation. carrier2 has 90° phaseshift compared to carrier1 -> IQ modulation
    carrier1 = cos(2*pi*f0*t)
    carrier2 = cos(2*pi*f0*t+pi/2)

    #Serial-to-Parallel Converter (1/4 of data rate)
    I1_bits = inputBits[::4] 
    I2_bits = inputBits[1::4]
    Q1_bits = inputBits[2::4]
    Q2_bits = inputBits[3::4]
    
    #Digital-to-Analog Conversion
    I_symbols = np.array([[TwoBitToAmplitudeMapper(I1_bits[x], I2_bits[x], amplitude1_I_signal, amplitude2_I_signal)] for x in range(0,I1_bits.size)]) 
    Q_symbols = np.array([[TwoBitToAmplitudeMapper(Q1_bits[x], Q2_bits[x], amplitude1_Q_signal, amplitude2_Q_signal)] for x in range(0,Q1_bits.size)])
    I_signal = (np.tile(I_symbols,(1,4*Ns))).ravel()
    Q_signal = (np.tile(Q_symbols,(1,4*Ns))).ravel()
   
    #Multiplicator / mixxer
    I_signal_modulated = I_signal * carrier1
    Q_signal_modulated = Q_signal * carrier2

    #Summation befor transmission
    QAM16_signal = I_signal_modulated + Q_signal_modulated

    #---------- Preperation Constellation Diagram ------------#
    dataSymbols = np.array([[GetQam16Symbol(I1_bits[x], I2_bits[x], Q1_bits[x], Q2_bits[x])] for x in range(0,I1_bits.size)])

    #Generate noise. Four sources for uncorelated noise for each amplitude.
    noiseStandardDeviation = 0.055
    noise1 = np.random.normal(0,noiseStandardDeviation,dataSymbols.size)
    noise2 = np.random.normal(0,noiseStandardDeviation,dataSymbols.size)
    noise3 = np.random.normal(0,noiseStandardDeviation,dataSymbols.size)
    noise4 = np.random.normal(0,noiseStandardDeviation,dataSymbols.size)

    #Transmitted and received symbols. Rx symbols are generated under the presence of noise
    Tx_symbols = np.array([[Qam16SymbolMapper(dataSymbols[x],
                                              amplitude1_I_signal, 
                                              amplitude1_Q_signal,
                                              amplitude2_I_signal, 
                                              amplitude2_Q_signal, 
                                              phaseOffset1 = np.deg2rad(0), 
                                              phaseOffset2 = np.deg2rad(0)
                                              )]for x in range(0,dataSymbols.size)])
    Rx_symbols = np.array([[Qam16SymbolMapper(dataSymbols[x],
                                              amplitude1_I_signal,
                                              amplitude1_Q_signal,
                                              amplitude2_I_signal, 
                                              amplitude2_Q_signal,
                                              noise1 = noise1[x],
                                              noise2 = noise2[x],
                                              noise3 = noise3[x],
                                              noise4 = noise4[x]
                                              )] for x in range(0,dataSymbols.size)])
    
    #-------------------------------------------#
    #---------- Data Representation ------------#
    #-------------------------------------------#

    #---------- Plot of QAM 16 Constellation Diagram ------------#
    fig, axis = plt.subplots(2,2,sharey='row')
    fig.suptitle('Constellation Diagram QAM16', fontsize=12)

    axis[0,0].plot(Tx_symbols.real, Tx_symbols.imag,'.', color='C1')
    axis[0,0].set_title('Tx (Source Code/ Block Diagram: "Tx_symbols")')
    axis[0,0].set_xlabel('Inphase [V]')
    axis[0,0].set_ylabel('Quadrature [V]')
    axis[0,0].set_xlim(-1.5,1.5)
    axis[0,0].set_ylim(-1.5,1.5)
    axis[0,0].set_xticks([-1,-0.5,0,0.5,1])
    axis[0,0].set_yticks([-1,-0.5,0,0.5,1])
    axis[0,0].grid(True)

    axis[0,1].plot(Tx_symbols.real, Tx_symbols.imag,'-',Tx_symbols.real, Tx_symbols.imag,'.')
    axis[0,1].set_title('Tx with Trajectory (Source Code/ Block Diagram: "Tx_symbols")')
    axis[0,1].set_xlabel('Inphase [V]')
    axis[0,1].set_ylabel('Quadrature [V]')
    axis[0,1].set_xlim(-1.5,1.5)
    axis[0,1].set_ylim(-1.5,1.5)
    axis[0,1].set_xticks([-1,-0.5,0,0.5,1])
    axis[0,1].set_yticks([-1,-0.5,0,0.5,1])
    axis[0,1].grid(True)

    axis[1,0].plot(Rx_symbols.real, Rx_symbols.imag,'.', color='C1')
    axis[1,0].set_title('Rx (Source Code/ Block Diagram: "Rx_symbols")')
    axis[1,0].set_xlabel('Inphase [V]')
    axis[1,0].set_ylabel('Quadrature [V]')
    axis[1,0].set_xticks([-1,-0.5,0,0.5,1])
    axis[1,0].set_yticks([-1,-0.5,0,0.5,1])
    axis[1,0].set_xlim(-1.5,1.5)
    axis[1,0].set_ylim(-1.5,1.5)
    axis[1,0].grid(True)

    axis[1,1].plot(Rx_symbols.real, Rx_symbols.imag,'-',Rx_symbols.real, Rx_symbols.imag,'.')
    axis[1,1].set_title('Rx with Trajectory (Source Code/ Block Diagram: "Rx_symbols")')
    axis[1,1].set_xlabel('Inphase [V]')
    axis[1,1].set_ylabel('Quadrature [V]')
    axis[1,1].set_xlim(-1.5,1.5)
    axis[1,1].set_ylim(-1.5,1.5)
    axis[1,1].set_xticks([-1,-0.5,0,0.5,1])
    axis[1,1].set_yticks([-1,-0.5,0,0.5,1])
    axis[1,1].grid(True)

    #---------- Plot of  Modulating Signals  ------------#
    fig, axis = plt.subplots(6,1,sharex='col')
    fig.suptitle('QAM16 Modulation', fontsize=12)

    axis[0].plot(t, inputSignal, color='C1')
    axis[0].set_title('Digital Data Signal (Source Code/ Block Diagram: "inputBits")')
    axis[0].set_xlabel('Time [s]')
    axis[0].set_ylabel('Amplitude [V]')
    axis[0].set_xlim(0,timeDomainVisibleLimit)
    axis[0].grid(linestyle='dotted')

    axis[1].plot(t, I_signal, color='C2')
    axis[1].set_title('Digital I-Signal (Source Code/ Block Diagram: "I_signal")')
    axis[1].set_xlabel('Time [s]')
    axis[1].set_xlim(0,timeDomainVisibleLimit)
    axis[1].set_ylabel('Amplitude [V]')
    axis[1].grid(linestyle='dotted')

    axis[2].plot(t, I_signal_modulated, color='C2')
    axis[2].set_title('Modulated I-Signal (Source Code/ Block Diagram: "I_signal_modulated")')
    axis[2].set_xlabel('Time [s]')
    axis[2].set_xlim(0,timeDomainVisibleLimit)
    axis[2].set_ylabel('Amplitude [V]')
    axis[2].grid(linestyle='dotted')

    axis[3].plot(t, Q_signal, color='C3')
    axis[3].set_title('Digital Q-Signal (Source Code/ Block Diagram: "Q_signal")')
    axis[3].set_xlabel('Time [s]')
    axis[3].set_xlim(0,timeDomainVisibleLimit)
    axis[3].set_ylabel('Amplitude [V]')
    axis[3].grid(linestyle='dotted')    
         
    axis[4].plot(t, Q_signal_modulated, color='C3')
    axis[4].set_title('Modulated Q-Signal (Source Code/ Block Diagram: "Q_signal_modulated")')
    axis[4].set_xlabel('Time [s]')
    axis[4].set_xlim(0,timeDomainVisibleLimit)
    axis[4].set_ylabel('Amplitude [V]')
    axis[4].grid(linestyle='dotted')

    axis[5].plot(t,QAM16_signal, color='C4')
    axis[5].set_title('QAM16 Signal Modulated (Source Code/ Block Diagram: "QAM16_signal")')
    axis[5].set_xlabel('Time [s]')
    axis[5].set_xlim(0,timeDomainVisibleLimit)
    axis[5].set_ylabel('Amplitude [V]')
    axis[5].grid(linestyle='dotted')

    plt.subplots_adjust(hspace=0.5)

     #---------- Plot of Modulated Signal and Spectrum ------------#
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, figure=fig)
    fig.suptitle('QAM16 Modulation', fontsize=12)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('Magnitude Spectrum (Source Code/ Block Diagram: "QAM16_signal")');
    ax1.magnitude_spectrum(QAM16_signal, Fs=fs, color='C1')
    ax1.set_xlim(0,6000)
    ax1.grid(linestyle='dotted')

    ax2 = fig.add_subplot(gs[1])
    ax2.set_title('Log. Magnitude Spectrum (Source Code/ Block Diagram: "QAM16_signal")')
    ax2.magnitude_spectrum(QAM16_signal, Fs=fs, scale='dB', color='C1')
    ax2.set_xlim(0,6000)
    ax2.grid(linestyle='dotted')

    ax3 = fig.add_subplot(gs[2])
    ax3.set_title('Power Spectrum Density (PSD) (Source Code/ Block Diagram: "QAM16_signal")')
    ax3.psd(QAM16_signal,NFFT=len(t),Fs=fs)
    ax3.set_xlim(0,6000)
    ax3.grid(linestyle='dotted')

    plt.subplots_adjust(hspace=0.5)
    plt.show() 
else:
    print("Error! Number of bits has to be a multiple of 4. Number of Bits entered: "+ str(Nbits)+".")
