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
from numpy import zeros
from numpy import r_
from  scipy.io.wavfile import read as wavread

# Used for symbol creation. Returns a decimal number from a 2 bit input
def GetQpskSymbol(bit1:bool, bit2:bool):
    if(~bit1 & ~bit2):
        return 0
    elif(~bit1 & bit2):
        return 1
    elif(bit1 & ~bit2):
        return 2
    elif(bit1 & bit2):
        return 3
    else:
        return -1

# Maps a given symbol to a complex signal. Optionally, noise and phase offset can be added.
def QpskSymbolMapper(symbols:int,amplitude_I, amplitude_Q,noise1=0, noise2=0,  phaseOffset1 = 0, phaseOffset2 = 0):
    if(symbols == 0):
        return sqrt(amplitude_I**2 + amplitude_Q**2)*(cos(np.deg2rad(45) + phaseOffset1)+ 1j *sin(np.deg2rad(45) + phaseOffset2)) + (noise1 + 1j*noise2)
    elif(symbols == 1):
        return sqrt(amplitude_I**2 + amplitude_Q**2)*(cos(np.deg2rad(135) + phaseOffset1) + 1j * sin(np.deg2rad(135)  + phaseOffset2)) + (noise1 + 1j*noise2)
    elif(symbols == 2):
        return sqrt(amplitude_I**2 + amplitude_Q**2)*(cos(np.deg2rad(225)  + phaseOffset1) + 1j * sin(np.deg2rad(225) + phaseOffset2)) + (noise1 + 1j*noise2)
    elif(symbols == 3):
        return sqrt(amplitude_I**2 + amplitude_Q**2)*(cos(np.deg2rad(315)  + phaseOffset1) + 1j *sin(np.deg2rad(315)  + phaseOffset2)) + (noise1 + 1j*noise2)
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

#----------------------------#
#---------- QPSK ------------#
#----------------------------#
#Input of the modulator
inputBits = np.random.randn(Nbits,1) > 0 
inputSignal = (np.tile(inputBits*2-1,(1,Ns))).ravel()

#Only calculate when dividable by 2
if(inputBits.size%2 == 0):

    #carrier signals used for modulation. carrier2 has 90° phaseshift compared to carrier1 -> IQ modulation
    carrier1 = cos(2*pi*f0*t)
    carrier2 = cos(2*pi*f0*t+pi/2)

    #Serial-to-Parallel Converter (1/2 of data rate)
    I_bits = inputBits[::2]
    Q_bits = inputBits[1::2]
       
    #Digital-to-Analog Conversion
    I_signal = (np.tile(I_bits*2-1,(1,2*Ns))).ravel()
    Q_signal = (np.tile(Q_bits*2-1,(1,2*Ns)) ).ravel()

    #Multiplicator / mixxer
    I_signal_modulated = I_signal * carrier1
    Q_signal_modulated = Q_signal * carrier2

    #Summation befor transmission
    QPSK_signal = I_signal_modulated + Q_signal_modulated

    #---------- Preperation QPSK Constellation Diagram ------------#
    dataSymbols = np.array([[GetQpskSymbol(I_bits[x],Q_bits[x])] for x in range(0,I_bits.size)])
    
    # amplitudes of I_signal and Q_signal (absolut values)
    amplitude_I_signal = 1
    amplitude_Q_signal = 1

    #Generate noise. Two sources for uncorelated noise for each amplitude.
    noiseStandardDeviation = 0.065
    noise1 = np.random.normal(0,noiseStandardDeviation,dataSymbols.size) 
    noise2 = np.random.normal(0,noiseStandardDeviation,dataSymbols.size)

    #Transmitted and received symbols. Rx symbols are generated under the presence of noise
    Tx_symbols = np.array([[QpskSymbolMapper(dataSymbols[x], 
                                             amplitude_I_signal, 
                                             amplitude_Q_signal,  
                                             phaseOffset1 = np.deg2rad(0), 
                                             phaseOffset2 = np.deg2rad(0)
                                             )] for x in range(0,dataSymbols.size)])
    Rx_symbols = np.array([[QpskSymbolMapper(dataSymbols[x], 
                                             amplitude_I_signal ,
                                             amplitude_Q_signal, 
                                             noise1=noise1[x], 
                                             noise2=noise2[x],
                                             )] for x in range(0,dataSymbols.size)])
    
    #---------- Plot of QPSK Constellation Diagram ------------#
    fig, axis = plt.subplots(2,2,sharey='row')
    fig.suptitle('Constellation Diagram QPSK', fontsize=12)

    axis[0,0].plot(Tx_symbols.real, Tx_symbols.imag,'.', color='C1')
    axis[0,0].set_title('Tx (Source Code/ Block Diagram: "Tx_symbols")')
    axis[0,0].set_xlabel('Inphase [V]')
    axis[0,0].set_ylabel('Quadrature [V]')
    axis[0,0].set_xlim(-2,2)
    axis[0,0].set_ylim(-2,2)

    axis[0,1].plot(Tx_symbols.real, Tx_symbols.imag,'-',Tx_symbols.real, Tx_symbols.imag,'.')
    axis[0,1].set_title('Tx with Trajectory (Source Code/ Block Diagram: "Tx_symbols")')
    axis[0,1].set_xlabel('Inphase [V]')
    axis[0,1].set_ylabel('Quadrature [V]')
    axis[0,1].set_xlim(-2,2)
    axis[0,1].set_ylim(-2,2)

    axis[1,0].plot(Rx_symbols.real, Rx_symbols.imag,'.', color='C1')
    axis[1,0].set_title('Rx (Source Code/ Block Diagram: "Rx_symbols")')
    axis[1,0].set_xlabel('Inphase [V]')
    axis[1,0].set_ylabel('Quadrature [V]')
    axis[1,0].set_xlim(-2,2)
    axis[1,0].set_ylim(-2,2)

    axis[1,1].plot(Rx_symbols.real, Rx_symbols.imag,'-',Rx_symbols.real, Rx_symbols.imag,'.')
    axis[1,1].set_title('Rx with Trajectory (Source Code/ Block Diagram: "Rx_symbols")')
    axis[1,1].set_xlabel('Inphase [V]')
    axis[1,1].set_ylabel('Quadrature [V]')
    axis[1,1].set_xlim(-2,2)
    axis[1,1].set_ylim(-2,2)

    #---------- Plot of QPSK Modulating Signals ------------#
    fig, axis = plt.subplots(6,1,sharex='col')
    fig.suptitle('QPSK Modulation', fontsize=12)

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

    axis[5].plot(t,QPSK_signal, color='C4')
    axis[5].set_title('QPSK Signal Modulated (Source Code/ Block Diagram: "QPSK_signal")')
    axis[5].set_xlabel('Time [s]')
    axis[5].set_xlim(0,timeDomainVisibleLimit)
    axis[5].set_ylabel('Amplitude [V]')
    axis[5].grid(linestyle='dotted')

    plt.subplots_adjust(hspace=0.5)

    #---------- Plot of Modulated Signal and Spectrum ------------#
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, figure=fig)
    fig.suptitle('QPSK Modulation', fontsize=12)
    ax = fig.add_subplot(gs[0, :])
     
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('Magnitude Spectrum (Source Code/ Block Diagram: "QPSK_signal")');
    ax1.magnitude_spectrum(QPSK_signal, Fs=fs, color='C1')
    ax1.set_xlim(0,6000)
    ax1.grid(linestyle='dotted')

    ax2 = fig.add_subplot(gs[1])
    ax2.set_title('Log. Magnitude Spectrum (Source Code/ Block Diagram: "QPSK_signal")')
    ax2.magnitude_spectrum(QPSK_signal, Fs=fs, scale='dB', color='C1')
    ax2.set_xlim(0,6000)
    ax2.grid(linestyle='dotted')

    ax3 = fig.add_subplot(gs[2])
    ax3.set_title('Power Spectrum Density (PSD) (Source Code/ Block Diagram: "QPSK_signal")')
    ax3.psd(QPSK_signal,NFFT=len(t),Fs=fs)
    ax3.set_xlim(0,6000)
    ax3.grid(linestyle='dotted')

    plt.subplots_adjust(hspace=0.5)
    plt.show()  
else:
    print("Error! Number of bits has to be a multiple of 2. Number of Bits entered: "+ str(Nbits)+".")

