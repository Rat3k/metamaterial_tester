import os
import numpy as np
import matplotlib.pyplot as plt

#set working directory
os.chdir('D:\Documents\LumpedPar')

#%%
#USE SI UNITS FOR ALL INPUTS!

#%% global parameters
#frequencies
freqs = np.logspace(np.log10(50), 3, 1000)
w = 2*np.pi*freqs

#default gas properties
rho0 = 1.21
c0 = 343
P0 = 1e5

#SPL reference pressure
Pref = 2e-5

#%% define objects
#amplifier with itnernal resisstance Rg and voltage output eg
class amp:
    def __init__(self, Rg, eg):
        self.Rg = Rg
        self.eg = eg

#loudpeaker with T/S parameters
class driver:
    rho = rho0
    def __init__(self, Sd, Re, Bl, Mms, Cms, Rms, Xmax_pp, Le = 0):
        self.Sd = Sd
        self.Re = Re
        self.Bl = Bl
        self.Cms = Cms
        self.Cas = Cms*Sd**2
        self.Rms = Rms
        self.a = np.sqrt(Sd/np.pi)
        self.Mmd = Mms - 2*2.67*driver.rho*self.a**3
        self.Xmax = Xmax_pp/2 #Xmax half pp to compare with excursion amplitude
        self.Le = Le

#metamaterial parameters by doi 10.1121/10.0017214, eq.10
class MAM:
    rho = rho0
    omegas = w
    def __init__(self, S, f1, f2, fp, Kp, Kf, G, fms = 0, Mms = 0):
        f = MAM.omegas/2/np.pi
        fpt = np.sqrt(fp**2 + Kf*G/4/np.pi**2/Kp)
        self.S = S
        self.a = np.sqrt(self.S/np.pi)
        self.meff = -(f1**2 - f**2)*(f2**2 - f**2)/f**2/Kp/(fpt**2 - f**2)
        self.Cas = S**2/Mms/(2*np.pi*fms)**2 if ((Mms > 0) & (fms > 0)) else 0
        self.Mmd = (Mms > 0)*(Mms - 2*2.67*MAM.rho*self.a**3)

#mechanical mass element with surface area S, thickness d and density rho
class MAM_mass:
    omegas = w
    def __init__(self, S, d, rho, Cas = 0, Mms = 0):
        self.S = S
        self.a = np.sqrt(self.S/np.pi)
        self.meff = rho*d
        self.Cas = Cas
        self.Mmd = (Mms > 0)*(Mms - 2*2.67*MAM.rho*self.a**3)

#enclosure with base l*w, bottom height h1, top height h2, absorption and leakage
#absorbent lining formula by beranek pp. 347
class enclosure:
    omegas = w
    rho = rho0
    c = c0
    Patm = P0
    gamma = c0**2*rho0/1e5
    def __init__(self, l, w, h1, h2, d_lining = 0, Rf = 0, Ql1 = 20, Ql2 = 15):
        Vmbot = d_lining*(l*w + 2*l*(h1 - d_lining) + 2*(w - d_lining)*(h1 - d_lining))
        Vmtop = d_lining*(l*w + 2*l*(h2 - d_lining) + 2*(w - d_lining)*(h2 - d_lining))
        Vabot = l*w*h1 - Vmbot
        Vatop = l*w*h2 - Vmtop
        self.Vtot = l*w*(h1 + h2)
        self.C1 = Vabot/enclosure.rho/enclosure.c**2 + Vmbot/enclosure.Patm
        self.C2 = Vatop/enclosure.rho/enclosure.c**2 + Vmtop/enclosure.Patm
        Ram1 = d_lining**2*Rf/3/Vmbot if Vmbot > 0 else 0
        Ram2 = d_lining**2*Rf/3/Vmtop if Vmtop > 0 else 0
        self.Rab1 = Ram1/((1 + Vabot/Vmbot/enclosure.gamma)**2 + enclosure.omegas**2*Ram1**2*(self.C1 - Vmbot/enclosure.Patm)**2) if Vmbot > 0 else 0
        self.Rab2 = Ram2/((1 + Vatop/Vmtop/enclosure.gamma)**2 + enclosure.omegas**2*Ram2**2*(self.C2 - Vmtop/enclosure.Patm)**2) if Vmtop > 0 else 0
        self.Ql1 = Ql1
        self.Ql2 = Ql2

#for the Padv function
class output:
    omegas = w
    def __init__(self, Pmam, Pspk, PmamB, PspkB, Pt, Uspm, Umam, Uspk):
        self.Pmam = Pmam #pressure with sample
        self.Pspk = Pspk #reference closed-box pressure
        self.PmamB = PmamB #pressure with MAM, baffle step added
        self.PspkB = PspkB #reference closed-box pressure, baffle step added
        self.Pt = Pt #pressure in top compliance
        self.dspm = Uspm/1j/output.omegas #speaker excursion with sample
        self.dmam = Umam/1j/output.omegas #sample excursion
        self.dspk = Uspk/1j/output.omegas #speaker excursion in reference box

#%% radiation impedance, acoustic
#beranek, pp. 199-201, assumptions
def radimpA(rho, c, radius, omegas):
    Ra1 = 0.1404*rho*c/radius**2
    Ra2 = rho*c/np.pi/radius**2
    Ca1 = 1.89*np.pi*radius**3/rho/c**2
    Ma1 = 8*rho/3/np.pi**2/radius
    
    Zra = 1/(1/1j/omegas/Ma1 + 1/(Ra2 + 1/(1/Ra1 + 1j*omegas*Ca1)))
    return Zra

#%% advanced circuits, at distance dist
def Padv(driver, amp, MAM, enclosure, dist = 1):
    rho = enclosure.rho
    c = enclosure.c
    omegas = MAM.omegas
    
    #electrical impedance and acoustic conversion
    Ze = amp.Rg + driver.Re + 1j*omegas*driver.Le
    Zae = driver.Bl**2/Ze/driver.Sd**2
    
    #mechanical impedance acoustic conversion
    Zam = (1j*omegas*driver.Mmd + driver.Rms + 1/1j/omegas/driver.Cms)/driver.Sd**2
    
    #sample impedance and radiation impedance
    Zmam = 1j*omegas*MAM.meff/MAM.S
    Zar = radimpA(rho, c, MAM.a, omegas)
    
    #leakage
    if (MAM.Cas > 0) & (MAM.Mmd > 0):
        Ma1 = 8*rho/3/np.pi**2/driver.a
        Ma1m = 8*rho/3/np.pi**2/MAM.a
        Mab1 = 0.5*rho/np.pi/driver.a
        Mab1m = 0.5*rho/np.pi/MAM.a
        wc1 = np.sqrt((driver.Cas + enclosure.C1)/driver.Cas/enclosure.C1/(driver.Mmd/driver.Sd**2 + 0.75*Ma1 + Mab1))
        wc2 = np.sqrt((MAM.Cas + enclosure.C2)/MAM.Cas/enclosure.C2/(MAM.Mmd/MAM.S**2 + 0.75*Ma1m + Mab1m))
        Ral1 = enclosure.Ql1/wc1/enclosure.C1
        Ral2 = enclosure.Ql2/wc2/enclosure.C2
    else:
        Ral1 = 1e15
        Ral2 = 1e15
    
    #circuit calculations, refer to schematics for notation
    Z345 = 1/(1/(Zmam + Zar) + 1/(1/1j/omegas/enclosure.C2 + enclosure.Rab2) + 1/Ral2)
    Z67 = 1/(1/Ral1 + 1/(1/1j/omegas/enclosure.C1 + enclosure.Rab1))
    Za = Z67 + Z345
    
    P = amp.eg*driver.Bl/Ze/driver.Sd
    Q1 = P/(Zae + Zam + Za)
    
    P01 = Z67*Q1
    P21 = Za*Q1
    P20 = P21 - P01
    
    Z35 = 1/(1/Ral2 + 1/(1/1j/omegas/enclosure.C2 + enclosure.Rab2))
    Q2 = P20/Z35
    Q4 = Q1 - Q2
    
    #results for reference box
    Zarspk = radimpA(rho, c, driver.a, omegas)
    Qspk = P/(Zae + Zam + Zarspk + Z67)
    
    #infinite baffle approximation
    k = omegas/c
    Pmam = -1j*rho*omegas*Q4*np.exp(-1j*k*dist)/2/np.pi/dist #with sample
    Pspk = -1j*rho*omegas*Qspk*np.exp(-1j*k*dist)/2/np.pi/dist #reference box
    
    #baffle-step approximation
    R = (3*enclosure.Vtot/4/np.pi)**(1/3) #avg. dimension of the enclosure
    PmamB = Pmam/2*(1 + 1j*k*R)/(1 + 1j*k*R/2) #with sample
    PspkB = Pspk/2*(1 + 1j*k*R)/(1 + 1j*k*R/2) #reference box
    return output(Pmam, Pspk, PmamB, PspkB, P20, Q1/driver.Sd, Q4/MAM.S, Qspk/driver.Sd)

#%% TS parameter conversion
#by beranek pp. 289
def getCms(Mms, fs):
    Cms = 1/(2*np.pi*fs)**2/Mms
    return Cms

def getRms(Qms, Mms, Cms):
    Rms = np.sqrt(Mms/Cms)/Qms
    return Rms

#%% example input data
spk1 = driver(37.2e-4, 7.1, 4.8, 5.6e-3, 0.42e-3, 0.55, 10e-3, 0.05e-3) #TEBM65C20F-8
spk11 = driver(19.6e-4, 3.94, 4.49, 2.26e-3, 0.42e-3, 0.39, 0, 0.03e-3) #TEBM46C20N-4B
spk2 = driver(5e-3, 3.2, 4.6, 5.7e-3, getCms(5.7e-3, 90), getRms(2.29, 5.7e-3, getCms(5.7e-3, 90)), 8e-3, 0.2e-3) #visaton FR10 - 4 Ohm
spk3 = driver(124e-4, 3.2, 5, 13.8e-3, 1.5e-3, 1.1, 9e-3, 0.56e-3) #SB16PFCR25-4

amp4 = amp(0, 2) #1W@4Ohm, 0 Ohm internal, no phase shift
amp41 = amp(0, np.sqrt(40)) #10W@4Ohm
amp8 = amp(0, 2.83) #1W@8Ohm, 0 Ohm internal, no phase shift
amp81 = amp(0, np.sqrt(80)) #10W@8Ohm
amp82 = amp(0, np.sqrt(160)) #20W@8Ohm
amp83 = amp(0, np.sqrt(240)) #30W@8Ohm
amp30Ws1 = amp(0, np.sqrt(spk1.Re*30)) #30W@ into Spk1 Re

mam1 = MAM((84e-3/2)**2*np.pi, 125, 1679, 677, 2.63, 1.16e10, 0) #current active mam, by paper, no Ral
mam1a = MAM((84e-3/2)**2*np.pi, 125, 1679, 677, 2.63, 1.16e10, 0, 180, 3.2e-3) #current active mam, by paper
mam1b = MAM((84e-3/2)**2*np.pi, 125, 1679, 677, 2.63, 1.16e10, -3e-3, 180, 3.2e-3) #current mam with active control gain

mass1 = MAM_mass((84e-3/2)**2*np.pi, 1e-3, 2000) #density by preliminary investigation paper, with exciter and accelerometer
hole1 = MAM_mass((84e-3/2)**2*np.pi, 0.012, rho0) #empty plate

enc1 = enclosure(0.1, 0.1, 0.1, 0.1)
enc2 = enclosure(0.2, 0.2, 0.2, 0.2)
enc2a = enclosure(0.2, 0.2, 0.2, 0.2, 0.013, 500/0.025)
enc3 = enclosure(0.3, 0.3, 0.3, 0.3)
encf = enclosure(0.2, 0.2, 0.11, 0.1, 0.013, 500/0.025) #final construction

#%% plots for bottom chamber size
def botplot(name, values, driver, amp, MAM, hole, refedge = 0.2, dist = 1, refh2 = 0.2, d = 0, Rf = 0, bafflestep = True):
    enc = np.empty(len(values), 'object')
    for i in range(len(values)):
        enc[i] = enclosure(refedge, refedge, values[i], refh2, d, Rf)
    
    outs_mam = np.empty(len(values), 'object')
    outs_empty = np.empty(len(values), 'object')
    for i in range(len(values)):
        outs_mam[i] = Padv(driver, amp, MAM, enc[i], dist)
        outs_empty[i] = Padv(driver, amp, hole, enc[i], dist)
    
    plt.figure(figsize = [7,7])
    plt.title('On-axis SPL at ' + str(dist) + ' m, $h_2=$' + str(refh2) + ' m, ' + name)
    for i in range(len(values)):
        Pmam = outs_mam[i].PmamB if bafflestep == True else outs_mam[i].Pmam
        Pemp = outs_empty[i].PmamB if bafflestep == True else outs_empty[i].Pmam
        plt.semilogx(freqs, 20*np.log10(np.abs(Pmam)/np.sqrt(2)/2e-5), c = 'C' + str(i), label = 'With MAM, $h_1=$' + str(values[i]))
        plt.semilogx(freqs, 20*np.log10(np.abs(Pemp)/np.sqrt(2)/2e-5), c = 'C' + str(i), ls = '--', label = 'W/o MAM, $h_1=$' + str(values[i]))
        plt.semilogx(freqs, 20*np.log10(np.abs(outs_mam[i].Pt)/np.sqrt(2)/2e-5), c = 'C' + str(i), ls = ':', label = 'Top chamber, w MAM, $h_1=$' + str(values[i]))
    plt.ylabel(r'dB re $2\times10^{-5}$ Pa')
    plt.xlabel('Frequency, Hz')
    plt.ylim((0, 150))
    plt.grid()
    plt.legend(loc = 3)
    plt.savefig('botSPL_' + name + '.png', dpi=600, bbox_inches = "tight")
    
    plt.figure(figsize = [7,7])
    plt.title('Unwrapped phase, $h_2=$' + str(refh2) + ' m, ' + name)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(360))
    for i in range(len(values)):
        Pmam = outs_mam[i].PmamB if bafflestep == True else outs_mam[i].Pmam
        Pemp = outs_empty[i].PmamB if bafflestep == True else outs_empty[i].Pmam
        plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(Pmam))), c = 'C' + str(i), label = 'With MAM, $h_1=$' + str(values[i]))
        plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(Pemp))), c = 'C' + str(i), ls = '--', label = 'W/o MAM, $h_1=$' + str(values[i]))
        plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(outs_mam[i].Pt))), c = 'C' + str(i), ls = ':', label = 'Top chamber, w MAM, $h_1=$' + str(values[i]))
    plt.ylabel('Degrees')
    plt.xlabel('Frequency, Hz')
    plt.grid()
    plt.legend(loc = 3)
    plt.savefig('botPhase_' + name + '.png', dpi=600, bbox_inches = "tight")
    
    plt.figure(figsize = [7,5])
    plt.title('Simulated (piston) excursion, $h_2=$' + str(refh2) + ' m, ' + name)
    for i in range(len(values)):
        plt.semilogx(freqs, np.abs(outs_mam[i].dspm)*1000, c = 'C' + str(i), label = 'Spk. with MAM, $h_1=$' + str(values[i]))
        plt.semilogx(freqs, np.abs(outs_empty[i].dspm)*1000, c = 'C' + str(i), ls = '--', label = 'Spk. w/o MAM, $h_1=$' + str(values[i]))
        plt.semilogx(freqs, np.abs(outs_mam[i].dmam)*1000, c = 'C' + str(i), ls = ':', label = 'MAM, $h_1=$' + str(values[i]))
    plt.semilogx(freqs, np.ones(len(freqs))*driver.Xmax*1000, c = 'k', ls = '--', label = r'Spk. $X_{max}$')
    plt.ylabel(r'mm')
    plt.xlabel('Frequency, Hz')
    plt.grid()
    plt.legend()
    plt.savefig('botExc_' + name + '.png', dpi=600, bbox_inches = "tight")
    return

#example
botplot('Speaker 1', np.array([0.05, 0.2, 0.6]), spk1, amp81, mam1a, hole1, d = 0.013, Rf = 500/0.025)

#%% plots for top chamber size
def topplot(name, values, driver, amp, MAM, hole, refedge = 0.2, dist = 1, refh1 = 0.2, d = 0, Rf = 0, bafflestep = True):
    enc = np.empty(len(values), 'object')
    for i in range(len(values)):
        enc[i] = enclosure(refedge, refedge, refh1, values[i], d, Rf)
    
    outs_mam = np.empty(len(values), 'object')
    outs_empty = np.empty(len(values), 'object')
    for i in range(len(values)):
        outs_mam[i] = Padv(driver, amp, MAM, enc[i], dist)
        outs_empty[i] = Padv(driver, amp, hole, enc[i], dist)
    
    plt.figure(figsize = [7,7])
    plt.title('On-axis SPL at ' + str(dist) + ' m, $h_1=$' + str(refh1) + ' m, ' + name)
    for i in range(len(values)):
        Pmam = outs_mam[i].PmamB if bafflestep == True else outs_mam[i].Pmam
        Pemp = outs_empty[i].PmamB if bafflestep == True else outs_empty[i].Pmam
        plt.semilogx(freqs, 20*np.log10(np.abs(Pmam)/np.sqrt(2)/2e-5), c = 'C' + str(i), label = 'With MAM, $h_2=$' + str(values[i]))
        plt.semilogx(freqs, 20*np.log10(np.abs(Pemp)/np.sqrt(2)/2e-5), c = 'C' + str(i), ls = '--', label = 'W/o MAM, $h_2=$' + str(values[i]))
        plt.semilogx(freqs, 20*np.log10(np.abs(outs_mam[i].Pt)/np.sqrt(2)/2e-5), c = 'C' + str(i), ls = ':', label = 'Top chamber, w MAM, $h_2=$' + str(values[i]))
    plt.ylabel(r'dB re $2\times10^{-5}$ Pa')
    plt.xlabel('Frequency, Hz')
    plt.ylim((0, 150))
    plt.grid()
    plt.legend(loc = 3)
    plt.savefig('topSPL_' + name + '.png', dpi=600, bbox_inches = "tight")
    
    plt.figure(figsize = [7,7])
    plt.title('Unwrapped phase, $h_1=$' + str(refh1) + ' m, ' + name)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(360))
    for i in range(len(values)):
        Pmam = outs_mam[i].PmamB if bafflestep == True else outs_mam[i].Pmam
        Pemp = outs_empty[i].PmamB if bafflestep == True else outs_empty[i].Pmam
        plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(Pmam))), c = 'C' + str(i), label = 'With MAM, $h_2=$' + str(values[i]))
        plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(Pemp))), c = 'C' + str(i), ls = '--', label = 'W/o MAM, $h_2=$' + str(values[i]))
        plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(outs_mam[i].Pt))), c = 'C' + str(i), ls = ':', label = 'Top chamber, w MAM, $h_2=$' + str(values[i]))
    plt.ylabel('Degrees')
    plt.xlabel('Frequency, Hz')
    plt.grid()
    plt.legend(loc = 3)
    plt.savefig('topPhase_' + name + '.png', dpi=600, bbox_inches = "tight")
    
    plt.figure(figsize = [7,5])
    plt.title('Simulated (piston) excursion, $h_1=$' + str(refh1) + ' m, ' + name)
    for i in range(len(values)):
        plt.semilogx(freqs, np.abs(outs_mam[i].dspm)*1000, c = 'C' + str(i), label = 'Spk. with MAM, $h_2=$' + str(values[i]))
        plt.semilogx(freqs, np.abs(outs_empty[i].dspm)*1000, c = 'C' + str(i), ls = ':', label = 'Spk. w/o MAM, $h_2=$' + str(values[i]))
        plt.semilogx(freqs, np.abs(outs_mam[i].dmam)*1000, c = 'C' + str(i), ls = ':', label = 'MAM, $h_2=$' + str(values[i]))
    plt.semilogx(freqs, np.ones(len(freqs))*driver.Xmax*1000, c = 'k', ls = '--', label = r'Spk. $X_{max}$')
    plt.ylabel(r'mm')
    plt.xlabel('Frequency, Hz')
    plt.grid()
    plt.legend()
    plt.savefig('topExc_' + name + '.png', dpi=600, bbox_inches = "tight")
    return

#example
topplot('Speaker 1', np.array([0.05, 0.2, 0.6]), spk1, amp81, mam1a, hole1, d = 0.013, Rf = 500/0.025)

#%% A-weighting curve by IEC standard
A_1000 = -2
f1 = 20.6
f2 = 107.7
f3 = 737.9
f4 = 12194

A = 20*np.log10(f4**2*freqs**4/(freqs**2 + f1**2)/np.sqrt(freqs**2 + f2**2)/np.sqrt(freqs**2 + f3**2)/(freqs**2 + f4**2)) - A_1000

#%% final plots for mam, mass and empty samples
#option to set bafflestep and distance

def finalplotsh(name, driver, amp, MAM, MAM_mass, MAM_hole, enclosure, dist = 1, bafflestep = True):
    out = Padv(driver, amp, MAM, enclosure, dist)
    outm = Padv(driver, amp, MAM_mass, enclosure, dist)
    outh = Padv(driver, amp, MAM_hole, enclosure, dist)
    
    Pmam = out.PmamB if bafflestep == True else out.Pmam
    Pspk = out.PspkB if bafflestep == True else out.Pspk
    Pmass = outm.PmamB if bafflestep == True else outm.Pmam
    Phole = outh.PmamB if bafflestep == True else outh.Pmam
    
    plt.figure(figsize = [7,7])
    plt.title('On-axis SPL at ' + str(dist) + ' m')
    plt.semilogx(freqs, 20*np.log10(np.abs(Pmam)/np.sqrt(2)/2e-5), c = 'b', label = 'With MAM')
    plt.semilogx(freqs, 20*np.log10(np.abs(Pmass)/np.sqrt(2)/2e-5), c = 'm', label = 'With equiv. mass')
    plt.semilogx(freqs, 20*np.log10(np.abs(Phole)/np.sqrt(2)/2e-5), c = 'g', label = 'Empty plate')
    plt.semilogx(freqs, 20*np.log10(np.abs(Pspk)/np.sqrt(2)/2e-5), c = 'r', ls = '--', label = 'Only speaker')
    plt.semilogx(freqs, 20*np.log10(np.abs(out.Pt)/np.sqrt(2)/2e-5), c = 'b', ls = ':', label = 'Top chamber, MAM')
    plt.semilogx(freqs, 20*np.log10(np.abs(outm.Pt)/np.sqrt(2)/2e-5), c = 'm', ls = ':', label = 'Top chamber, eq. m.')
    plt.semilogx(freqs, 20*np.log10(np.abs(outh.Pt)/np.sqrt(2)/2e-5), c = 'g', ls = ':', label = 'Top chamber, e. pl.')
    plt.ylabel(r'dB re $2\times10^{-5}$ Pa')
    plt.xlabel('Frequency, Hz')
    plt.ylim((0, 150))
    plt.grid()
    plt.legend(loc = 3)
    plt.savefig('SPL_' + name + '.png', dpi=600, bbox_inches = "tight")
    
    plt.figure(figsize = [7,7])
    plt.title('On-axis SPL at ' + str(dist) + ' m, A-weighted')
    plt.semilogx(freqs, 20*np.log10(np.abs(Pmam)/np.sqrt(2)/2e-5) + A, c = 'b', label = 'With MAM')
    plt.semilogx(freqs, 20*np.log10(np.abs(Pmass)/np.sqrt(2)/2e-5) + A, c = 'm', label = 'With equiv. mass')
    plt.semilogx(freqs, 20*np.log10(np.abs(Phole)/np.sqrt(2)/2e-5) + A, c = 'g', label = 'Empty plate')
    plt.semilogx(freqs, 20*np.log10(np.abs(Pspk)/np.sqrt(2)/2e-5) + A, c = 'r', ls = '--', label = 'Only speaker')
    plt.semilogx(freqs, 20*np.log10(np.abs(out.Pt)/np.sqrt(2)/2e-5) + A, c = 'b', ls = ':', label = 'Top chamber, MAM')
    plt.semilogx(freqs, 20*np.log10(np.abs(outm.Pt)/np.sqrt(2)/2e-5) + A, c = 'm', ls = ':', label = 'Top chamber, eq. m.')
    plt.semilogx(freqs, 20*np.log10(np.abs(outh.Pt)/np.sqrt(2)/2e-5) + A, c = 'g', ls = ':', label = 'Top chamber, e. pl.')
    plt.ylabel(r'dB re $2\times10^{-5}$ Pa')
    plt.xlabel('Frequency, Hz')
    plt.ylim((0, 150))
    plt.grid()
    plt.legend(loc = 3)
    plt.savefig('SPL(A)_' + name + '.png', dpi=600, bbox_inches = "tight")
    
    plt.figure(figsize = [7,7])
    plt.title('Unwrapped phase')
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(360))
    plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(Pmam))), c = 'b', label = 'With MAM')
    plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(Pmass))), c = 'm', label = 'With equiv. mass')
    plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(Phole))), c = 'g', label = 'Empty plate')
    plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(Pspk))), c = 'r', ls = '--', label = 'Only speaker')
    plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(out.Pt))), c = 'b', ls = ':', label = 'Top chamber, MAM')
    plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(outm.Pt))), c = 'm', ls = ':', label = 'Top chamber, eq. m.')
    plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(outh.Pt))), c = 'g', ls = ':', label = 'Top chamber, e. pl.')
    plt.ylabel('Degrees')
    plt.xlabel('Frequency, Hz')
    plt.grid()
    plt.legend(loc = 3)
    plt.savefig('phase_' + name + '.png', dpi=600, bbox_inches = "tight")
    
    plt.figure(figsize = [7,5])
    plt.title('Simulated (piston) excursion')
    plt.semilogx(freqs, np.abs(out.dspm)*1000, c = 'b', label = 'With MAM')
    plt.semilogx(freqs, np.abs(outm.dspm)*1000, c = 'm', label = 'With equiv. mass')
    plt.semilogx(freqs, np.abs(outh.dspm)*1000, c = 'g', label = 'Empty plate')
    plt.semilogx(freqs, np.abs(out.dspk)*1000, c = 'r', ls = '--', label = 'Only speaker')
    plt.semilogx(freqs, np.abs(out.dmam)*1000, c = 'b', ls = ':', label = 'MAM')
    plt.semilogx(freqs, np.abs(outm.dmam)*1000, c = 'm', ls = ':', label = 'Equiv. mass')
    plt.semilogx(freqs, np.ones(len(freqs))*driver.Xmax*1000, c = 'k', ls = '--', label = r'Spk. $X_{max}$')
    plt.ylabel(r'mm')
    plt.xlabel('Frequency, Hz')
    plt.grid()
    plt.legend()
    plt.savefig('exc_' + name + '.png', dpi=600, bbox_inches = "tight")
    
    plt.figure(figsize = [7,5])
    plt.title('Insertion loss (reference: no plate)')
    plt.semilogx(freqs, 20*np.log10(np.abs(Pspk)/np.sqrt(2)/2e-5) - 20*np.log10(np.abs(Pmam)/np.sqrt(2)/2e-5), c = 'b', label = 'MAM')
    plt.semilogx(freqs, 20*np.log10(np.abs(Pspk)/np.sqrt(2)/2e-5) - 20*np.log10(np.abs(Pmass)/np.sqrt(2)/2e-5), c = 'm', label = 'Equiv. mass')
    plt.semilogx(freqs, 20*np.log10(np.abs(Pspk)/np.sqrt(2)/2e-5) - 20*np.log10(np.abs(Phole)/np.sqrt(2)/2e-5), c = 'g', label = 'Empty plate')
    plt.ylabel(r'dB re $2\times10^{-5}$ Pa')
    plt.xlabel('Frequency, Hz')
    plt.grid()
    plt.legend()
    plt.savefig('IL1_' + name + '.png', dpi=600, bbox_inches = "tight")
    
    plt.figure(figsize = [7,5])
    plt.title('Insertion loss (reference: empty plate)')
    plt.semilogx(freqs, 20*np.log10(np.abs(Phole)/np.sqrt(2)/2e-5) - 20*np.log10(np.abs(Pmam)/np.sqrt(2)/2e-5), c = 'b', label = 'MAM')
    plt.semilogx(freqs, 20*np.log10(np.abs(Phole)/np.sqrt(2)/2e-5) - 20*np.log10(np.abs(Pmass)/np.sqrt(2)/2e-5), c = 'm', label = 'Equiv. mass')
    plt.ylabel(r'dB re $2\times10^{-5}$ Pa')
    plt.xlabel('Frequency, Hz')
    plt.grid()
    plt.legend()
    plt.savefig('IL2_' + name + '.png', dpi=600, bbox_inches = "tight")
    
    return

#%% example results
o1 = Padv(spk2, amp4, mam1a, enc2a, 1)

plt.figure(figsize = [7,7])
plt.title('On-axis SPL at 1 m')
plt.semilogx(freqs, 20*np.log10(np.abs(o1.Pmam)/np.sqrt(2)/2e-5), c = 'b', label = 'With MAM')
plt.semilogx(freqs, 20*np.log10(np.abs(o1.Pspk)/np.sqrt(2)/2e-5), c = 'r', ls = '--', label = 'W/o MAM')
plt.semilogx(freqs, 20*np.log10(np.abs(o1.Pt)/np.sqrt(2)/2e-5), c = 'm', ls = ':', label = 'Top chamber')
plt.ylabel(r'dB re $2\times10^{-5}$ Pa')
plt.xlabel('Frequency, Hz')
plt.ylim((0, 150))
plt.grid()
plt.legend(loc = 3)
#plt.savefig('SPL.png', dpi=600, bbox_inches = "tight")

plt.figure(figsize = [7,7])
plt.title('Unwrapped phase')
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(360))
plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(o1.Pmam))), c = 'b', label = 'With MAM')
plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(o1.Pspk))), c = 'r', ls = '--', label = 'W/o MAM')
plt.semilogx(freqs, np.rad2deg(np.unwrap(np.angle(o1.Pt))), c = 'm', ls = ':', label = 'Top chamber')
plt.ylabel('Degrees')
plt.xlabel('Frequency, Hz')
plt.grid()
plt.legend(loc = 3)
#plt.savefig('phase.png', dpi=600, bbox_inches = "tight")

plt.figure(figsize = [7,5])
plt.title('Simulated (piston) excursion')
plt.semilogx(freqs, np.abs(o1.dspm)*1000, c = 'b', label = 'Spk. with MAM')
plt.semilogx(freqs, np.abs(o1.dspk)*1000, c = 'r', ls = '--', label = 'Spk. w/o MAM')
plt.semilogx(freqs, np.abs(o1.dmam)*1000, c = 'tab:orange', ls = '-.', label = 'MAM')
plt.ylabel(r'mm')
plt.xlabel('Frequency, Hz')
plt.grid()
plt.legend()
#plt.savefig('exc.png', dpi=600, bbox_inches = "tight")

#IL for a speaker
plt.figure(figsize = [7,5])
plt.title('Insertion loss')
plt.semilogx(freqs, 20*np.log10(np.abs(o1.Pspk)/np.sqrt(2)/2e-5) - 20*np.log10(np.abs(o1.Pmam)/np.sqrt(2)/2e-5), c = 'b')
plt.ylabel('dB')
plt.xlabel('Frequency, Hz')
plt.grid()
#plt.savefig('IL.png', dpi=600, bbox_inches = "tight")

#saved predictions
PRempty = Padv(spk1, amp(0, 3.3), hole1, encf)
emptypred = np.array([PRempty.omegas, PRempty.Pmam, PRempty.PmamB])
PRmass = Padv(spk1, amp(0, 3.3), mass1, encf)
masspred = np.array([PRmass.omegas, PRmass.Pmam, PRmass.PmamB])
PRmam = Padv(spk1, amp(0, 3.3), mam1a, encf)
mampred = np.array([PRmam.omegas, PRmam.Pmam, PRmam.PmamB])

#adjusted paramters
mam2 = MAM((84e-3/2)**2*np.pi, 110, 1679, 400, 2.63, 1.16e10, 0, 180, 3.2e-3)
PRmam2 = Padv(spk1, amp(0, 3.3), mam2, encf)
mam2pred = np.array([PRmam2.omegas, PRmam2.Pmam, PRmam2.PmamB])

#%% plots for kR/ ka assumptions
k1 = w/c0
Vb = np.array([2*0.1**3, 2*0.2**3, 2*0.3**3])
R1 = (3*Vb/4/np.pi)**(1/3)


plt.figure()
for i in range(3):
    plt.semilogx(freqs, k1*R1[i], label = r'edge = ' + str(np.around((i + 1)*0.1, 1)) + ' m')
plt.axhline(1, c = 'r')
plt.legend()

a = 0.05
plt.figure()
plt.semilogx(freqs, k1*a)
plt.axhline(0.5, c = 'r')

#plot for round baffle step
plt.figure()
for i in range(3):
    plt.semilogx(freqs, 20*np.log10(np.abs((1 + 1j*k1*R1[i])/(1 + 1j*k1*R1[i]/2))), label = r'edge = ' + str(np.around((i + 1)*0.1, 1)) + ' m')
plt.legend()

#%% notes
#driver selection criteria
# ideally low fs, high cone mass, long VC
# Qts>0.3, Qtc~0.7 (ideally small)
# fs/Qes<50
# for 6-8', Xmax>2mm
