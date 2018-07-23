# _N,_K = 100,20
# PHCOLS0 = ['0l%d_%d' % (k,i) for k in range(_K) for i in range(_N)]
# PHCOLS1 = ['1l%d_%d' % (k,i) for k in range(_K) for i in range(_N)]
# PHCOLS = PHCOLS0 + PHCOLS1

# STFTCOLS = ['fa_%d' % i for i in range(1024)] + ['fb_%d' % i for i in range(1024)]


SNRS = [-10, -6, -2, 2, 6, 10]

MODS = ['16PSK','2FSK_5KHz','2FSK_75KHz','8PSK','AM_DSB','AM_SSB',
        'APSK16_c34','APSK32_c34','BPSK','CPFSK_5KHz','CPFSK_75KHz',
        'FM_NB','FM_WB','GFSK_5KHz','GFSK_75KHz','GMSK','MSK','NOISE',
        'OQPSK','PI4QPSK','QAM16','QAM32','QAM64','QPSK']               # all mods

MTOI = { MODS[i] : i for i in range(len(MODS)) }

ICOLS = ['MOD','MODi','SNR'] + MODS

COLS = ['a_%d' % i for i in range(1024)] + ['b_%d' % i for i in range(1024)]
COL_DICT = {COLS[i] : i for i in range(len(COLS))}

# COMP_GROUP_NAMES = ['AllMod','AllPurePhase','AllPF','AllPureAmplitude',
#         'AllAnalog','AllAnAM','AllAnFM','AllFSK','AllPSK','AltPSK','AllMSK',
#         'AllQAM','AllPhase','AllFreq','AllAmplitude','AllDigital']
#
# GROUP_NAMES = COMP_GROUP_NAMES + MODS
#
# COMP_GROUPS = [MODS,                                                         # all
#        ['16PSK','8PSK','BPSK','GMSK','MSK','OQPSK','PI4QPSK','QPSK'],   # pure phase
#        ['CPFSK_5KHz','CPFSK_75KHz'],                                    # pf
#        ['AM_DSB','AM_SSB','QAM16','QAM32','QAM64'],                     # pure amplitude
#        ['AM_DSB','AM_SSB','FM_NB','FM_WB'],                             # analog
#        ['AM_DSB','AM_SSB'],                                             # analog am
#        ['FM_NB','FM_WB'],                                               # analog fm
#        ['2FSK_5KHz','2FSK_75KHz','GFSK_5KHz','GFSK_75KHz'],             # fsk
#        ['16PSK','8PSK','BPSK','QPSK'],                                  # psk
#        ['OQPSK','PI4QPSK'],                                             # other psk
#        ['GMSK','MSK'],                                                  # msk
#        ['QAM16','QAM32','QAM64'],                                       # quam
#        ['16PSK','8PSK','APSK16_c34','APSK32_c34','BPSK','CPFSK_5KHz',
#        'CPFSK_75KHz','GMSK','MSK','OQPSK','PI4QPSK','QPSK'],            # all phase
#        ['2FSK_5KHz','2FSK_75KHz','CPFSK_5KHz',
#        'CPFSK_75KHz','GFSK_5KHz','GFSK_75KHz'],                         # freq
#        ['AM_DSB','AM_SSB','APSK16_c34',
#        'APSK32_c34','QAM16','QAM32','QAM64'],                           # amplitude
#        ['16PSK','2FSK_5KHz','2FSK_75KHz','8PSK','APSK16_c34',
#        'APSK32_c34','BPSK','CPFSK_5KHz','CPFSK_75KHz',
#        'GFSK_5KHz','GFSK_75KHz','GMSK','MSK','NOISE',
#        'OQPSK','PI4QPSK','QAM16','QAM32','QAM64','QPSK']]               # digital
#
# GROUPS = COMP_GROUPS + [[mod] for mod in MODS]

GROUP_NAMES = ['AllMod'] + MODS
GROUPS = [MODS] + [[mod] for mod in MODS]

GROUP_DICT = {GROUP_NAMES[i] : GROUPS[i] for i in range(len(GROUP_NAMES))}

def name_file(chunk, snr, suff=''):
    suff = '_'.join(['',suff]) if len(suff) > 0 else suff
    if snr == None or all(s in snr for s in SNRS):
        return 'chunk%d%s' % (chunk,suff)
    else:
        fsnr = ':'.join(str(s) for s in sorted(snr))
        return 'SNR' + fsnr + '_%d%s' % (chunk,suff)

def name_lcols(dim, n, k):
    return ['%dlandscape%d_%d' % (dim,k,i) for k in range(k) for i in range(n)]
