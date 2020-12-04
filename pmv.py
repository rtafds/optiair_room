# initial condition
def calc_PMV(TA=20,VA=0.3,TR=20,RH=50,AL=1,CLO=1):
    #AL = 1  # 活動量[met]
    #CLO = 1 # 着衣量[clo]
    #TA = 20  #  温度[℃]
    #TR = 20  # MRT[℃]
    #VA = 0.3  # 流速[m/s]
    #RH = 50  # 相対湿度[%]
    #
    #***************************************************
    # 外部仕事 W＝0 [W/㎡]とする。
    #***************************************************
    # PMV 計算準備
    #
    M = AL * 58.15
    LCL = CLO
    W = 0
    #PA = (RH / 100 * np.exp(18.6686 - 4030.18 / (TA + 235))) / 0.00750062
    PPK = 673.4 - 1.8 * TA
    PPA = 3.2437814 + 0.00326014 * PPK + 2.00658 * 1E-9 * PPK * PPK * PPK
    PPB = (1165.09 - PPK) * (1 + 0.00121547 * PPK)
    PA = RH / 100 * 22105.8416 / np.exp(2.302585 * PPK * PPA / PPB) * 1000
    EPS = 1E-5
    MW = M - W
    # FCL＝着衣表面積／裸体表面積の比
    if LCL > 0.5:
        FCL = 1.05 + 0.1 * LCL
    else:
        FCL = 1 + 0.2 * LCL
    # 衣服表面温度TCLの初期値設定
    TCL = TA
    TCLA = TCL
    NOI = 1
    # 着衣表面温度の計算
    while True:
        TCLA = 0.8 * TCLA + 0.2 * TCL
        HC = 12.1 * np.sqrt(VA)
        if 2.38 * np.sqrt(np.sqrt(abs(TCL - TA))) > HC:
            HC = 2.38 * np.sqrt(np.sqrt(abs(TCL - TA)))
        TCL = 35.7 - 0.028 * MW - 0.155 * LCL * (3.96 * 1E-8 * FCL * ((TCLA + 273) ** 4 - (TR + 273) ** 4) + FCL * HC * (TCLA - TA))
        NOI = NOI + 1
        if NOI > 150:
            PMV = 999990.999
            PPD = 100
            return (PMV,PPD)
        if not abs(TCLA - TCL) > EPS:
            break
    #PMVの計算
    PM1 = 3.96 * 1E-8 * FCL * ((TCL + 273) ** 4 - (TA + 273) ** 4)
    PM2 = FCL * HC * (TCL - TA)
    PM3 = 0.303 * np.exp(-0.036 * M) + 0.028
    if MW > 58.15:
        PM4 = 0.42 * (MW - 58.15)
    else:
        PM4 = 0
    PMV = PM3 * (MW - 3.05 * 0.001 * (5733 - 6.99 * MW - PA) - PM4 - 1.7 * 1E-5 * M * (5867 - PA) - 0.0014 * M * (34 - TA) - PM1 - PM2)
        #PRINT PMV
    if abs(PMV) > 3:
        PMV = 999990.999
        PPD = 100
        return (PMV,PPD)
    
    PPD = 100 - 95 * np.exp(-0.0335 * PMV ** 4 - 0.2179 * PMV ** 2)
    
    return(PMV,PPD)