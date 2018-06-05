import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import embed

df = pd.read_csv('scalability_main.csv')

plt.style.use('classic')
w = 2.5

if True:
    import pickle
    time2d, time3d, time4d, timesum, timedc = pickle.load(open('../results/scalability_running.pkl', 'rb'))
    dom2d = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    dom3d = np.array([16, 32, 64, 128, 256, 512, 1024])
    dom4d = np.array([16, 32, 64, 128])

    dom2d = dom2d[1:]
    dom3d = dom3d[1:]
    dom4d = dom4d[1:]
    domdc = np.arange(2, 10)

    df = pd.read_csv('../results/scalability_other.csv')

    w = 4.5
    s = 12
    ticksize = 25 #'xx-large'
    labelsize = 30 #'xx-large'

    x_low = 1
    x_high = 2*10**9
    y_low = 0.0 # 5e-2
    y_high = 4000


    plt.rc('text', usetex=True)
    #plt.figure(figsize=(8,6.5))
    #plt.plot(dom2d**2, time2d, 'ro-', markersize=6, linewidth=w,label='$OPT_{\otimes}: N = n^2$')
    #plt.plot(dom4d**4, time4d, 'rD-', markersize=6, linewidth=w,label='$OPT_{\otimes}: N = n^4$')
    plt.plot(dom2d**2, timesum, 'k:', marker='$+$', markersize=s, linewidth=w,label='$OPT_+$')
    plt.plot(10**domdc[1:], timedc[1:], 'ks-', markersize=s, linewidth=w,label='$OPT_M$')
    plt.plot(dom3d**3, time3d, 'kx--', marker='$\otimes$', markersize=s, linewidth=w,label='$OPT_{\otimes}$')
    #plt.axes().set_aspect('equal')

    #plt.plot(df.Domain, df.Identity, 'co-', markersize=6, linewidth=w, label='Identity')
    #plt.plot(df.Domain, df.Privelet, 'cD-', markersize=6, linewidth=w, label='Privelet')
    #plt.plot(df.Domain, df.HB, 'cs-', markersize=6, linewidth=w, label='HB')
    #plt.plot(df.Domain, df.QuadTree, 'c*-', markersize=6, linewidth=w, label='QuadTree')

    #plt.loglog()
    plt.xscale('log')
    plt.yscale('symlog', linthreshy=1.0)
    plt.legend(loc='upper left', fontsize='xx-large')
    plt.title(' ', fontsize=labelsize)
    plt.xlim(10**3/2,10**9*2)
    plt.xlabel('Domain Size', fontsize=labelsize)
    #plt.ylabel('Time (s)', fontsize='xx-large')
    plt.xticks([1, 10**3, 10**6, 10**9], fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    #plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('scalability_mechanism.png')
    plt.savefig('scalability_mechanism.pdf')
    plt.show()


if False:
    w = 4.5
    s = 12

    x_low = 1
    x_high = 2*10**9
    y_low = 0.0 # 5e-2
    y_high = 4000

    labelsize = 30 #'xx-large'
    ticksize = 25 #'xx-large'
    #scale = plt.loglog 
    #scale = plt.semilogx
    def scale():
        plt.xscale('log')
        plt.yscale('symlog', linthreshy=1.0)
    layout = lambda: plt.gcf().subplots_adjust(bottom=0.15) #plt.tight_layout
    
    ## 1D Scalability ##

    df = pd.read_csv('scalability_main.csv')
    lrm = df[df.Mechanism == 'LRM 1D']
    hdmm = df[df.Mechanism == 'HDMM 1D']
    hdmm = hdmm.iloc[:-2]
    greedyh = df[df.Mechanism == 'GreedyH 1D']
    plt.plot(lrm.Domain, lrm.Time, 'bo-', linewidth=w, markersize=s, label='LRM')
    plt.plot(greedyh.Domain, greedyh.Time, 'ro-', linewidth=w, markersize=s, label='GreedyH')
    plt.plot(hdmm.Domain, hdmm.Time, 'ko-', linewidth=w, markersize=s, label='HDMM')
    plt.plot([], [], 'go-', linewidth=w, markersize=s, label='DataCube')
    scale()
    plt.title('Workload: Prefix (1D)', fontsize=labelsize)
    plt.legend(loc='upper right', fontsize='xx-large')
    #ax1.set_xlabel('Domain Size', fontsize=labelsize)
    plt.xlabel('Domain Size ($N = n$)', fontsize=labelsize)
    plt.ylabel('Time (s)', fontsize=labelsize)
    plt.xticks([1, 10**3, 10**6, 10**9], fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    layout()
    plt.savefig('scalability-revised-1d.pdf')
    plt.show()

    ## 3D Scalability ##

    lrm = df[df.Mechanism == 'LRM 3D']
    hdmm = df[df.Mechanism == 'HDMM 3D']
    plt.plot(lrm.Domain, lrm.Time, 'bo-', linewidth=w, markersize=s, label='LRM')
    plt.plot(hdmm.Domain, hdmm.Time, 'ko-', linewidth=w, markersize=s, label='HDMM')
    scale()
    plt.title('Workload: Prefix (3D)', fontsize=labelsize)
    plt.xlabel('Domain Size ($N = n^3$)', fontsize=labelsize)
    plt.xticks([1, 10**3, 10**6, 10**9], fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    layout()
    plt.savefig('scalability-revised-3d.pdf')
    plt.show()

    hdmm = df[df.Mechanism == 'HDMM 8D']
    datacube = df[df.Mechanism == 'DataCube 8D']
    lrm = df[df.Mechanism == 'LRM 8D']
    plt.plot(datacube.Domain, datacube.Time, 'go-', linewidth=w, markersize=s, label='DataCube')
    plt.plot(hdmm.Domain, hdmm.Time, 'ko-', linewidth=w, markersize=s, label='HDMM')
    plt.plot(lrm.Domain, lrm.Time, 'bo-', linewidth=w, markersize=s, label='LRM')
    scale()
    plt.title('Workload: 3 Way Marginals (8D)', fontsize=labelsize)
    plt.xlabel('Domain Size ($N=n^8$)', fontsize=labelsize)
    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    plt.xticks([1, 10**3, 10**6, 10**9], fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    layout()
    plt.savefig('scalability-revised-8d.pdf')
    plt.show()
  

   

if False:
    
    w = 4.5
    s = 12

    x_low = 1
    x_high = 2*10**9
    y_low = 5e-2
    y_high = 1500

    labelsize = 'xx-large'
    ticksize = 'x-large'
    scale = plt.loglog # plt.semilogx

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(20, 5))

    ## 1D Scalability ##

    lrm = df[df.Mechanism == 'LRM 1D']
    hdmm = df[df.Mechanism == 'HDMM 1D']
    hdmm = hdmm.iloc[:-2]
    greedyh = df[df.Mechanism == 'GreedyH 1D']
    ax1.plot(lrm.Domain, lrm.Time, 'bo-', linewidth=w, markersize=s, label='LRM')
    ax1.plot(greedyh.Domain, greedyh.Time, 'ro-', linewidth=w, markersize=s, label='GreedyH')
    ax1.plot(hdmm.Domain, hdmm.Time, 'ko-', linewidth=w, markersize=s, label='HDMM')
    ax1.plot([], [], 'go-', linewidth=w, markersize=s, label='DataCube')
    ax1.set_title('(a) Workload: Prefix (1D) \n N/A: DataCube', fontsize='x-large')
    ax1.legend(loc='upper right', fontsize='xx-large')
    #ax1.set_xlabel('Domain Size', fontsize=labelsize)
    ax1.set_ylabel('Time (s)', fontsize=labelsize)
    #ax1.set_xticks(fontsize=ticksize)
    #ax1.set_yticks(fontsize=ticksize)
    ax1.set_xlim(x_low, x_high)
    ax1.set_ylim(y_low, y_high)
    #ax1.loglog()
    ax1.semilogx()
    #scale()
    #plt.savefig('scalability-revised-1d.pdf')
    #plt.show()

    ## 3D Scalability ##

    lrm = df[df.Mechanism == 'LRM 3D']
    hdmm = df[df.Mechanism == 'HDMM 3D']
    ax2.plot(lrm.Domain, lrm.Time, 'bo-', linewidth=w, markersize=s, label='LRM')
    ax2.plot(hdmm.Domain, hdmm.Time, 'ko-', linewidth=w, markersize=s, label='HDMM')
    ax2.set_title('(b) Workload: Prefix (3D) \n N/A: GreedyH, DataCube', fontsize='x-large')
    #ax2.legend(loc='lower right', fontsize='xx-large')
    ax2.set_xlabel('Domain Size', fontsize=labelsize)
    #ax2.ylabel('Time (s)', fontsize=labelsize)
    #ax2.xticks(fontsize=ticksize)
    #ax2.yticks(fontsize=ticksize)
    #ax2.xlim(x_low, x_high)
    #ax2.ylim(y_low, y_high)
    ax2.set_xlim(x_low, x_high)
    #ax2.loglog()
    ax2.semilogx()
    #scale() 
    #plt.savefig('scalability-revised-3d.pdf')
    #plt.show()

    hdmm = df[df.Mechanism == 'HDMM 8D']
    datacube = df[df.Mechanism == 'DataCube 8D']
    lrm = df[df.Mechanism == 'LRM 8D']
    ax3.plot(datacube.Domain[:-1], datacube.Time[:-1], 'go-', linewidth=w, markersize=s, label='DataCube')
    ax3.plot(hdmm.Domain, hdmm.Time, 'ko-', linewidth=w, markersize=s, label='HDMM')
    ax3.plot(lrm.Domain, lrm.Time, 'bo-', linewidth=w, markersize=s, label='LRM')
    ax3.set_title('(c) Workload: 3 Way Marginals (8D) \n N/A: GreedyH', fontsize='x-large')
    #ax3.loglog()
    ax3.semilogx()
    ax3.set_xlim(x_low, x_high)
    #ax3.legend(loc='lower right', fontsize='xx-large')
    #ax3.xlabel('Domain Size', fontsize=labelsize)
    #ax3.ylabel('Time (s)', fontsize=labelsize)
    #ax3.xticks(fontsize=ticksize)
    #ax3.yticks(fontsize=ticksize)
    #ax3.xlim(x_low, x_high)
    #ax3.ylim(y_low, y_high)
    #scale()
    #plt.savefig('scalability-revised-8d.pdf')
    #plt.show()
  
    ax1.tick_params(labelsize=ticksize) 
    ax2.tick_params(labelsize=ticksize) 
    ax3.tick_params(labelsize=ticksize) 
    ax1.set_xticks([1, 10**3, 10**6, 10**9])
    ax2.set_xticks([1, 10**3, 10**6, 10**9])
    ax3.set_xticks([1, 10**3, 10**6, 10**9])
    #plt.tight_layout()
    plt.savefig('scalability-revised-all.pdf', bbox_inches='tight')
    plt.show()



if False:

    lrm = df[df.Mechanism == 'LRM']
    hdmm = df[df.Mechanism == 'HDMM']
    greedyh = df[df.Mechanism == 'GreedyH']
    datacube = df[df.Mechanism == 'DataCube']

    plt.style.use('classic')
    plt.plot(lrm.Domain, lrm.Time, 'bo-', linewidth=w, label='Low Rank Mechanism')
    plt.plot(greedyh.Domain, greedyh.Time, 'ro-', linewidth=w, label='GreedyH')
    plt.plot(datacube.Domain, datacube.Time, 'go-', linewidth=w, label='DataCube')
    plt.plot(hdmm.Domain, hdmm.Time, 'ko-', linewidth=w, label='HDMM')

    plt.legend(loc='upper left', fontsize='medium')
    plt.xlabel('Domain Size', fontsize='x-large')
    plt.ylabel('Time (s)', fontsize='x-large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.semilogx()
    #plt.loglog()
    plt.savefig('scalability-revised.pdf')
    plt.show()

if False:
    w = 4.5
    s = 12

    df = pd.read_csv('../results/togus_scalability.csv')

    plt.plot(df.domain, df.time, 'ko-', linewidth=w, markersize=s, label='$OPT_0$')
    plt.legend(loc='lower right', fontsize='xx-large')
    plt.xlabel('Domain Size', fontsize='xx-large')
    plt.ylabel('Time (s)', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.loglog()
    plt.savefig('scalability-opt0.pdf')
    plt.show()


    df = pd.read_csv('../results/togus_scalability_marginals_uniform.csv')
    df = df.groupby('dims').time.mean()[:-1]

    plt.plot(df.index[::2], df.values[::2], 'ks-', linewidth=w, markersize=s, label='$OPT_M$')
    plt.legend(loc='lower right', fontsize='xx-large')
    plt.xlabel('# Dimensions', fontsize='xx-large')
    plt.ylabel('Time (s)', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.yscale('log')
    plt.savefig('scalability-optm.pdf')
    plt.show()

