"""CLs limit setting for simplified model combination project"""

from __future__ import division
import numpy as np
from numpy.lib import recfunctions
import os
import clstools.clstools as tools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter
from matplotlib import colors
import scipy.interpolate as interp

#tmp
from mpl_toolkits.mplot3d import Axes3D

import sys

# Get value of SSYS to use from the command line

print sys.argv
SSYS = np.float(sys.argv[1])
print SSYS

baseoutdir='results_s-{0}'.format(SSYS)

#=======================================================================
# Define experiments
#=======================================================================

# We are dealing with the following four analyses (cout statements give the format in the data files provided by MArtin)

#0 LEP: ATLAS-CONF-2013-024
#cout << "RESULTS 0LEP " << _numSR1 << " " <<  _numSR2 << " " << _numSR3 << endl;

#1 LEP: ATLAS-CONF-2013-037
# cout << "RESULTS 1LEP " << _numTN1Shape_bin1 << " " <<  _numTN1Shape_bin2 << " " << _numTN1Shape_bin3 << " " << _numTN2 << " " <<  _numTN3 << " " << _numBC1 << " " << _numBC2 << " " << _numBC3 << endl;

#2 LEP: ATLAS-CONF-2013-048
#cout << "RESULTS 2LEP " << _numSRM90SF << " " << _numSRM100SF << " " << _numSRM110SF << " " << _numSRM120SF << " " << _numSRM90DF << " " << _numSRM100DF << " " << _numSRM110DF << " " << _numSRM120DF << endl;

#2 b: arXiv:1308.2631
#cout << "RESULTS 2B " << _numSRA << " "  << _numSRB << " " << _numSRA15 << " " <<  _numSRA20 << " " <<  _numSRA25 << " " << _numSRA30 << " " <<  _numSRA35 << endl;

#-----------------------------------------------------------------------

# 0 LEP: ATLAS-CONF-2013-024
# _numSR1, _numSR2, _numSR3
# Background systematic uncertainties from table 3. May be correlated between signal regions.
# Signal systematics pg 11:
# "For the signal prediction, the systematic uncertainty is approximately 30%, constant across the plane of top squark and LSP masses. The uncertainty is dominated by the uncertainties on the jet energy scale (25%) and b-tagging (15%) as well as by the theoretical uncertainties on the cross section (14%).
# tuning notes: Observed limit is being overestimated, so turning down the signal efficiency scaling parameter
SR1_0lep = tools.Experiment(name='SR1_0lep', o=15, ssys=SSYS, b=17.5, bsys=0.18, sK=1., outdir=baseoutdir)
SR2_0lep = tools.Experiment(name='SR2_0lep', o=2 , ssys=SSYS, b= 4.7, bsys=0.33, sK=1., outdir=baseoutdir)
SR3_0lep = tools.Experiment(name='SR3_0lep', o=1 , ssys=SSYS, b= 2.7, bsys=0.45, sK=1., outdir=baseoutdir)

# For tuning purposes we also use the official observed and expected limits on the number of signal events in each signal region (table 4):
SR_obslim_0lep = [10.0,3.6,3.9]
SR_explim_0lep = [10.6,5.3,4.5]

SRs_0lep = [SR1_0lep,SR2_0lep,SR3_0lep]

L_0lep = 20.5 #fb^-1 - integrated luminosity used for 0 lep. analysis
N_0lep = 100000 # Number of detector events simulated by Martin

#-----------------------------------------------------------------------

# 1 LEP: ATLAS-CONF-2013-037
# _numTN1Shape_bin1, _numTN1Shape_bin2, _numTN1Shape_bin3, _numTN2, _numTN3, _numBC1, _numBC2, _numBC3 
# Data from tables 2-4
# Signal systematics estimated initially from table 5:
# "For the signal prediction, the systematic uncertainty is approximately 30%, constant across the plane of top squark and LSP masses. The uncertainty is dominated by the uncertainties on the jet energy scale (25%) and b-tagging (15%) as well as by the theoretical uncertainties on the cross section (14%).
# tuning notes: Observed limit is being overestimated, so turning down the signal efficiency scaling parameter
SRtN2_1lep = tools.Experiment(name='SRtN2_1lep', o=14 , ssys=SSYS, b=13. , bsys=3./13.  , sK=1., outdir=baseoutdir)
SRtN3_1lep = tools.Experiment(name='SRtN3_1lep', o=7  , ssys=SSYS, b=5.  , bsys=2./5.   , sK=1., outdir=baseoutdir)

SRbC1_1lep = tools.Experiment(name='SRbC1_1lep', o=456, ssys=SSYS, b=482., bsys=76./482., sK=1., outdir=baseoutdir)
SRbC2_1lep = tools.Experiment(name='SRbC2_1lep', o=25 , ssys=SSYS, b=18. , bsys=5./18.  , sK=1., outdir=baseoutdir)
SRbC3_1lep = tools.Experiment(name='SRbC3_1lep', o=6  , ssys=SSYS, b=7.  , bsys=3./7.   , sK=1., outdir=baseoutdir)

# Only the 3 most constraining bins of the shape data are used in the ATLAS fit
# (data in top row of figure 2)
SRtN1shape1_1lep = tools.Experiment(name='SRtN1.shape(1)_1lep', o=253, ssys=SSYS, b=250., bsys=57./250., sK=1 , outdir=baseoutdir)
SRtN1shape2_1lep = tools.Experiment(name='SRtN1.shape(2)_1lep', o=165, ssys=SSYS, b=174., bsys=28./174., sK=1., outdir=baseoutdir)
SRtN1shape3_1lep = tools.Experiment(name='SRtN1.shape(3)_1lep', o=235, ssys=SSYS, b=262., bsys=34./262., sK=1., outdir=baseoutdir)

# For tuning purposes we also use the official observed and expected limits on the number of signal events in each signal region (table 8):
#(SRtN2,SRtN3,SRbC1,SRbC2,SRbC3,SRtN_shape_bin1,SRtN_shape_bin2,SRtN_shape_bin3)
SR_obslim_1lep = [10.7,8.5,83.2,19.5,7.6,85.7,49.8,38.9]
SR_explim_1lep = [10.0,6.7,97.6,15.7,7.6,89.8,45.0,55.5]

SRs_1lep = [SRtN2_1lep,SRtN3_1lep,SRbC1_1lep,SRbC2_1lep,SRbC3_1lep,SRtN1shape1_1lep,SRtN1shape2_1lep,SRtN1shape3_1lep
]

L_1lep = 20.7 #fb^-1 - integrated luminosity used for 1 lep. analysis
N_1lep = 100000 # Number of detector events simulated by Martin

#-----------------------------------------------------------------------

# 2 LEP: ATLAS-CONF-2013-048
# _numSRM90SF, _numSRM100SF, _numSRM110SF, _numSRM120SF, _numSRM90DF, _numSRM100DF, _numSRM110DF, _numSRM120DF
# Data from tables 7
# Signal systematics initially estimated from table 7:
# "For the signal prediction, the systematic uncertainty is approximately 30%, constant across the plane of top squark and LSP masses. The uncertainty is dominated by the uncertainties on the jet energy scale (25%) and b-tagging (15%) as well as by the theoretical uncertainties on the cross section (14%).
# tuning notes: Observed limit is being overestimated, so turning down the signal efficiency scaling parameter
SRM90_2lep  = tools.Experiment(name='SRM90_2lep' , o=260, ssys=SSYS, b=300., bsys=40./300., sK=1., outdir=baseoutdir)
SRM100_2lep = tools.Experiment(name='SRM100_2lep', o=3  , ssys=SSYS, b=4.8 , bsys=2.2/4.8 , sK=1., outdir=baseoutdir)
SRM110_2lep = tools.Experiment(name='SRM110_2lep', o=7  , ssys=SSYS, b=11. , bsys=4./11.  , sK=1., outdir=baseoutdir)
SRM120_2lep = tools.Experiment(name='SRM120_2lep', o=3  , ssys=SSYS, b=4.3 , bsys=1.3/4.3 , sK=1., outdir=baseoutdir)

SRs_2lep = [SRM90_2lep,SRM100_2lep,SRM110_2lep,SRM120_2lep]

L_2lep = 20.3 #fb^-1 - integrated luminosity used for 2 lep. analysis
N_2lep = 100000 # Number of detector events simulated by Martin

# For tuning purposes we also use the official observed and expected limits on the number of signal events in each signal region (table 8):
# (SRM90_2lep,SRM100_2lep,SRM110_2lep,SRM120_2lep)
# Have observed and expected limits on cross section; multiply by luminosity to get number of events
SR_obslim_2lep = L_2lep*np.array([2.5,0.27,0.40,0.23])
SR_explim_2lep = L_2lep*np.array([3.5,0.30,0.42,0.27])

#-----------------------------------------------------------------------

# 2 b-jets : http://arxiv.org/pdf/1308.2631v1.pdf 
# _numSRA, _numSRB, _numSRA15, _numSRA20, _numSRA25, _numSRA30, _numSRA35
# Data from table 6
# Signal systematics pg 11:
# "For the signal prediction, the systematic uncertainty is approximately 30%, constant across the plane of top squark and LSP masses. The uncertainty is dominated by the uncertainties on the jet energy scale (25%) and b-tagging (15%) as well as by the theoretical uncertainties on the cross section (14%).
# tuning notes: Observed limit is being overestimated, so turning down the signal efficiency scaling parameter
SRA150_2b = tools.Experiment(name='SRA150_2b', o=102, ssys=SSYS, b=94. , bsys=13./94. , sK=1., outdir=baseoutdir)
SRA200_2b = tools.Experiment(name='SRA200_2b', o=48 , ssys=SSYS, b=39. , bsys=6./39.  , sK=1., outdir=baseoutdir)
SRA250_2b = tools.Experiment(name='SRA250_2b', o=14 , ssys=SSYS, b=15.8, bsys=2.8/15.8, sK=1., outdir=baseoutdir)
SRA300_2b = tools.Experiment(name='SRA300_2b', o=7  , ssys=SSYS, b=5.9 , bsys=1.1/5.9 , sK=1., outdir=baseoutdir)
SRA350_2b = tools.Experiment(name='SRA350_2b', o=3  , ssys=SSYS, b=2.5 , bsys=0.6/2.5 , sK=1., outdir=baseoutdir)
SRB_2b    = tools.Experiment(name='SRB_2b'   , o=65 , ssys=SSYS, b=64. , bsys=10./64. , sK=1., outdir=baseoutdir)

# For tuning purposes we also use the official observed and expected limits on the number of signal events in each signal region (table 7):
SR_obslim_2b = [38,26,9.0, 7.5,5.2,27]
SR_explim_2b = [32,19,10.2,6.5,4.7,26]

SRs_2b = [SRA150_2b,SRA200_2b,SRA250_2b,SRA300_2b,SRA350_2b,SRB_2b]

L_2b = 20.1 #fb^-1 - integrated luminosity used for 2 b-jet analysis
N_2b = 100000 # Number of detector events simulated by Martin

#-----------------------------------------------------------------------

#===================================================================================
#  Do analysis
#  

#Script options
XSxBR_limits_only = True
#genpdfs=True   #turn this to false once pdfs have been generated sufficiently accurately
#regenCL='ifNone'  #turn to false once CL values have been generated for the given input parameters
regenCL=True
#regenQdist='ifNone'
regenQdist=True
#svals = np.concatenate((np.arange(0.05,1,0.01),np.arange(1,4,0.02),np.arange(5,10,0.1),np.arange(10,20,0.5),np.arange(20,30,1)))
#svals =  np.concatenate((np.arange(0.001,5,0.5),np.arange(5,50,2)))   #fast version
#svals =  np.concatenate((np.arange(0.001,5,0.5),np.arange(5,20,2),np.arange(20,60,5)))   #faster version
#svals =  np.concatenate((np.arange(0.001,5,1),np.arange(5,20,3)))   #fastest version
svals =  np.concatenate((np.arange(0.001,5,0.5),np.arange(5,20,2),np.arange(20,50,5),np.arange(50,100,10),np.arange(100,200,20)))   #extended version



# Test mcmc
# Run mcmc with various chain lengths, see how CLs varies
# (want to minimise the mcmc samples needed for good stability)
"""
CLstests = []
for i in range(100):
   print i
   CLs,svals = SR1_0lep.getCL('tmp',15,svals,regen=True,method='simulate',N=20000)
   CLstests+=[CLs]

print CLstests
print np.mean(CLstests)
print np.std(CLstests)

quit()

keysSB, prSB = SR1_0lep.get_QCLs_dist_cpp(s=10,muT=1)
keysB,  prB  = SR1_0lep.get_QCLs_dist_cpp(s=10,muT=0)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.stem(keysSB,prSB,markerfmt='ro',linefmt='r-') 
ax.stem(keysB,prB,markerfmt='bo',linefmt='b-')
plt.show()
 
quit()
"""
# Extract simulated data from text files
dnamesSIM = ( 'strRESULTS1','str0LEP','numSR1','numSR2','numSR3',\
              'strRESULTS2','str1LEP','numTN1Shape_bin1','numTN1Shape_bin2','numTN1Shape_bin3','numTN2','numTN3','numBC1','numBC2','numBC3',\
              'strRESULTS3','str2LEP','numSRM90SF','numSRM100SF','numSRM110SF','numSRM120SF','numSRM90DF','numSRM100DF','numSRM110DF','numSRM120DF',\
              'strRESULTS4','str2B','numSRA','numSRB','numSRA15','numSRA20','numSRA25','numSRA30','numSRA35' )
dfmatSIM =  ( 'S16','S16','f4','f4','f4',\
              'S16','S16','f4','f4','f4','f4','f4','f4','f4','f4',\
              'S16','S16','f4','f4','f4','f4','f4','f4','f4','f4',\
              'S16','S16','f4','f4','f4','f4','f4','f4','f4' )

TNmasses = ('Mstop','Mneut')
dtypeTN = { 'names'  : TNmasses+dnamesSIM,
            'formats': ('f4','f4')+dfmatSIM }
BCmasses = ('Mstop','Mneut','Mchar')
dtypeBC = { 'names'  : BCmasses+dnamesSIM,
            'formats': ('f4','f4','f4')+dfmatSIM }
MIXmasses = ('Mstop','Mneut','Mchar')
dtypeMIX = {'names'  : MIXmasses+dnamesSIM,
            'formats': ('f4','f4','f4')+dfmatSIM }

#(the invalid_raise=False flag lets us skip records with missing data)
#Stop pair production with both stops decaying to top neutralino
data_tn  = np.genfromtxt("data/top_neutralino.dat",dtype=dtypeTN,invalid_raise=False)
#Stop pair production with both stops decaying to b chargino.
data_bc  = np.genfromtxt("data/bchargino_merged.dat",dtype=dtypeBC,invalid_raise=False,skiprows=1)
#Stop pair production with one stop decaying to b chargino and the other decaying to top neutralino.
data_mix = np.genfromtxt("data/mixed_merged.dat",dtype=dtypeMIX,invalid_raise=False)

#--------------------------------------
# Can fiddle with the entries of this list to produce plots for some subset of the models

modeldata = []
modeldata += [('top_neutralino', data_tn,  TNmasses )]
modeldata += [('bchargino',      data_bc,  BCmasses )]
modeldata += [('mixed',          data_mix, MIXmasses)]
#--------------------------------------

modelnames = [name   for name,data,masses in modeldata]
datalist   = [data   for name,data,masses in modeldata]
masslists  = [masses for name,data,masses in modeldata]

# Also need the production cross sections
dataXS = np.loadtxt("data/TNGridxsSum.dat")
# The cross sections only vary with stop mass; make an interpolating function out of it.
uniqM,sorti = np.unique(dataXS[:,0],return_index=True) #get indices of elements with unique stop masses
XSifunc = interp.interp1d(uniqM,dataXS[sorti,2])

# Append to data arrays
newdatalist = []
for data in datalist:
   
   data = recfunctions.rec_append_fields(data,"XS",XSifunc(data['Mstop'])*1000) #convert units to fb

   # Append entries for signal yield predictions in each signal region
   Nprod_0lep = L_0lep*data["XS"] # Number of stop pairs produced (integrated luminosity * production cross section)
   Nprod_1lep = L_1lep*data["XS"]
   Nprod_2lep = L_2lep*data["XS"]
   Nprod_2b   = L_2b*data["XS"]  
   
   # Fraction of stop pairs that decayed into events ID'd as top + neutralino
   #  =(e.g.) numSR1/N_0lep (number of events counted / total number of stop pairs simulated)
   
   # Mean numbers of signal events predicted in the various signal regions:
   # 0 lep:
   SR_sdata_0lep = ["SR1_s","SR2_s","SR3_s"]
   data = recfunctions.rec_append_fields(data,\
            SR_sdata_0lep,\
            [Nprod_0lep*data['numSR1']/N_0lep,\
             Nprod_0lep*data['numSR2']/N_0lep,\
             Nprod_0lep*data['numSR3']/N_0lep,\
            ]
          )
   
   # 1 lep:
   SR_sdata_1lep = ["SRtN2_s","SRtN3_s",\
                    "SRbC1_s","SRbC2_s","SRbC3_s",\
                    "SRtN1.shape(1)_s","SRtN1.shape(2)_s","SRtN1.shape(3)_s"]
   data = recfunctions.rec_append_fields(data,
             SR_sdata_1lep,\
            [Nprod_1lep*data['numTN2']/N_1lep,\
               Nprod_1lep*data['numTN3']/N_1lep,\
             Nprod_1lep*data['numBC1']/N_1lep,\
               Nprod_1lep*data['numBC2']/N_1lep,\
               Nprod_1lep*data['numBC3']/N_1lep,\
             Nprod_1lep*data['numTN1Shape_bin1']/N_1lep,\
               Nprod_1lep*data['numTN1Shape_bin1']/N_1lep,\
               Nprod_1lep*data['numTN1Shape_bin1']/N_1lep,\
            ]
          )
   
   # 2 lep:
   # Have to sum the predictions for SF and DF regions in this case
   SR_sdata_2lep = ["SRM90_s","SRM100_s","SRM110_s","SRM120_s"]
   data = recfunctions.rec_append_fields(data,
            SR_sdata_2lep,\
            [Nprod_2lep*(data['numSRM90SF'] +data['numSRM90DF'] )/N_2lep,\
             Nprod_2lep*(data['numSRM100SF']+data['numSRM100DF'])/N_2lep,\
             Nprod_2lep*(data['numSRM110SF']+data['numSRM110DF'])/N_2lep,\
             Nprod_2lep*(data['numSRM120SF']+data['numSRM120DF'])/N_2lep,\
            ]
          )
   
   # 2 b-jet:
   SR_sdata_2b = ["SRA150_s","SRA200_s","SRA250_s","SRA300","SRA350","SRB"]
   data = recfunctions.rec_append_fields(data,\
             SR_sdata_2b,\
            [Nprod_2b*data['numSRA15']/N_2b,\
             Nprod_2b*data['numSRA20']/N_2b,\
             Nprod_2b*data['numSRA25']/N_2b,\
             Nprod_2b*data['numSRA30']/N_2b,\
             Nprod_2b*data['numSRA35']/N_2b,\
             Nprod_2b*data['numSRB']/N_2b,\
            ]
          )
   
   newdatalist += [data]

datalist = newdatalist
   

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(data['mass1'],data['mass2'],data['SR1_s'])
#plt.show()
#quit()

#fig=plt.figure()
#plt.scatter(dataXS[:,0],dataXS[:,1],color='r',alpha=0.3)
#plt.scatter(data['mass1'],data['mass2'],color='b',alpha=0.3)
#plt.show()
#quit()

# Extract ATLAS official limits from file
#files = ["data3/atlasSRA.txt", "data3/atlasSRB.txt", "data3/atlasSRC.txt", "data3/ATLAS-1fb-limit.dat"]
#reallimits = [np.loadtxt(f) for f in files]
#sri = [2,3,4,5]   #columns of pointdata containing counts for each signal region


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(data['mass1'],data['mass2'],data['numSR3'])
#plt.show()
#quit()


# Plot limits!   a.getCLvals(svals,regen=regenCLs)

Anames = ['0lep','1lep','2lep','2b']

SR_experiments = [SRs_0lep,
                  SRs_1lep,
                  SRs_2lep,
                  SRs_2b]
                  
SR_sfields = [SR_sdata_0lep,
              SR_sdata_1lep,
              SR_sdata_2lep,
              SR_sdata_2b]

SR_obslims = [SR_obslim_0lep,
              SR_obslim_1lep,
              SR_obslim_2lep,
              SR_obslim_2b]

SR_explims = [SR_explim_0lep,
              SR_explim_1lep,
              SR_explim_2lep,
              SR_explim_2b]

# Now enter loop to produce results for each simplified model:
firstmodel=True
for modeli,(data,modelname,masslist) in enumerate(zip(datalist,modelnames,masslists)):
   # some calculations are model independent: do these once only.
   if modeli>0:
      firstmodel=False   

   for Aname, SR_exps, SR_sdata, SR_olims, SR_elims \
    in zip(Anames,SR_experiments, SR_sfields, SR_obslims, SR_explims):
      Nexps = len(SR_exps)
   
      outdir = baseoutdir+'/'+Aname
      if not os.path.exists(outdir):
        os.makedirs(outdir) 
      if not os.path.exists("{0}/{1}".format(outdir,modelname)):
        os.makedirs("{0}/{1}".format(outdir,modelname)) 

      #BRS = np.arange(0.,50.05,0.05)  # branching ratio slices on which to compute CL values
      BRS = np.concatenate((np.arange(0.,1.,0.05),np.arange(1.,2.,0.1),np.arange(2.,10.,0.5),np.arange(10.,50.,1.)))
      obsCLBRstacks={}
      expCLBRstacks={}
      for i,(a,SR,obslim,explim) in enumerate(zip(SR_exps,SR_sdata,SR_olims,SR_elims)):
      #for i,(a,SR,obslim,explim) in enumerate(zip([SR1_0lep],['SR1_s'],[SR_obslim[0]],[SR_explim[0]])):
         print "Generating limits for {0}...".format(a.name)

         # Determine how CL varies with s hypothesis only once per signal region         
         if firstmodel==True: 
            fig1 = plt.figure(figsize=(6,4))
            #plt.subplots_adjust(left=0.08,right=0.97,bottom=0.07,top=0.93,hspace=0.3)
            axCL = fig1.add_subplot(111)
            axCL.axvline(x=obslim,ls='--',color='r')
            axCL.axvline(x=explim,ls='--',color='k')
            #a.getCLvals(svals,regen=regenCL,method='asymptotic')
            #a.plotCL(ax=axCL)
            a.getCLvals(svals,regen=regenCL,method='simulate',N=100000,regenQdist=regenQdist) #'marg') #'asymptotic')
            a.plotCL(ax=axCL)
            axCL.set_xscale('log')
            axCL.set_xlim(1,np.max(svals))
            axCL.set_xticks([1,2,3,6,10,20,30,60,100,200])
            axCL.get_xaxis().set_major_formatter(ScalarFormatter())
            fig1.savefig('{0}/{1}_CL.png'.format(outdir,a.name))
   
         # now the hard bit; need to rescale all the data to simulate varying branching ratios to the final states looked for in each search. For now just get the 95% limit; i.e. vary the data, and interpolate the branching ratio value that gives the 95% CLs value at each point (if possible; if it is already excluded with greater than 95% CLs with a BR of 100% then it will remain excluded with BR<100%).
         # To do this we'll need a stack of pvalue maps, for each of the branching ratios, for each channel. 
         obsBRstack = np.zeros((len(BRS),len(data)))
         expBRstack = np.zeros((len(BRS),len(data)))
         for i,br in enumerate(BRS):
            print "    Computing CL values for BR={0} hypothesis".format(br) 
            # recompute CL values with the new branching ratio folded in
            obsBRstack[i] = a.interpolate_pvalues('obs',br*data[SR])
            expBRstack[i] = a.interpolate_pvalues('b',br*data[SR])
            #print 'min p-val:',np.min(obsBRstack[i])
         obsCLBRstacks[a.name] = obsBRstack 
         expCLBRstacks[a.name] = expBRstack
   
  
      # Compute the upper 95% limit on the total cross section for the channel
      # To do this have to fold in the known stop pair production cross section, and combine the limits from the various signal regions.
      # First do the combination. For each branching ratio slice we need to find, at every model point, which channel has the highest expected sensitivity (lowest expected CLs value)
      combi_maps = np.zeros((len(BRS),len(data)),dtype=int)
      combCLBR_obs = np.zeros((len(BRS),len(data)))
      for i,br in enumerate(BRS):   
         print "    Combining CL values for BR={0} hypothesis".format(br) 
         exppvals_i = np.array([expCLBRstacks[SR.name][i] for SR in SR_exps])
         # for each data point, determine which SR has the lowest expected p-value
         combi_maps[i] = np.argmin(exppvals_i,axis=0)
         # Determine the combined observed CLs values
         combCLBR_obs[i] = np.choose(combi_maps[i],choices=[obsCLBRstacks[SR.name][i] for SR in SR_exps])
      # Create interpolation functions for the observed CLs values for every data point, interpolating through the BR stack
      #CLinterpfuncs = []
      limitBRvals = []
      print "    Finding 95% confidence limits on BR"
      for datapointCL in combCLBR_obs.T:
         # sort data into monotonically increasing order
         sorti = np.argsort(datapointCL)
         # each row should be the various CL values one gets for different br values. Interpolate these
         CLifunc = interp.interp1d(datapointCL[sorti],BRS[sorti],bounds_error=False,fill_value=np.nan)
         # Use the interpolating function to find the br value on the CL=0.05 limit
         limitBRvals += [CLifunc(0.05)]
   
      limitBRvals=np.array(limitBRvals)
      
      # Print out a file containing the limits on the XS*BR
      masses = [data[mass] for mass in masslist] 
      toprint = np.array(masses+[data['XS'],limitBRvals,data['XS']*limitBRvals]).T
      np.savetxt('{0}/{1}/{1}-XSxBR_limits.txt'.format(outdir,modelname),toprint,fmt="%1.4e")

      if XSxBR_limits_only: continue #skip the rest if we don't want the visualisations.

      #----------------------------------------
      # Plot combined limits (take p value at each point from the channel with the best expected sensitivity)

      # need to figure out how to do this with 3 mass dimensions...
      # Do it for every mass3 slice? pretty tedious to look through... 
      # don't have much alternative though.
      # sigh...
   
       
      # # Sort through masses and organise them into groups with equal chargino mass.
      # if len(masslist)>2:
      #    unique_Mchars, sorti = np.unique(data['Mchar'],return_index=True) #get indices of elements with unique chargino massess
      #    groupmasks = {} # grouping masks stored here
      #    for Mchar in unique_Mchars:     
      #       groupmasks[Mchar] = data['Mchar']==Mchar #True for all elements with this chargino mask
      # else:
      #    groupmasks = {'':np.ones(len(data),dtype=bool)}
      #  Begin loop over chargino masses 
      # for Mchar,mask in groupmasks.items():
      
      # Ok the masking thing didn't work well because the data is not organised nicely into slices. Try doing the full 3d interpolation and taking slices of that instead.
      if len(masses)>2:
         gridx, gridy, gridz = np.arange(0,820,20), np.arange(0,820,20), np.array([1,50,100,200,300,400,500,600])
         #pointlist = np.array([(x,y,z) for x in gridx for y in gridy for z in gridz]
         grid_x3D, grid_y3D, grid_z3D = np.meshgrid(gridx,gridy,gridz)
         xipoints = (grid_x3D,grid_y3D,grid_z3D)
   
         grid3D_joined = {}
         for i,(a,SRname) in enumerate(zip(SR_exps,SR_sdata)):
            grid3D_joined[SRname] = interp.griddata( points = np.array(masses).T,
                                                     values = data[SRname],
                                                     xi = (grid_x3D,grid_y3D,grid_z3D)
                                                   )

         # need to pull grid3Ds apart into slices, stored in seperate dictionaries
         s_slices = []
         for zi,z in enumerate(gridz):
            newslice = {}
            for SRname in SR_sdata:
               # cut out a z slice
               newslice[SRname] = grid3D_joined[SRname][:,:,zi].flatten()
            s_slices += [newslice]

         limitBR3D = interp.griddata( points = np.array(masses).T,
                                      values = limitBRvals,
                                      xi = (grid_x3D,grid_y3D,grid_z3D)
                                    ).transpose([2,0,1])
         limitBRslices = [BRslice.flatten() for BRslice in limitBR3D]

         limitXS3D = interp.griddata( points = np.array(masses).T,
                                      values = data['XS']*limitBRvals,
                                      xi = (grid_x3D,grid_y3D,grid_z3D)
                                    ).transpose([2,0,1])     
         limitXSslices = [XSslice.flatten() for XSslice in limitXS3D]
  
         xypairs = np.array([grid_x3D[:,:,0].flatten(),grid_y3D[:,:,0].flatten()]).T
      else:
         # this should recover the normal, uninterpolated behaviour I had for the 2D data
         gridz = ['any']
         s_slices = [data] # we are doing some kind of wild duck typing here
         limitBRslices = [limitBRvals]
         limitXSslices = [data['XS']*limitBRvals]
         xypairs = data[['Mstop','Mneut']].view((np.float32,2)) # xy coords of s values

      #if modeli==0: continue
      # loop through slices of the 3D interpolation
      for zi,(Mchar,sdata,limitBRslice,limitXSslice) in enumerate(zip(gridz,s_slices,limitBRslices,limitXSslices)):
         
         skipslice=False
         gridx, gridy = np.arange(0,800,10), np.arange(0,620,10)
         grid_x, grid_y = np.meshgrid(gridx,gridy)
         chardir = "mchar={0}".format(Mchar)     

         if not os.path.exists("{0}/{1}/{2}/".format(outdir,modelname,chardir)):
            os.makedirs("{0}/{1}/{2}".format(outdir,modelname,chardir)) 

         # Make a figure to show where we have data
         #fig = plt.figure()
         #ax = fig.add_subplot(111)
         #ax.scatter(xypairs[:,0],xypairs[:,1])
         #fig.savefig('{0}/{1}/{2}/dataloc.png'.format(outdir,modelname,chardir))
          
         # Figure for output 
         fig2 = plt.figure(figsize=(5*Nexps,5))
         pvals = {}
         for i,(a,SRname) in enumerate(zip(SR_exps,SR_sdata)):
            ax = fig2.add_subplot(1,Nexps,i+1)
            try:
               pvals[a.name] = a.plot2Dlimit(ax,xypairs,sdata[SRname],(gridx,gridy),colormap='obs')
            except RuntimeError:
               # If there are too few points in some chargino slice this error may occur when trying to do the interpolation to the grid. In this case just skip the slice
               skipslice=True
               break
             #ax.plot(reallim[:,0],reallim[:,1], lw=2, color='orange')
            ax.set_title(a.name)
         if skipslice: continue
         fig2.savefig('{0}/{1}/{2}/limits.png'.format(outdir,modelname,chardir))
      
         # Get expected p values
         exppvals = np.array([pvals[SR.name]['b'] for SR in SR_exps])
         
         # for each point in the grid, determine which SR has the lowest expected p-value
         combi = np.argmin(exppvals,axis=0)
        
         figexp = plt.figure(figsize=(6.7,5))
         ax = figexp.add_subplot(111)
         cols = ['white','red','blue','green','purple','orange','pink','brown','cyan','gray']
         cmap=colors.ListedColormap(cols)
         bounds=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5]
         norm = colors.BoundaryNorm(bounds,cmap.N)
      
         ax.imshow(combi,origin='lower',extent=(min(gridx),max(gridx),min(gridy),max(gridy)),
                     cmap=cmap,norm=norm,interpolation='nearest')
      
         #create a legend
         proxies = []
         labels = []
         for i,SR in enumerate(SR_exps):
            proxies+=[Rectangle((0,0),1,1,fc=cols[i])]
            labels+=[SR.name]
         leg = ax.legend(proxies,labels,loc=2)
         plt.setp(leg.get_texts(), fontsize='10')
         leg.get_frame().set_alpha(0.5)
         figexp.savefig("{0}/{1}/{2}/limfrom.pdf".format(outdir,modelname,chardir),dpi=(800/8),bbox_inches='tight')
      
         print combi
         print combi.shape
         
         # Create arrays for each observation using these indices
         comb = {}
         for obs in ['b-1sig', 'b', 'obs', 'b+1sig']:
            print [pvals[SR.name][obs].shape for SR in SR_exps]
            comb[obs] = np.choose(combi,choices=[pvals[SR.name][obs] for SR in SR_exps])
        
         #Plot combined limits (take max p value for each signal region point)
         
         fig2com = plt.figure(figsize=(6.7,5))
         ax = fig2com.add_subplot(111)
         
         fig3 = plt.figure()
         axtemp = fig3.add_subplot(111)
         
    
         #background colors from combined expected p-values 
         im = ax.imshow(comb['b'], interpolation='bilinear', origin='lower',
                            extent=(min(gridx),max(gridx),min(gridy),max(gridy)), aspect='auto',alpha=0.5)
        
         CS1 = ax.contour(grid_x, grid_y, comb['b-1sig'], levels=[0.05],linewidths=[2],linestyles=['dashed'],colors=['b'])
         CS2 = ax.contour(grid_x, grid_y, comb['b'], levels=[0.05],linewidths=[2],linestyles=['dashed'],colors=['k'])
         CS3 = ax.contour(grid_x, grid_y, comb['b+1sig'], levels=[0.05],linewidths=[2],linestyles=['dashed'],colors=['g'])
         CS4 = ax.contour(grid_x, grid_y, comb['obs'], levels=[0.05],linewidths=[2],linestyles=['solid'],colors=['r'])
        
         """
         #extract data from contours so we can filled between them
         def getpoints(CS):
            try: 
               points = CS.collections[0].get_paths()[0]
               return points.vertices
            except IndexError:
               return np.array([gridx,np.zeros(len(gridx))]).T
         
         v1 = getpoints(CS1)      #get points from contour path
         v2 = getpoints(CS2)
         v3 = getpoints(CS3)
         v4 = getpoints(CS4)
         x = gridx
         c1 = np.interp(x,v1[:,0],v1[:,1])
         c2 = np.interp(x,v2[:,0],v2[:,1])
         c3 = np.interp(x,v3[:,0],v3[:,1])
         c4 = np.interp(x,v4[:,0],v4[:,1])
         l2 = ax.plot(x,c2,lw=2,color='black',linestyle='dashed',zorder=0.1)
         ax.fill_between(x,c1,c3,color=(0,1,1),zorder=0)
         l4 = ax.plot(x,c4,lw=2,color='red',linestyle='solid',zorder=0.1)
         r2 = Rectangle((0, 0), 1, 1,color=(0,1,1)) # creates rectangle patch for legend use.
         #ax.plot(x,c2)
         """
         #x.set_xlim(50,1300)
         #ax.set_ylim(700,1200)
         
         #yminorLocator   = MultipleLocator(20)
         #xminorLocator   = MultipleLocator(50)
         #ax.yaxis.set_minor_locator(yminorLocator)
         #ax.xaxis.set_minor_locator(xminorLocator)
         
         #increase tick and border thickness
         minorticks = ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True)
         majorticks = ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines()
         for line in minorticks:
             line.set_markeredgewidth(2)
             line.set_markersize(3)
         for line in majorticks:
             line.set_markeredgewidth(2)
             line.set_markersize(5)
         [i.set_linewidth(1.5) for i in ax.spines.itervalues()]
         
         
         font = {'family'     : 'serif',
                 'color'      : 'k',
                 'weight' : 'normal',
                 'size'   : 20,
                 }
                 
         #ax.set_xlabel('$m_{\~{\chi}_1^0}$ [GeV]', font)
         #ax.set_ylabel('$m_{\~{g}}$ [GeV]', font)
         
         #leg=ax.legend([l1[0],r1,l2[0],r2],['ATLAS','ATLAS $\pm 1\sigma$','Delphes','Delphes $\pm 1\sigma$'],\
         #    loc='lower left')#,title='$95\%$ C.L.')
         #leg.draw_frame(False)
         
         #ax.text(990,800,'$\~{g}$ NLSP', fontdict=font)
         
         #plt.tight_layout(h_pad=2)
       
         # record the state of the plot so we can remove the annotations we are about to make
         keep_lst = ax.get_children()[:]
      
         # draw the upper 95% CLs limits on the branching ratios onto the plots
         for (x,y),BRlim in zip(xypairs,limitBRslice):
            if np.isfinite(BRlim):
               ax.text(x,y,"%.2f" % BRlim,fontsize=4,horizontalalignment='center',verticalalignment='center')
      
         print outdir,modelname,chardir 
         fig2com.savefig("{0}/{1}/{2}/comb_wBRlims.pdf".format(outdir,modelname,chardir),dpi=(800/8),bbox_inches='tight')
      
         # Now remove the anotations and save the figure again with different ones
         for a in ax.get_children()[:]:
            if a not in keep_lst:
               a.remove()
      
         # annotate with upper 95% CLs limits on the full cross section (i.e. multiply in the production cross section we assumed to get the branching ratios)
         for (x,y),BRlim,XSlim in zip(xypairs,limitBRslice,limitXSslice):
            if np.isfinite(BRlim):
               ax.text(x,y,"%.2f" % (XSlim/1000.),fontsize=4,horizontalalignment='center',verticalalignment='center')
       
         fig2com.savefig("{0}/{1}/{2}/comb_wXSlims.pdf".format(outdir,modelname,chardir),dpi=(800/8),bbox_inches='tight')
         




