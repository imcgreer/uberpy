#!/usr/bin/env python

import os
from collections import defaultdict
import numpy as np
from scipy.stats import scoreatpercentile
from astropy.stats import sigma_clip
from astropy.time import Time,TimeDelta
import fitsio

import matplotlib.pyplot as plt
from matplotlib import ticker

#from .ubercal import CalibrationObject,CalibrationObjectSet,ubercal_solve
from ubercal import CalibrationObject,CalibrationObjectSet,ubercal_solve,init_flatfields

nX,nY = 4096,4032
nX2 = nX//2
nY2 = nY//2
nCCD = 4

# divide nights into contiguous observing blocks
bok_runs = [
  ('20131222',),
  ('20140114', '20140115', '20140116', '20140117', '20140118', 
   '20140119',),
  ('20140120', '20140121', '20140123', '20140124', '20140126', 
   '20140127', '20140128', '20140129',),
  ('20140213', '20140214', '20140215', '20140216', '20140217', 
   '20140218', '20140219',),
  ('20140312', '20140313', '20140314', '20140315', '20140316', 
   '20140317', '20140318', '20140319',),
  ('20140413', '20140414', '20140415', '20140416', '20140417', 
   '20140418',),
  ('20140424', '20140425', '20140426', '20140427', '20140428',),
  ('20140512', '20140513', '20140514', '20140515', '20140516', 
   '20140517', '20140518',),
  ('20140609', '20140610', '20140611', '20140612', '20140613',),
  ('20140629', '20140630', '20140701', '20140702', '20140703',),
#   '20140705',), # no observations, doesn't have a log
  ('20140710', '20140711', '20140713', 
#   '20140716', # no observations, doesn't have a log
   '20140717', '20140718',),
]

# exclude clearly non-photometric nights
bad_night_list = ['20131222',]

def bok_run_index():
	return {utd:run for run,utds in enumerate(bok_runs) for utd in utds}

def get_mjd(utDate,utTime):
	utStr = '-'.join([utDate[:4],utDate[4:6],utDate[6:]]) + ' ' + utTime
	return Time(utStr,scale='utc').mjd

def select_photometric_images(filt):
	'''Add initial cut on photometricity, based on seeing values and zeropoints
	   calculated from the SExtractor catalogs.'''
	zpmin = {'g':25.4}
	iqdir = os.path.join(os.environ['BOK90PRIMEDIR'],'py')
	iqlog = np.loadtxt(os.path.join(iqdir,'bokimagequality_%s.log'%filt),
	                   dtype=[('utdate','S8'),('frame','i4'),
	                          ('sky','f4'),('seeing','f4'),('zeropoint','f4')])
	isphoto = ( (iqlog['seeing'] < 2.5) &
	            (iqlog['zeropoint'] > zpmin[filt]) )
	print 'iter1: rejected %d frames out of %d' % ((~isphoto).sum(),len(iqlog))
	# require neighboring images to be photometric as well, to handle varying
	# conditions
	min_nphoto = 5
	nphoto = np.zeros(len(iqlog),dtype=np.int)
	for i in np.where(isphoto)[0]:
		up,down = True,True
		for j in range(1,min_nphoto+1):
			if up:
				if i+j<len(iqlog) and isphoto[i+j]:
					nphoto[i] += 1
				else:
					up = False
			if down:
				if i-j>0 and isphoto[i-j]:
					nphoto[i] += 1
				else:
					down = False
	isphoto &= nphoto > min_nphoto
	print 'iter2: rejected %d frames out of %d' % ((~isphoto).sum(),len(iqlog))
	return iqlog,isphoto

def build_frame_list(filt,nightlyLogs=None):
	'''Collapse the nightly observing logs into a master frame list containing
	   the relevant info for each observation, namely:
	     mjd,expTime,airmass   observing parameters
         nightIndex            0-indexed from the list of observing nights
	     nightFrameNum         frame number from observing logs
	'''
	import boklog
	refTimeUT = '07:00:00.0' # 7h UT = midnight MST
	if nightlyLogs is None:
		nightlyLogs = boklog.load_Bok_logs()
	iqlog,isphoto = select_photometric_images(filt)
	frameList = []
	for night,utd in enumerate(sorted(nightlyLogs.keys())):
		frames = nightlyLogs[utd]
		ii = np.where((frames['filter']==filt) &
		              (frames['imType']=='object'))[0]
		if len(ii)==0:
			continue
		mjds = np.array([get_mjd(utd,frames['utStart'][i]) for i in ii])
		epochIndex = np.repeat(night,len(ii))
		refTime = get_mjd(utd,refTimeUT)
		dt = 24*(mjds-refTime)
		jj = np.array([np.where((iqlog['utdate']==utd) &
		                        (iqlog['frame']==i))[0][0]
		                 for i in ii])
		frameList.append((mjds,dt,
		                  frames['expTime'][ii],frames['airmass'][ii],
		                  epochIndex,ii,isphoto[jj]))
	frameList = np.hstack(frameList)
	frameList = np.core.records.fromarrays(frameList,
	                     dtype=[('mjd','f8'),('dt','f4'),
	                            ('expTime','f4'),('airmass','f4'),
	                            ('nightIndex','i4'),('nightFrameNum','i4'),
	                            ('isPhoto','i2')])
	return frameList

def collect_observations(filt,catpfx='sdssbright'):
	import boklog
	import bokcat
	photdir = os.path.join(os.environ['BOK90PRIMEOUTDIR'],'catalogs_v2')
	aperNum = -1
	mag0 = 25.0
	SNRcut = 20.0
	utd2run = bok_run_index()
	nightlyLogs = boklog.load_Bok_logs()
	frameList = build_frame_list(filt,nightlyLogs)
	objectList = defaultdict(list)
	refcat = bokcat.load_targets('SDSSstars')
	for night,utd in enumerate(sorted(nightlyLogs.keys())):
		try:
			catfn = '.'.join([catpfx,utd,filt,'cat','fits'])
			fits = fitsio.FITS(os.path.join(photdir,catfn))
		except ValueError:
			continue
		print catfn
		runarr = np.repeat(utd2run[utd],100)# to avoid creating repeatedly
		night_jj = np.where(frameList['nightIndex']==night)[0]
		#
		data = fits[1].read()
		for starNum,i1,i2 in fits[2]['TINDEX','i1','i2'][:]:
			good = ( (data['flags'][i1:i2,aperNum] == 0) &
			         (data['aperCounts'][i1:i2,aperNum] > 0) & 
			         (data['aperCountsErr'][i1:i2,aperNum] <
			           (1/SNRcut)*data['aperCounts'][i1:i2,aperNum]) )
			if good.sum() == 0:
				continue
			good = i1 + np.where(good)[0]
			# XXX to catch a bug in the catalog - some objects appear multiple
			#     times in the same frame!
			frameNums,jj = np.unique(data['frameNum'][good],return_index=True)
			if len(frameNums) != len(good):
				print 'WARNING: object with multiple instances ',
				print starNum,data['frameNum'][good]
				good = good[jj] # restrict to unique frames
			jj = np.where(np.in1d(frameList['nightFrameNum'][night_jj],
			                      data['frameNum'][good]))[0]
			if len(jj) != len(good):
				print good,jj
				print frameList['nightFrameNum'][night_jj]
				print data['frameNum'][good]
				raise ValueError
			jj = night_jj[jj]
			expTime = frameList['expTime'][jj]
			counts = data['aperCounts'][good,aperNum]
			mags = mag0 - 2.5*np.log10(counts/expTime)
			errs = (2.5/np.log(10))*data['aperCountsErr'][good,aperNum]/counts
			ccdNums = data['ccdNum'][good] 
			ampNums = (data['x'][good]//nX2).astype(np.int) + \
			          2*(data['y'][good]//nY2).astype(np.int)
			nightIndex = frameList['nightIndex'][jj]
			runIndex = runarr[:len(good)]
			refMag = np.repeat(refcat[filt][starNum],len(good))
			objectList[starNum].append((mags,errs,ccdNums,ampNums,
			                            data['x'][good],data['y'][good],
			                            runIndex,nightIndex,jj,refMag))
	for starNum in objectList:
		arr = np.hstack(objectList[starNum])
		objectList[starNum] = np.core.records.fromarrays(arr,
		                        dtype=[('magADU','f4'),('errADU','f4'),
		                               ('ccdNum','i4'),('ampNum','i4'),
		                               ('x','f4'),('y','f4'),
		                               ('runIndex','i4'),('nightIndex','i4'),
	                                   ('frameIndex','i4'),('refMag','f4')])
	return frameList,objectList

def cache_bok_data(frameList,objectList,fileName):
	fits = fitsio.FITS(fileName,'rw')
	indx = np.empty(len(objectList),
	                dtype=[('starNum','i4'),('i1','i4'),('i2','i4')])
	i1 = 0
	for i,starNum in enumerate(objectList):
		if i==0:
			fits.write(objectList[starNum])
		else:
			fits[-1].append(objectList[starNum])
		indx['starNum'][i] = starNum
		indx['i1'][i] = i1
		indx['i2'][i] = i1 + len(objectList[starNum])
		i1 += len(objectList[starNum])
	fits.write(indx)
	fits.write(frameList)
	fits.close()

def load_cached_bok_data(fileName):
	fits = fitsio.FITS(fileName)
	data = fits[1].read()
	indexes = fits[2].read()
	frameList = fits[3].read()
	objectList = {}
	for starNum,i1,i2 in indexes:
		objectList[starNum] = data[i1:i2]
	return frameList,objectList

class SimFlatField(object):
	def __init__(self,n=1,kind='gradient',dm=0.3):
		#self.coeffs = np.random.rand(n,nCCD,2)
		self.coeffs = np.array([[[0.0,0.0],
		                         [0.0,1.0],
		                         [1.0,0.0],
		                         [1.0,1.0]]])
		self.dims = (nY,nX)
		self.dm = dm
		if kind=='gradient':
			self.flatfun = self._gradientfun
	def _gradientfun(self,coeff,x,y):
		norm = coeff.sum(axis=-1)
		if np.isscalar(norm):
			if norm>0:
				norm **= -1
		else:
			norm[norm>0] **= -1
		return ( (coeff[...,0]*(x/float(nX)) + 
		          coeff[...,1]*(y/float(nY)))
		         * self.dm * norm )
	def __call__(self,indices,x,y):
		coeff = self.coeffs[indices]
		return self.flatfun(coeff,x,y)
	def make_image(self,indices):
		Y,X = np.indices(self.dims)
		return self.__call__(indices,X,Y)

def sim_init(a_init,k_init,objs,**kwargs):
	a_range = kwargs.get('sim_a_range',0.3)
	k_range = kwargs.get('sim_k_range',0.2)
	fixed_mag = kwargs.get('sim_fixed_mag',18.)
	fixed_err = kwargs.get('sim_fixed_err',0.03)
	print 'SIMULATION: a_range=%.2f  k_range=%.2f' % (a_range,k_range)
	np.random.seed(1)
	simdat = {}
	simdat['a_true'] = a_range*np.random.random_sample(a_init.shape)
	simdat['a_true'] -= np.median(simdat['a_true'])
	simdat['k_true'] = k_range*np.random.random_sample(k_init.shape)
	simdat['errMin'] = kwargs.get('errMin',0.01)
	print 'SIMULATION: minimum rms %.3f' % simdat['errMin']
	if kwargs.get('sim_userealmags',True):
		simdat['mag'] = np.array([objs[i]['refMag'][0] for i in objs])
		print 'SIMULATION: using real magnitudes'
	else:
		simdat['mag'] = np.repeat(fixed_mag,len(objs))
		print 'SIMULATION: using fixed magnitude %.2f' % fixed_mag
	if kwargs.get('sim_userealerrs',True):
		simdat['err'] = np.array([np.median(objs[i]['errADU']) for i in objs])
		print 'SIMULATION: using real errors'
	else:
		simdat['err'] = np.repeat(fixed_err,len(objs))
		print 'SIMULATION: using fixed errors %.2f' % fixed_err
	simdat['outlier_frac'] = kwargs.get('sim_outlierfrac',0.1)
	print 'SIMULATION: fraction of outliers %g' % simdat['outlier_frac']
	simdat['is_outlier'] = []
	if kwargs.get('sim_addflatfield',True):
		dm = kwargs.get('sim_flatfield_range',0.3)
		simdat['flatfield'] = SimFlatField(n=1,kind='gradient',dm=dm)
		print 'SIMULATION: applying %s flat field' % 'gradient'
		print 'SIMULATION:   maximum range dm=%.2f' % dm
	else:
		simdat['flatfield'] = lambda *args: 0
		print 'SIMULATION: no flat field variation'
	return simdat

def sim_initobject(i,obj,frames,simdat,rmcal):
	x = frames['airmass'][obj['frameIndex']]
	dt = frames['dt'][obj['frameIndex']]
	dk_dt = rmcal.get_terms('dkdt',0) # using a fixed value
	flatfield = simdat['flatfield']
	flatIndex = (np.repeat(0,len(obj)),obj['ccdNum']-1)
	mags = simdat['mag'][i] - (
	         simdat['a_true'][obj['nightIndex'],obj['ccdNum']-1] 
	          - (simdat['k_true'][obj['nightIndex']] + dk_dt*dt)*x
	           + flatfield(flatIndex,obj['x'],obj['y']) )
	errs = np.repeat(simdat['err'][i],len(mags))
	mags[:] += errs*np.random.normal(size=mags.shape)
	if simdat['outlier_frac'] > 0:
		is_outlier = np.random.poisson(simdat['outlier_frac'],len(mags))
		ii = np.where(is_outlier)[0]
		# start at 5sigma and decline as a power law with index 1.5
		nsig_outlier = (np.random.pareto(1.5,len(ii)) + 1) * 5.0
		sgn = np.choose(np.random.rand(len(ii)) > 0.5,[-1,1])
		mags[ii] += sgn*nsig_outlier*errs[ii]
		simdat['is_outlier'].append(is_outlier)
	return CalibrationObject(mags,errs,errMin=simdat['errMin'])

def sim_finish(rmcal,simdat):
	gk = np.where(~rmcal.params['k']['terms'].mask)
	dk = (rmcal.params['k']['terms']-simdat['k_true'])[gk].flatten()
	ga = np.where(~rmcal.params['a']['terms'].mask)
	da = (rmcal.params['a']['terms']-simdat['a_true'])[ga].flatten()
	median_a_offset = np.median(da)
	#
	plt.figure(figsize=(9,4))
	plt.subplots_adjust(0.08,0.06,0.98,0.98,0.23,0.15)
	ax1 = plt.subplot2grid((2,4),(0,0),colspan=3)
	plt.axhline(0,c='gray')
	plt.plot(dk)
	plt.xlim(0,len(gk[0]))
	plt.ylim(-0.5,0.5)
	plt.ylabel(r'$\Delta(k)$',size=12)
	#
	ax2 = plt.subplot2grid((2,4),(1,0),colspan=3)
	plt.axhline(0,c='gray')
	plt.plot(da-median_a_offset)
	plt.xlim(0,len(ga[0]))
	plt.ylim(-0.8,0.8)
	plt.ylabel(r'$\Delta(a)$',size=12)
	#
	dm = []
	for i,obj in enumerate(rmcal):
		mag,err = rmcal.get_object_phot(obj)
		dm.append(obj.refMag - (mag - median_a_offset))
		try:
			is_outlier = simdat['is_outlier'][i].astype(np.bool)
			is_masked = obj.mags.mask
#		print '%4d %4d %4d %4d %4d' % (len(mag),np.sum(is_outlier),np.sum(is_outlier&is_masked),np.sum(is_outlier&~is_masked),np.sum(~is_outlier&is_masked))
		except:
			pass
	dm = np.ma.concatenate(dm)
	dm3 = sigma_clip(dm,sig=3,iters=1)
	frac_sig3 = np.sum(dm3.mask & ~dm.mask) / float(np.sum(~dm.mask))
	mm = 1000 # millimag
	print
	print '<da> = %.1f' % (mm*np.median(da))
	print '<dk> = %.1f' % (mm*np.median(dk)),
	print '    @AM=2.0  %.1f' % (mm*np.median(dk)*2)
	print
	print '%8s %8s %8s %8s %8s   [millimag]' % \
	        ('<dm>','sig','sig3','%(3sig)','sig0')
	print '%8s %8s %8s %8s %8s' % tuple(['-'*6]*5)
	print '%8.2f %8.2f %8.2f %8.2f %8.2f' % \
	       (mm*dm.mean(),mm*dm.std(),mm*dm3.std(),100*frac_sig3,0.0)
	#
	ax3 = plt.subplot2grid((2,4),(0,3),rowspan=2)
	plt.hist(dm.data,50,(-0.2,0.2),edgecolor='none',color='r',normed=True)
	plt.hist(dm.data[~dm.mask],50,(-0.2,0.2),edgecolor='none',
	         color='b',normed=True)
	ax3.text(0.95,0.98,r'$\Delta(mag)$',size=12,ha='right',va='top',
	         transform=ax3.transAxes)
	ax3.axvline(dm.mean(),c='purple')
	ax3.axvline(dm.mean()-dm3.std(),c='purple',ls='--')
	ax3.axvline(dm.mean()+dm3.std(),c='purple',ls='--')
	plt.xlim(-0.2,0.2)
	ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
	for ax in [ax1,ax2,ax3]:
		for tick in ax.xaxis.get_major_ticks()+ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(9)

def cal_finish(rmcal):
	dm = []
	for i,obj in enumerate(rmcal):
		mag,err = rmcal.get_object_phot(obj)
		dm.append(obj.refMag - mag)
	dm = np.ma.concatenate(dm)
	dm3 = sigma_clip(dm,sig=3,iters=1)
	median_dm = np.ma.median(dm3)
	dm -= median_dm
	dm3 -= median_dm
	frac_sig3 = np.sum(dm3.mask & ~dm.mask) / float(np.sum(~dm.mask))
	mm = 1000 # millimag
	print
	print '%8s %8s %8s %8s %8s %8s  [millimag]' % \
	        ('<dm>','sig','sig3','%(3sig)','sig0')
	print '%8s %8s %8s %8s %8s' % tuple(['-'*6]*5)
	print '%8.2f %8.2f %8.2f %8.2f %8.2f' % \
	       (mm*dm.mean(),mm*dm.std(),mm*dm3.std(),100*frac_sig3,0.0)
	#
	plt.figure()
	plt.hist(dm,50,(-1,1))

#def reject_outliers(rmcal,**kwargs):
def reject_outliers(rmcal,simdat,**kwargs):
	sig = kwargs.get('reject_sig',3.0)
	iters = kwargs.get('reject_niter',2)
	for i,obj in enumerate(rmcal):
		mags,errs = rmcal.get_object_phot(obj)
		clipped = sigma_clip(mags,sig=sig,iters=iters)
# need a verbose argument
#		if clipped.mask.sum() > mags.mask.sum():
#			print 'object %d rejected %d' % (i,(clipped.mask&~mags.mask).sum())
		obj.update_mask(clipped.mask)

def fiducial_model(frames,objs,verbose=True,dosim=False,niter=1,**kwargs):
	ndownsample = kwargs.get('downsample',1)
	doflats = kwargs.get('doflats',True)
	numCCDs = 4
	numFrames = len(frames)
	# identify nights to process
	bok_nights = np.array([utd for run in bok_runs for utd in run])
	numNights = len(bok_nights)
	framesPerNight = np.array([np.sum(frames['nightIndex']==i) 
	                              for i in range(numNights)])
	bad_nights = np.where( np.in1d(bok_nights,bad_night_list) | 
	                       (framesPerNight==0) )[0]
	# initialize the a-term array to zeros, masking non-photometric nights
	a_init = np.ma.array(np.zeros((numNights,numCCDs)),mask=False)
	a_init[bad_nights] = np.ma.masked
	# initialize the k-term array to zeros, masking non-photometric nights
	k_init = np.ma.array(np.zeros(numNights),mask=False)
	k_init[bad_nights] = np.ma.masked
	# initialize the flat field arrays 
	if doflats:
		flatfield_init = init_flatfields((numCCDs,),nX,nY,method='spline')
	else:
		flatfield_init = init_flatfields((numCCDs,),nX,nY,method='null')
	# construct the container for the global ubercal parameters
	rmcal = CalibrationObjectSet(a_init,k_init,frames['dt'],
	                             frames['airmass'],flatfield_init)
	# currently using a fixed value for the time derivate of k, taken from P08
	rmcal.set_fixed_dkdt(0)
	#rmcal.set_fixed_dkdt(-0.7e-2/10) # given as mag/airmass/10h
	#
	if dosim:
		simdat = sim_init(a_init,k_init,objs,**kwargs)
	# loop over individual stars and set their particulars for each 
	# observation, then add them to the calibration set
	for i,(starNum,obj) in enumerate(objs.items()):
		if (starNum % ndownsample) != 0:
			continue
		if dosim:
			calobj = sim_initobject(i,obj,frames,simdat,rmcal)
			calobj.set_reference_mag(simdat['mag'][i])
		else:
			# construct a calibration object from the flux/err vectors
			calobj = CalibrationObject(obj['magADU'],obj['errADU'])
			# mask the pre-assigned non-photometric observations
			# XXX before doing this, reasonable a and k values must be set
			#calobj.update_mask(frames['isPhoto'][obj['frameIndex']]==0)
			# set the catalog magnitude for this object
			calobj.set_reference_mag(obj['refMag'][0])
		calobj.set_xy(obj['x'],obj['y'])
		# XXX should require all of these to be tuples of arrays for consistency
		calobj.set_a_indices((obj['nightIndex'],obj['ccdNum']-1))
		calobj.set_k_indices(obj['nightIndex'])
		calobj.set_flat_indices((obj['ccdNum']-1,))
		calobj.set_x_indices(obj['frameIndex'])
		calobj.set_t_indices(obj['frameIndex'])
		rmcal.add_object(calobj)
	if verbose:
		print 'number nights: ',np.sum(framesPerNight>0)
		print 'number good nights: ', \
		          np.sum(np.any(~rmcal.params['a']['terms'].mask,axis=1))
		print 'number frames: ',numFrames
		print 'number objects: ',rmcal.num_objects()
		print 'number observations: ',rmcal.num_observations()
		print 'number parameters: ',rmcal.num_params()
	# iteratively solve for the calibration parameters
	for iternum in range(niter):
		pars = ubercal_solve(rmcal,**kwargs)
		rmcal.update_params(pars)
		if doflats:
			rmcal.update_flatfields()
		if dosim:
			sim_finish(rmcal,simdat)
		if iternum < niter-1:
			#reject_outliers(rmcal,**kwargs)
			reject_outliers(rmcal,simdat,**kwargs) # XXX
	if dosim:
		return rmcal,simdat
	return rmcal


def sim_make_residual_images(rmcal,binX=32,binY=32):
	xBins = np.arange(0,nX+1,binX)
	yBins = np.arange(0,nY+1,binY)
	median_a_offset = 0
	dmag = []
	for i,obj in enumerate(rmcal):
		mag,err = rmcal.get_object_phot(obj)
		dmag.append(obj.refMag - (mag - median_a_offset))
	dmag = np.concatenate(dmag)
	xy = np.hstack( [ [rmcal.objs[i].xpos,rmcal.objs[i].ypos] 
	                             for i in range(rmcal.num_objects()) ] )
	# XXX hack that last index in a_indices is ccdNum
	ccds = np.concatenate( [ rmcal.objs[i].a_indices[-1]
	                             for i in range(rmcal.num_objects()) ] )
	ffmaps = []
	for ccdNum in range(4):
		ffmap = [[[] for xi in xBins] for yi in yBins]
		ii = np.where(ccds==ccdNum)[0]
		for xi,yi,dm in zip(np.digitize(xy[0,ii],xBins),
		                    np.digitize(xy[1,ii],yBins),
		                    dmag[ii]):
			ffmap[yi][xi].append(dm)
		for xi in range(len(xBins)):
			for yi in range(len(yBins)):
				if len(ffmap[yi][xi])==0:
					ffmap[yi][xi] = np.nan
				else:
					ffmap[yi][xi] = np.median(ffmap[yi][xi])
		ffmaps.append(np.array(ffmap))
	return np.array(ffmaps)

def _init_fov_fig():
	cmap = plt.get_cmap('jet')
	cmap.set_bad('gray',1.)
	plt.figure(figsize=(10,9))
	plt.subplots_adjust(0.04,0.04,0.99,0.99,0.1,0.1)

def sim_show_residual_images(rmcal,**kwargs):
	_init_fov_fig()
	ffmaps = sim_make_residual_images(rmcal,**kwargs)
	for ccdNum in range(1,5):
		ffim = np.ma.array(ffmaps[ccdNum-1],mask=np.isnan(ffmaps[ccdNum-1]))
		v1 = scoreatpercentile(ffim[~ffim.mask],10)
		v2 = scoreatpercentile(ffim[~ffim.mask],90)
		print ccdNum,ffim.mean(),ffim.std(),v1,v2
		plt.subplot(2,2,ccdNum)
		plt.imshow(ffim,vmin=v1,vmax=v2,
		           origin='lower',extent=[0,nX,0,nY],interpolation='nearest')
		plt.colorbar()

def sim_show_fake_flatfields(simdat):
	_init_fov_fig()
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.imshow(simdat['flatfield'].make_image((0,i,)),
		           origin='lower',extent=[0,nX,0,nY])
		plt.colorbar()

def show_fit_flatfields(rmcal):
	_init_fov_fig()
	for i,ff in enumerate(rmcal.flatfields):
		plt.subplot(2,2,i+1)
		plt.imshow(ff.make_image(res=64),
		           origin='lower',extent=[0,nX,0,nY])
		plt.colorbar()

