#!/usr/bin/env python

import os
from collections import defaultdict
import numpy as np
from astropy.stats import sigma_clip
from astropy.time import Time,TimeDelta
import fitsio

import matplotlib.pyplot as plt

#from .ubercal import CalibrationObject,CalibrationObjectSet,ubercal_solve
from ubercal import CalibrationObject,CalibrationObjectSet,ubercal_solve

nX,nY = 4096,4032
nX2 = nX//2
nY2 = nY//2

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
	frameList = []
	for night,utd in enumerate(sorted(nightlyLogs.keys())):
		frames = nightlyLogs[utd]
		ii = np.where((frames['filter']==filt) &
		              (frames['imType']=='object'))[0]
		mjds = np.array([get_mjd(utd,frames['utStart'][i]) for i in ii])
		epochIndex = np.repeat(night,len(ii))
		refTime = get_mjd(utd,refTimeUT)
		dt = 24*(mjds-refTime)
		frameList.append((mjds,dt,
		                  frames['expTime'][ii],frames['airmass'][ii],
		                  epochIndex,ii))
	frameList = np.hstack(frameList)
	frameList = np.core.records.fromarrays(frameList,
	                     dtype=[('mjd','f8'),('dt','f4'),
	                            ('expTime','f4'),('airmass','f4'),
	                            ('nightIndex','i4'),('nightFrameNum','i4')])
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
			errs = 1.0856*data['aperCountsErr'][good,aperNum]/counts
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

def sim_init(a_init,k_init,objs,**kwargs):
	a_range = kwargs.get('sim_a_range',0.3)
	k_range = kwargs.get('sim_k_range',0.2)
	fixed_mag = kwargs.get('sim_fixed_mag',18.)
	fixed_err = kwargs.get('sim_fixed_err',0.03)
	np.random.seed(1)
	simdat = {}
	#simdat['a_true'] = a_range*(0.5-np.random.random_sample(a_init.shape))
	simdat['a_true'] = a_range*np.random.random_sample(a_init.shape)
	simdat['k_true'] = k_range*np.random.random_sample(k_init.shape)
	if kwargs.get('sim_userefmag',False):
		simdat['mag'] = np.array([objs[i]['refMag'][0] for i in objs])
	else:
		simdat['mag'] = np.repeat(fixed_mag,len(objs))
	if kwargs.get('sim_userealerrors',False):
		simdat['err'] = np.array([np.median(objs[i]['errADU']) for i in objs])
	else:
		simdat['err'] = np.repeat(fixed_err,len(objs))
	return simdat

def sim_initobject(i,obj,frames,simdat,rmcal):
	x = frames['airmass'][obj['frameIndex']]
	dt = frames['dt'][obj['frameIndex']]
	dk_dt = rmcal.get_terms('dkdt',0) # using a fixed value
	flatfield = 0
	mags = simdat['mag'][i] - (
	         simdat['a_true'][obj['nightIndex'],obj['ccdNum']-1] 
	          - (simdat['k_true'][obj['nightIndex']] + dk_dt*dt)*x
	           + flatfield )
	errs = np.repeat(simdat['err'][i],len(mags))
	mags[:] += errs*np.random.normal(size=mags.shape)
	return CalibrationObject(mags,errs)

def sim_finish(rmcal,simdat):
	plt.figure(figsize=(12,6))
	plt.subplots_adjust(0.03,0.03,0.99,0.99)
	plt.subplot2grid((2,4),(0,0),colspan=3)
	g = np.where(~rmcal.params['k']['terms'].mask)
	dk = (rmcal.params['k']['terms']-simdat['k_true'])[g].flatten()
	plt.axhline(0,c='gray')
	plt.plot(dk)
	#plt.scatter(simdat['k_true'][g].flatten(),
	#            (rmcal.params['k']['terms']-simdat['k_true'])[g].flatten())
	#plt.plot([0,0.2],[0,0.2],c='g')
	plt.xlim(0,len(g[0]))
	plt.ylim(-0.5,0.5)
	#
	plt.subplot2grid((2,4),(1,0),colspan=3)
	g = np.where(~rmcal.params['a']['terms'].mask)
	da = (rmcal.params['a']['terms']-simdat['a_true'])[g].flatten()
	plt.axhline(0,c='gray')
	plt.plot(da)
	#plt.scatter(simdat['a_true'][g].flatten(),
	#            (rmcal.params['a']['terms']-simdat['a_true'])[g].flatten())
	#plt.plot([-0.15,0.15],[-0.15,0.15],c='g')
	plt.xlim(0,len(g[0]))
	plt.ylim(-0.8,0.8)
	#
	dm = []
	for i,obj in enumerate(rmcal):
		mag,err = rmcal.get_object_phot(obj)
		dm.append(obj.refMag - mag)
	dm = np.ma.concatenate(dm)
	dm3 = sigma_clip(dm,sig=3,iters=1)
	frac_sig3 = np.sum(dm3.mask & ~dm.mask) / float(np.sum(~dm.mask))
	mm = 1000 # millimag
	print
	print '<da> = %.1f' % (mm*np.median(da))
	print '<dk> = %.1f' % (mm*np.median(dk)),
	print '    @AM=2.0  %.1f' % (mm*np.median(dk)*2)
	print
	print '%.2f %.2f %.2f %.2f %.2f' % \
	       (mm*dm.mean(),mm*dm.std(),mm*dm3.std(),100*frac_sig3,0.0)
	#
	plt.subplot2grid((2,4),(0,3),rowspan=2)
	plt.hist(dm,50)
	#plt.xlim(0,len(dm))
	#plt.ylim(-0.8,0.8)

def reject_outliers(rmcal):
	for i,obj in enumerate(rmcal):
		mags,errs = rmcal.get_object_phot(obj)
		clipped = sigma_clip(mags)
		if clipped.mask.sum() > mags.mask.sum():
			print 'object %d rejected %d' % (i,(clipped.mask&~mags.mask).sum())
		obj.update_mask(clipped.mask)

def fiducial_model(frames,objs,verbose=True,dosim=False,niter=1,**kwargs):
	ndownsample = kwargs.get('downsample',1)
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
	# initialize the flat field arrays to zeros, one per CCD
	flatfield_init = np.zeros((numCCDs,nY,nX))
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
			calobj = CalibrationObject(obj['magADU'],obj['errADU'])
			calobj.set_reference_mag(obj['refMag'][0])
		calobj.set_xy(obj['x'],obj['y'])
		calobj.set_a_indices((obj['nightIndex'],obj['ccdNum']-1))
		calobj.set_k_indices(obj['nightIndex'])
		calobj.set_flat_indices(obj['ccdNum']-1)
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
		if dosim:
			sim_finish(rmcal,simdat)
		if iternum < niter-1:
			reject_outliers(rmcal)
	if dosim:
		return rmcal,simdat
	return rmcal

# rv = bokrmcal.fiducial_model(frames,objs,dosim=True,downsample=10,sim_userefmag=True,sim_userealerrors=True)

