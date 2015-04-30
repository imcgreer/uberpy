#!/usr/bin/env python

import os
from collections import defaultdict
import numpy as np
from astropy.stats import sigma_clip
from astropy.time import Time,TimeDelta
import fitsio

#from .ubercal import CalibrationObject,CalibrationObjectSet,ubercal_solve
from ubercal import CalibrationObject,CalibrationObjectSet,ubercal_solve

import boklog

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
  ('20140629', '20140630', '20140701', '20140702', '20140703', 
   '20140705',),
  ('20140710', '20140711', '20140713', '20140716', '20140717', 
   '20140718',),
]

def bok_run_index():
	return {utd:run for run,utds in enumerate(bok_runs) for utd in utds}

def get_mjd(utDate,utTime):
	utStr = '-'.join([utDate[:4],utDate[4:6],utDate[6:]]) + ' ' + utTime
	return Time(utStr,scale='utc').mjd

def build_frame_list(filt,nightlyLogs=None):
	'''Collapse the nightly observing logs into a master frame list containing
	   the relevant infor for each observation, namely:
	     mjd,expTime,airmass   observing parameters
         nightIndex            0-indexed from the list of observing nights
	     nightFrameNum         frame number from observing logs
	'''
	if nightlyLogs is None:
		nightlyLogs = boklog.load_Bok_logs()
	frameList = []
	for night,utd in enumerate(sorted(nightlyLogs.keys())):
		frames = nightlyLogs[utd]
		ii = np.where((frames['filter']==filt) &
		              (frames['imType']=='object'))[0]
		mjds = [get_mjd(utd,frames['utStart'][i]) for i in ii]
		epochIndex = np.repeat(night,len(ii))
		frameList.append((mjds,frames['expTime'][ii],frames['airmass'][ii],
		                  epochIndex,ii))
	frameList = np.hstack(frameList)
	frameList = np.core.records.fromarrays(frameList,
	                     dtype=[('mjd','f8'),('expTime','f4'),('airmass','f4'),
	                            ('nightIndex','i4'),('nightFrameNum','i4')])
	return frameList

def collect_data(filt,catpfx='sdssbright'):
	photdir = os.path.join(os.environ['BOK90PRIMEOUTDIR'],'catalogs_v2')
	aperNum = -1
	mag0 = 25.0
	SNRcut = 20.0
	utd2run = bok_run_index()
	nightlyLogs = boklog.load_Bok_logs()
	frameList = build_frame_list(filt,nightlyLogs)
	objectList = defaultdict(list)
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
			objectList[starNum].append((mags,errs,ccdNums,ampNums,
			                            data['x'][good],data['y'][good],
			                            runIndex,nightIndex,jj))
	for starNum in objectList:
		arr = np.hstack(objectList[starNum])
		objectList[starNum] = np.core.records.fromarrays(arr,
		                        dtype=[('magADU','f4'),('errADU','f4'),
		                               ('ccdNum','i4'),('ampNum','i4'),
		                               ('x','f4'),('y','f4'),
		                               ('runIndex','i4'),('nightIndex','i4'),
	                                   ('frameIndex','i4')])
	return frameList,objectList

def cache_object_list(objectList,fileName):
	fits = fitsio.FITS(fileName,'rw')
	indx = np.empty(len(objectList),
	                dtype=[('starNum','i4'),('i1','i4'),('i2','i4')])
	i1 = 0
	for i,starNum in enumerate(objectList):
		print 'star ',i,' out of ',len(objectList)
		if i==0:
			fits.write(objectList[starNum])
		else:
			fits[-1].append(objectList[starNum])
		indx['starNum'][i] = starNum
		indx['i1'][i] = i1
		indx['i2'][i] = i1 + len(objectList[starNum])
		i1 += len(objectList[starNum])
	fits.write(indx)
	fits.close()

def load_cached_object_list(fileName):
	fits = fitsio.FITS(fileName)
	data = fits[1].read()
	indexes = fits[2].read()
	objectList = {}
	for starNum,i1,i2 in indexes:
		objectList[starNum] = data[i1:i2]
	return objectList

# frames = bokrmcal.build_frame_list('g')
# objs = bokrmcal.load_cached_object_list('test.fits')

def fiducial_model(frames,objs,verbose=True,**kwargs):
	nightlyLogs = boklog.load_Bok_logs() # just for numNights...
	numCCDs = 4
	numNights = len(nightlyLogs)
	numFrames = len(frames)
	a_init = np.zeros((numNights,numCCDs))
	k_init = np.zeros(numNights)
	flatfield_init = np.zeros((numCCDs,nY,nX))
	dt = np.zeros(numFrames) # for now
	rmcal = CalibrationObjectSet(a_init,k_init,dt,
	                             frames['airmass'],flatfield_init)
	#
	for starNum,obj in objs.items():
		if (starNum % 5) != 0:
			continue
		calobj = CalibrationObject(obj['magADU'],obj['errADU'])
		calobj.set_xy(obj['x'],obj['y'])
		calobj.set_a_indices(np.vstack([obj['nightIndex'],
		                                obj['ccdNum']-1]).transpose())
		calobj.set_k_indices(obj['nightIndex'])
		calobj.set_flat_indices(obj['ccdNum'])
		calobj.set_x_indices(obj['nightIndex'])
		calobj.set_t_indices(obj['frameIndex'])
		rmcal.add_object(calobj)
	if verbose:
		print 'number nights: ',numNights
		print 'number frames: ',numFrames
		print 'number objects: ',rmcal.num_objects()
		print 'number observations: ',rmcal.num_observations()
		print 'number parameters: ',rmcal.num_params()
	niter = 1
	for iternum in range(niter):
		pars = ubercal_solve(rmcal,**kwargs)
		rmcal.update_params(pars)
	return rmcal

def sim_fiducial_model(frames,objs,verbose=True,**kwargs):
	nightlyLogs = boklog.load_Bok_logs() # just for numNights...
	numCCDs = 4
	numNights = len(nightlyLogs)
	numFrames = len(frames)
	a_init = np.zeros((numNights,numCCDs))
	k_init = np.zeros(numNights)
	flatfield_init = np.zeros((numCCDs,nY,nX))
	dt = np.zeros(numFrames) # for now
	rmcal = CalibrationObjectSet(a_init,k_init,dt,
	                             frames['airmass'],flatfield_init)
	#
	a_true = 0.15 - 0.3*np.random.random_sample(a_init.shape)
	k_true = 0.2*np.random.random_sample(k_init.shape)
	#mag_sim = 18. + np.random.random_sample(len(objs))
	#k_true = 0.2 + 0*k_init
	mag_sim = np.repeat(18.,len(objs))
	#
	for i,(starNum,obj) in enumerate(objs.items()):
		if (starNum % 5) != 0:
			continue
		x = frames['airmass'][obj['frameIndex']]
		mags = mag_sim[i] + a_true[obj['nightIndex'],obj['ccdNum']-1] \
		          - k_true[obj['nightIndex']]*x
		errs = np.repeat(0.03,len(mags))
		calobj = CalibrationObject(mags,errs)
		calobj.set_xy(obj['x'],obj['y'])
		calobj.set_a_indices(np.vstack([obj['nightIndex'],
		                                obj['ccdNum']-1]).transpose())
		calobj.set_k_indices(obj['nightIndex'])
		calobj.set_flat_indices(obj['ccdNum'])
		calobj.set_x_indices(obj['frameIndex'])
		calobj.set_t_indices(obj['frameIndex'])
		rmcal.add_object(calobj)
	if verbose:
		print 'number nights: ',numNights
		print 'number frames: ',numFrames
		print 'number objects: ',rmcal.num_objects()
		print 'number observations: ',rmcal.num_observations()
		print 'number parameters: ',rmcal.num_params()
	niter = 1
	for iternum in range(niter):
		pars = ubercal_solve(rmcal,**kwargs)
		rmcal.update_params(pars)
	framesPerNight = [np.sum(frames['nightIndex']==i) 
	                    for i in range(numNights)]
	return rmcal,a_true,k_true

