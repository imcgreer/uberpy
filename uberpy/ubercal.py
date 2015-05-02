#!/usr/bin/env python

import numpy as np

class CalibrationObject(object):
	def __init__(self,mags,errs):
		'''CalibrationObject(mags,errs) 
		    An object (star or galaxy) with multiple observations that can be
		    used for relative photometric calibration. The object is defined
		    by instrumental magnitudes and errors at each epoch of observation,
		    the zeropoint (a) and airmass (k) terms corresponding to that
		    observation, the time of observation (t), and the flatfield 
		    for that observation.
		    Each term is defined by a set of indices into the master list
		    and is defined for an object by the set_TERM_indices() method.
		    Not defining the term indices means that the term will be
		    ignored.
		   INPUT:
		   mags,errs: n-element vectors containing instrumental magnitudes
		              and errors, measured in ADU
		'''
		#mask = errs <= 0
		#self.mags = np.ma.masked_array(mags,mask)
		#self.ivars = np.ma.masked_array(np.clip(errs,1e-10,np.inf)**-2,mask)
		self.mags = mags
		self.ivars = np.clip(errs,1e-10,np.inf)**-2
		self.nobs = len(self.mags)
		self.a_indices = None
		self.k_indices = None
		self.t_indices = None
		self.x_indices = None
		self.flat_indices = None
		self.xpos = None
		self.ypos = None
	def set_xy(self,x,y):
		self.xpos = x
		self.ypos = y
	def set_a_indices(self,a_indices):
		self.a_indices = a_indices
	def set_k_indices(self,k_indices):
		self.k_indices = k_indices
	def set_t_indices(self,t_indices):
		self.t_indices = t_indices
	def set_x_indices(self,x_indices):
		self.x_indices = x_indices
	def set_flat_indices(self,flat_indices):
		self.flat_indices = flat_indices
	def get_numobs(self):
		return self.nobs
	def get_instrumental_mags(self):
		return self.mags,self.ivars
	def get_term_indices(self):
		return (self.a_indices,self.k_indices,
		        self.t_indices,self.x_indices,self.flat_indices)

class CalibrationObjectSet(object):
	def __init__(self,aTerms,kTerms,tVals,airmasses,flatfields):
		self.objs = []
#		self.aTerms = aTerms.copy()
#		self.kTerms = kTerms.copy()
		self.tVals = tVals
		self.airmasses = airmasses
		self.flatfields = flatfields
		self.nobs = 0
		self.params = {
		     'a':{ 'fit':True,  'terms':aTerms, 'num':aTerms.size },
		     'k':{ 'fit':True,  'terms':kTerms, 'num':kTerms.size },
		  'dkdt':{ 'fit':False, 'terms':None,   'num':0 },
		  'flat':{ 'fit':False, 'terms':None,   'num':0 },
		}
		self.npar = np.array([self.params[paramName]['num'] 
		                        for paramName in ['a','k','dkdt','flat']
		                          if self.params[paramName]['fit']])
	def add_object(self,calobj):
		self.objs.append(calobj)
		self.nobs += calobj.get_numobs()
	def num_params(self):
		return np.sum(self.npar)
	def num_objects(self):
		return len(self.objs)
	def num_observations(self):
		return self.nobs
	def get_aterms(self,a_indices):
		return self.params['a']['terms'][a_indices]
	def get_kterms(self,k_indices):
		return self.params['k']['terms'][k_indices]
	def get_obstimes(self,t_indices):
		return self.tVals[t_indices]
	def get_airmasses(self,x_indices):
		return self.airmasses[x_indices]
	def update_params(self,par):
		i0 = 0
		for p in ['a','k','dkdt','flat']:
			nterms = self.params[p]['num']
			if self.params[p]['fit']:
				shape = self.params[p]['terms'].shape
				self.params[p]['terms'] = par[i0:i0+nterms].reshape(shape)
			i0 += nterms
	def __iter__(self):
		for obj in self.objs:
			yield obj
	def parameter_indices(self,paramName,indices):
		if not self.params[paramName]['fit']:
			raise ValueError
		i0 = 0
		for p in ['a','k','dkdt','flat']:
			if paramName == p:
				pshape = self.params[p]['terms'].shape
				if len(pshape)==1:
					par_ii = i0 + np.asarray(indices)
				else:
					par_ii = [i0 + np.ravel_multi_index(ii,pshape)
					                for ii in indices]
				break
			else:
				i0 += self.params[p]['num']
		return np.array(par_ii)

def ubercal_solve(calset,**kwargs):
	'''Find the best fit parameter values by solving the least squared problem
	   given in Padmanabhan et al. (2008) eq. 14.
	   Returns the updated parameters.
	'''
	minNobs = kwargs.get('minNobs',1)
	bigmatrix = kwargs.get('bigmatrix',False)
	#rmsFloor = kwargs.get('rmsFloor',0.02)
	#
	npar = calset.num_params()
	nobs = calset.num_observations()
	if bigmatrix:
		A = np.zeros((nobs,npar))
		b = np.zeros(nobs)
		cinv = np.zeros(nobs)
		i1 = 0
	else:
		atcinvb = np.zeros(npar)
		atcinva = np.zeros((npar,npar))
	# iterate over all objects (stars)
	for n,obj in enumerate(calset):
		if ((n+1)%50)==0:
			print 'star #',n+1,' out of ',calset.num_objects()
		# collect all observations of this object and the associated
		# calibration terms
		m_inst,ivar_inst = obj.get_instrumental_mags()
		a_indices,k_indices,t_indices,x_indices,flat_indices = \
		                                    obj.get_term_indices()
		a = calset.get_aterms(a_indices)
		k = calset.get_kterms(k_indices)
		dk_dt = 0 # placeholder
		x = calset.get_airmasses(x_indices)
		#flatfield = calset.get_flatfields(flat_indices)
		flatfield = 0 # placeholder
		dt = 0 # placeholder
		# construct indices
		nobs_i = obj.get_numobs()
		par_a_indx = calset.parameter_indices('a',a_indices)
		par_k_indx = calset.parameter_indices('k',k_indices)
		ii = np.repeat(np.arange(nobs_i),nobs_i)
		jj = np.tile(np.arange(nobs_i),nobs_i)
		ai,aj = par_a_indx[ii],par_a_indx[jj]
		ki,kj = par_k_indx[ii],par_k_indx[jj]
		#
		# construct << A^T * C^-1 * B >>
		#
		# update instrumental magnitude based on current values for fixed
		# parameters and for for objects with poorly defined free parameters
		# XXX currently skipped
		#m = m_inst + a - (k + dk_dt*dt)*x + flatfield
		# normalized inverse variance weights
		w = ivar_inst / np.sum(ivar_inst)
		# inverse-variance-weighted mean instrumental magnitude
		m_mean = np.sum(w*m_inst)
		# if requested, construct the large matrices instead
		if bigmatrix:
			i2 = i1 + nobs_i
			_ii = np.arange(i1,i2)      # indexes into rows of A (observations)
			b[i1:i2] = m_inst - m_mean
			cinv[i1:i2] = ivar_inst
			A[_ii,par_a_indx] = 1
			A[_ii,par_k_indx] = -x
			np.add.at( A[i1:i2], (ii,aj),    -w[jj] )
			np.add.at( A[i1:i2], (ii,kj), (w*x)[jj] )
			i1 += nobs_i
			continue
		# b column vector (eq. 13)
		b = (m_inst - m_mean)*ivar_inst
		wb = np.sum(b)*w
		np.add.at( atcinvb, par_a_indx,   b-wb    )
		np.add.at( atcinvb, par_k_indx, -(b-wb)*x )
		#
		# construct << A^T * C^-1 * A >>
		#
		at_sub = np.eye(nobs_i) - np.tile(w,(nobs_i,1))
		a_sub = at_sub.transpose().copy()
		for i in range(nobs_i):
			a_sub[:,i] *= ivar_inst
		wt = np.dot(at_sub,a_sub)
		#
		np.add.at( atcinva, (ai,aj),  wt[ii,jj]             )
		np.add.at( atcinva, (ai,kj), -wt[ii,jj]*x[jj]       )
		np.add.at( atcinva, (ki,aj), -wt[ii,jj]*x[ii]       )
		np.add.at( atcinva, (ki,kj),  wt[ii,jj]*x[ii]*x[jj] )
	if bigmatrix:
		atcinvb = np.dot(A.T,cinv*b)
		# should use scipy.sparse here but getting a warning
		atcinva = np.dot(np.dot(A.T,np.diag(cinv)),A)
	# Solve for p 
	p, _, _, _ = np.linalg.lstsq(atcinva,atcinvb)
	return p


