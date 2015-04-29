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
		mask = errs <= 0
		self.mags = np.ma.masked_array(mags,mask)
		self.ivars = np.ma.masked_array(np.clip(errs,1e-10,np.inf)**-2,mask)
		self.nobs = len(self.mags)
		self.a_indices = None
		self.k_indices = None
		self.t_indices = None
		self.x_indices = None
		self.flat_indices = None
	def set_a_indices(self,a_indices):
		self.a_indices = a_indices
	def set_k_indices(self,k_indices):
		self.k_indices = k_indices
	def set_t_indices(self,t_indices):
		self.t_indices = t_indices
	def set_x_indices(self,x_indices):
		self.x_indices = x_indices
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
		self.aTerms = aTerms.copy()
		self.kTerms = kTerms.copy()
		self.tVals = tVals
		self.airmasses = airmasses
		self.flatfields = flatfields
		self.na = self.aTerms.size
		self.nk = self.kTerms.size
		self.npar = self.na + self.nk
		self.nobs = 0
	def add_object(self,calobj):
		self.objs.append(calobj)
		self.nobs += calobj.get_numobs()
	def num_params(self):
		return self.npar
	def num_objects(self):
		return len(self.objs)
	def num_observations(self):
		return self.nobs
	def get_aterms(self,a_indices):
		return self.aTerms[a_indices]
	def get_kterms(self,k_indices):
		return self.kTerms[k_indices]
	def get_obstimes(self,t_indices):
		return self.tVals[t_indices]
	def get_airmasses(self,x_indices):
		return self.airmasses[x_indices]
	def update_params(self,p):
		i0 = 0
		for nterms,terms in zip([self.na,self.nk],[self.aTerms,self.kTerms]):
			terms[:] = p[i0:i0+nterms].reshape(terms.shape)
			i0 += nterms
	def __iter__(self):
		for obj in self.objs:
			yield obj

def ubercal_solve(calset,**kwargs):
	'''Find the best fit parameter values by solving the least squared problem
	   given in Padmanabhan et al. (2008) eq. 14.
	   Returns the updated parameters.
	'''
	#from scipy.sparse import hstack,eye,diags
	minNobs = kwargs.get('minNobs',1)
	bigmatrix = kwargs.get('bigmatrix',False)
	#rmsFloor = kwargs.get('rmsFloor',0.02)
	#
	npar = calset.num_params()
	nobs = calset.num_observations()
	if bigmatrix:
		A = np.zeros((nobs,npar))
		b = np.zeros(nobs)
		i1 = 0
	else:
		atcinvb = np.zeros(npar)
		atcinva = np.zeros((npar,npar))
	# iterate over all objects (stars)
	for n,obj in enumerate(calset):
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
		# translate indices from the term arrays (which may be 
		# multidimensional) into indices in parameter array.
		# ordering of parameter array is [a1..aN,k1..kN,(dkdt1..dkdtN)]
		i0 = 0
#		par_a_indx = [i0 + np.ravel_multi_index(ai,calset.aTerms.shape)
#		                for ai in a_indices]
		par_a_indx = i0 + a_indices # XXX for single dim
		i0 += calset.na
#		par_k_indx = [i0 + np.ravel_multi_index(ki,calset.kTerms.shape)
#		                for ki in k_indices]
		par_k_indx = i0 + k_indices # XXX for single dim
		i0 += calset.nk
		assert i0==npar
		nobs_i = obj.get_numobs()
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
			b[i1:i2] = (m_mean - m_inst)*ivar_inst 
			for i in range(nobs_i):
				A[i1+i,par_a_indx[i]] = 1
				A[i1+i,par_k_indx[i]] = -x[i]
				for j in range(nobs_i):
					A[i1+i,par_a_indx[j]] -= w[j]
					A[i1+i,par_k_indx[j]] += w[j]*x[j]
			i1 += nobs_i
			continue
		# b column vector (eq. 13)
		b = (m_mean - m_inst)*ivar_inst
		wb = np.sum(b)*w
		# fill matrix by groups of a and k terms
		for i,(ai,ki) in enumerate(zip(par_a_indx,par_k_indx)):
			atcinvb[ai] += b[i] - wb[i]
			atcinvb[ki] -= (b[i] - wb[i])*x[i]
		#
		# construct << A^T * C^-1 * A >>
		#
		at_sub = np.eye(nobs_i) - np.tile(w,(nobs_i,1))
		a_sub = at_sub.transpose().copy()
		for i in range(nobs_i):
			a_sub[:,i] *= ivar_inst
		wt = np.dot(at_sub,a_sub)
		for i in range(nobs_i):
			for j in range(nobs_i):
				atcinva[par_a_indx[i],par_a_indx[j]] += wt[i,j]
		for i in range(nobs_i):
			for j in range(nobs_i):
				atcinva[par_k_indx[i],par_k_indx[j]] += wt[i,j]*x[i]*x[j]
				atcinva[par_k_indx[i],par_a_indx[j]] -= wt[i,j]*x[i]
				atcinva[par_a_indx[i],par_k_indx[j]] -= wt[i,j]*x[j]
	if bigmatrix:
		atcinvb = np.dot(A.T,b)
		B = A.copy()
		i1 = 0
		for n,obj in enumerate(calset):
			m_inst,ivar_inst = obj.get_instrumental_mags()
			nobs_i = obj.get_numobs()
			B[i1:i1+nobs_i] = ivar_inst[:,np.newaxis]*B[i1:i1+nobs_i]
			i1 += nobs_i
		atcinva = np.dot(A.T,B)
		# could construct a sparse diagonal matrix for cinv and do it this way
		#atcinva = np.dot(np.dot(A.T,np.diag(cinv)),A)
	# Solve for p 
	p, _, _, _ = np.linalg.lstsq(atcinva,atcinvb)
	return p


