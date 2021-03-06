#!/usr/bin/env python

from collections import defaultdict
import numpy as np
from scipy.interpolate import LSQBivariateSpline

######### ######### ######### ######### ######### ######### ######### ######### 
#########                       Flat Field models                     #########
######### ######### ######### ######### ######### ######### ######### ######### 

class NullFlatField(object):
	def __init__(self,*args):
		return
	def fit(self,*args,**kwargs):
		return
	def __call__(self,x,y):
		return np.zeros_like(x)

class SplineFlatField(object):
	def __init__(self,nx,ny):
		self.nx = nx
		self.ny = ny
		# XXX hardcoding in the knots and spline order for now
		self.tx = [0,nx/2,nx]
		self.ty = [0,ny/2,ny]
		self.kx = 3
		self.ky = 3
		self.splineFit = lambda x,y: np.zeros_like(x)
	def fit(self,x,y,f,ivar=None):
		self.splineFit = LSQBivariateSpline(x,y,f,self.tx,self.ty,w=ivar,
		                                    kx=self.kx,ky=self.ky)
	def __call__(self,x,y):
		return np.array( [ self.splineFit(_x,_y).squeeze() 
		                            for _x,_y in zip(x,y) ] )
	def make_image(self,res=1):
		x = np.arange(0,self.nx,res)
		y = np.arange(0,self.ny,res)
		return self.splineFit(x,y).transpose()

class FlatFieldSet(object):
	def __init__(self,shape):
		self.shape = shape
		self.flatfields = []
	def get_shape(self):
		return self.shape
	def __call__(self,indices,x,y):
		rv = []
		ii = np.ravel_multi_index(indices,self.shape)
		for i,flat in enumerate(self.flatfields):
			jj = np.where(ii==i)[0]
			if len(jj)>0:
				rv.append(flat(x[jj],y[jj]))
		return np.concatenate(rv)
	def __iter__(self):
		for ff in self.flatfields:
			yield ff

def init_flatfields(shape,nx,ny,method='spline',**kwargs):
	flatfields = FlatFieldSet(shape)
	print flatfields
	if method=='spline':
		generator = SplineFlatField
	elif method=='array':
		generator = lambda nx,ny,**kwargs: np.zeros((nY,nX))
	elif method=='null':
		generator = NullFlatField
	for i in range(np.product(shape)):
		flatfields.flatfields.append(generator(nx,ny,**kwargs))
	return flatfields



######### ######### ######### ######### ######### ######### ######### ######### 
#########                   Ubercalibration algorithm                 #########
######### ######### ######### ######### ######### ######### ######### ######### 

class CalibrationObject(object):
	def __init__(self,mags,errs,errMin=0.01):
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
		self.ivars = np.ma.masked_array(np.clip(errs,errMin,np.inf)**-2,mask)
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
	def set_reference_mag(self,refMag):
		self.refMag = refMag
	def get_numobs(self):
		return self.nobs
	def get_instrumental_mags(self):
		return self.mags,self.ivars
	def get_term_indices(self):
		return (self.a_indices,self.k_indices,
		        self.t_indices,self.x_indices,self.flat_indices)
	def get_xy(self):
		return (self.xpos,self.ypos)
	def update_mask(self,mask):
		self.mags.mask |= mask
		self.ivars.mask |= mask

class CalibrationObjectSet(object):
	def __init__(self,aTerms,kTerms,tVals,airmasses,flatfields,**kwargs):
		fit_a = kwargs.get('fit_a',True)
		fit_k = kwargs.get('fit_k',True)
		fit_dkdt = kwargs.get('fit_dkdt',False)
		fit_flat = kwargs.get('fit_flat',False)
		dTerms = np.array([])
		fTerms = np.array([])
		if kwargs.get('flat_poly2d',False):
			fTerms = kwargs.get('flat_poly2d')
		elif kwargs.get('flat_iteratedfit',False):
			pass
		#
		self.objs = []
		self.tVals = tVals
		self.airmasses = airmasses
		self.flatfields = flatfields
		self.nobs = 0
		self.params = {
		     'a':{ 'fit':fit_a,    'terms':aTerms, 'num':aTerms.size },
		     'k':{ 'fit':fit_k,    'terms':kTerms, 'num':kTerms.size },
		  'dkdt':{ 'fit':fit_dkdt, 'terms':dTerms, 'num':dTerms.size },
		  'flat':{ 'fit':fit_flat, 'terms':fTerms, 'num':fTerms.size },
		}
		self.npar = np.array([self.params[paramName]['num'] 
		                        for paramName in ['a','k','dkdt','flat']
		                          if self.params[paramName]['fit']])
	def add_object(self,calobj):
		self.objs.append(calobj)
		self.nobs += calobj.get_numobs()
	def set_fixed_dkdt(self,dkdt):
		self.params['dkdt']['terms'] = np.array([dkdt,])
	def get_object_phot(self,obj,returnBoth=False):
		ai,ki,ti,xi,fi = obj.get_term_indices()
		m_inst,ivar_inst = obj.get_instrumental_mags()
		a = self.params['a']['terms'][ai]
		k = self.params['k']['terms'][ki]
		x = self.get_airmasses(xi)
		dt = self.get_obstimes(ti)
		dk_dt = self.get_terms('dkdt',0) # using a fixed value
		flatfield = self.get_flatfields(fi,*obj.get_xy())
		m_cal = m_inst + a - (k + dk_dt*dt)*x + flatfield
		if returnBoth:
			return m_cal,1/np.sqrt(ivar_inst),m_inst
		else:
			return m_cal,1/np.sqrt(ivar_inst)
	def num_params(self):
		return np.sum(self.npar)
	def num_objects(self):
		return len(self.objs)
	def num_observations(self):
		return self.nobs
	def get_terms(self,p,indices):
		if self.params[p]['terms'] is None:
			return 0
		else:
			return self.params[p]['terms'][indices]
	def get_obstimes(self,t_indices):
		return self.tVals[t_indices]
	def get_airmasses(self,x_indices):
		return self.airmasses[x_indices]
	def get_flatfields(self,flat_indices,x,y):
		return self.flatfields(flat_indices,x,y)
	def update_params(self,par):
		i0 = 0
		for p in ['a','k','dkdt','flat']:
			nterms = self.params[p]['num']
			if self.params[p]['fit']:
				shape = self.params[p]['terms'].shape
				self.params[p]['terms'].data[:] = \
				                         par[i0:i0+nterms].reshape(shape)
			i0 += nterms
	def update_flatfields(self):
		resv = defaultdict(list)
		for obj in self.objs:
			mag,ivar = self.get_object_phot(obj) # actually returns err
			_,_,_,_,fi = obj.get_term_indices()
			x,y = obj.get_xy()
			ivar[ivar>0] **= -2 # convert to inverse variance
			ii = np.ravel_multi_index(fi,self.flatfields.get_shape())
			for i,flat in enumerate(self.flatfields):
				# XXX should assume masking is correct here and be cleaner?
				jj = np.where((ii==i) & (ivar>0) & ~mag.mask)[0]
				if len(jj) > 0:
					# XXX refmag should be optional, default is just offset
					#     to weighted mean mag
					dmag = obj.refMag - mag[jj]
					resv[i].append((x[jj],y[jj],dmag,ivar[jj]))
		for i,flat in enumerate(self.flatfields):
			resarr = np.hstack(resv[i])
			flat.fit(*resarr)
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
					par_ii = i0 + np.ravel_multi_index(indices,pshape)
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
		a = calset.get_terms('a',a_indices)
		k = calset.get_terms('k',k_indices)
		x = calset.get_airmasses(x_indices)
		dk_dt = calset.get_terms('dkdt',0) # using a fixed value
		dt = calset.get_obstimes(t_indices)
		flatfield = calset.get_flatfields(flat_indices,*obj.get_xy())
		# construct indices into the parameter axis
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
		# parameters and for objects with poorly defined free parameters
		# XXX need to check here that masked arrays are being handled properly
		a_bad = np.ma.masked_array(a.data,~a.mask).filled(0)
		k_bad = np.ma.masked_array(k.data,~k.mask).filled(0)
		m_inst.data[:] += a_bad - (k_bad + dk_dt*dt)*x + flatfield
		# normalized inverse variance weights
		w = ivar_inst / np.sum(ivar_inst)
		# inverse-variance-weighted mean instrumental magnitude
		m_mean = np.sum(w*m_inst)
		# if requested, construct the large matrices instead
		if bigmatrix:
			i2 = i1 + nobs_i
			_ii = np.arange(i1,i2)      # indexes into rows of A (observations)
			b[i1:i2] = -(m_inst - m_mean)
			cinv[i1:i2] = ivar_inst
			A[_ii,par_a_indx] = 1
			A[_ii,par_k_indx] = -x
			np.add.at( A[i1:i2], (ii,aj),    -w[jj] )
			np.add.at( A[i1:i2], (ii,kj), (w*x)[jj] )
			i1 += nobs_i
			continue
		# b column vector (eq. 13)
		b = -(m_inst - m_mean)*ivar_inst
		wb = np.sum(b)*w
		np.add.at( atcinvb, par_a_indx,   b-wb    )
		np.add.at( atcinvb, par_k_indx, -(b-wb)*x )
		#
		# construct << A^T * C^-1 * A >>
		#
		at_sub = np.eye(nobs_i) - np.tile(w,(nobs_i,1))
		wt = np.dot(at_sub,np.transpose(ivar_inst*at_sub))
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


