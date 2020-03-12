# encoding: utf8
#from __future__ import unicode_literals
import numpy
import matplotlib.pyplot as plt
import random
import matplotlib.mlab as mlab


cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double log(double)


cdef class GP:
    cdef double beta
    cdef int ns
    cdef xt, yt
    cdef double[:,:] i_cov
    cdef double[:] param
    cdef mu_sigma

    cpdef double covariance_func(self, double xi, double xj):
        #theta0 = 1.0
        #theta1 = 16.0*10
        #theta2 = 0.0
        #theta3 = 16.0
        cdef double theta0 = 1.0
        cdef double theta1 = 1.0
        cdef double theta2 = 0
        cdef double theta3 = 16.0
        return theta0 * exp(-0.5 * theta1 * (xi - xj) * (xi - xj)) + theta2 + theta3 * xi * xj

    cpdef double normpdf(self, double x, double mu, double sigma):
        return 1./(sqrt(2*numpy.pi)*sigma)*exp(-0.5 * (1./sigma*(x - mu))**2)


    def __init__( self,beta=10.0 ):
        self.beta = beta
        self.mu_sigma = {}
    def learn(self, xt, yt ):
        cdef int i,j
        self.xt = xt
        self.yt = yt

        self.ns = len(xt)

        # construct covariance
        cdef double[:,:] cov = numpy.zeros((self.ns, self.ns))
        for i in range(self.ns):
            for j in range(self.ns):
                name = "{0:d}_{1:d}".format(xt[i], xt[j])
                cov[i,j] = self.covariance_func(xt[i], xt[j])
                if i==j:
                    cov[i,j] += 1/self.beta


        self.i_cov = numpy.linalg.inv(cov)
        self.param = numpy.dot(self.i_cov, self.yt)
        self.mu_sigma = {}


    def plot( self, xs ):
        n = len(xs)
        y_lo = [0.0] * n
        y_mi = [0.0] * n
        y_hi = [0.0] * n
        tt = [y - numpy.random.normal() / self.beta for y in self.yt]
        for k in range(n):
            v = numpy.zeros((self.ns))
            for i in range(self.ns):
                v[i] = self.covariance_func(xs[k], self.xt[i])
            c = self.covariance_func(xs[k], xs[k]) + 1.0 / self.beta
            mu = numpy.dot(v, numpy.dot(self.i_cov, tt))
            sigma = c - numpy.dot(v, numpy.dot(self.i_cov, v))

            y_lo[k] = mu - sigma * 2.0
            y_mi[k] = mu
            y_hi[k] = mu + sigma * 2.0

        #plt.scatter(self.xt, self.yt, marker='x' ,color='k')
        plt.fill_between( xs, y_lo, y_hi, facecolor="lavender" , alpha=0.9 , edgecolor="lavender"  )
        plt.plot(xs, y_lo, 'b--')
        plt.plot(xs, y_mi, 'b-')
        plt.plot(xs, y_hi, 'b--')

        #plt.xlim([-1.2, 1.2])
        #plt.ylim([-5.0, 5.0])
        #plt.show()

    def generate(self, xs):
        n = len(xs)
        y_mi = [0.0] * n
        tt = [y - numpy.random.normal() / self.beta for y in self.yt]
        for k in range(n):
            v = numpy.zeros((self.ns))
            for i in range(self.ns):
                v[i] = self.covariance_func(xs[k], self.xt[i])
            mu = numpy.dot(v, numpy.dot(self.i_cov, tt))

            y_mi[k] = mu

        return y_mi

    def generate2(self, xs):
        n = len(xs)
        y_lo = [0.0] * n
        y_mi = [0.0] * n
        y_hi = [0.0] * n
        sig  = [0.0] * n
        tt = [y - numpy.random.normal() / self.beta for y in self.yt]
        for k in range(n):
            v = numpy.zeros((self.ns))
            for i in range(self.ns):
                v[i] = self.covariance_func(xs[k], self.xt[i])
            c = self.covariance_func(xs[k], xs[k]) + 1.0 / self.beta
            mu = numpy.dot(v, numpy.dot(self.i_cov, tt))
            sigma = c - numpy.dot(v, numpy.dot(self.i_cov, v))
            sig[k]  = sigma
            y_lo[k] = mu - sigma 
            y_mi[k] = mu
            y_hi[k] = mu + sigma 
        return y_lo,y_mi,y_hi,sig

    def generate3(self, xs):
        n = len(xs)
        y_lo = [0.0] * n
        y_mi = [0.0] * n
        y_hi = [0.0] * n
        sig  = [0.0] * n
        tt = [y - numpy.random.normal() / self.beta for y in self.yt]
        for k in range(n):
            v = numpy.zeros((self.ns))
            for i in range(self.ns):
                v[i] = self.covariance_func(xs[k], self.xt[i])
            c = self.covariance_func(xs[k], xs[k]) + 1.0 / self.beta
            mu = numpy.dot(v, numpy.dot(self.i_cov, tt))
            sigma = c - numpy.dot(v, numpy.dot(self.i_cov, v))
            sig[k]  = sigma
            y_lo[k] = mu - sigma 
            y_mi[k] = mu
            y_hi[k] = mu + sigma 
        return y_lo,y_mi,y_hi,sig        

    cpdef double calc_lik( self, double[:] xs, double[:] ys , double max_len):

        cdef int k,i
        cdef int n = len(xs)
        cdef double lik = 0
        cdef int ns = self.ns
        cdef double c,p,mu,sigma



        cdef double[:] y_lo = numpy.zeros( n )
        cdef double[:] y_mi = numpy.zeros( n )
        cdef double[:] y_hi = numpy.zeros( n )
        # cdef double[:] tt = numpy.array( self.yt )
        cdef double[:] v= numpy.zeros((ns))
        #cdef double[:,:] i_cov  = self.i_cov


        #y_lo = [0.0] * n
        #y_mi = [0.0] * n
        #y_hi = [0.0] * n
        #tt = [y for y in self.yt]


        for k in range(n):
            if xs[k] in self.mu_sigma:
                # 既に計算してあれば使い回す
                mu = self.mu_sigma[xs[k]][0]
                sigma = self.mu_sigma[xs[k]][1]
            else:
                v = numpy.zeros((ns))
                for i in range(ns):
                    v[i] = self.covariance_func(xs[k], self.xt[i])
                c = self.covariance_func(xs[k], xs[k]) + 1.0 / self.beta
                #mu = numpy.dot(v, numpy.dot(i_cov, tt))
                mu = numpy.dot(v, self.param)
                sigma = c - numpy.dot(v, numpy.dot(self.i_cov, v))

                # 計算した結果を保存しておく
                self.mu_sigma[xs[k]] = (mu,sigma)

            p = self.normpdf( ys[k] , mu, sigma )
            if p<=0:
                p = 0.000000000001
            #print ys[k], mu, p
            lik += log( p )

        #print "------------------------"

        return lik



if __name__=='__main__':
    pass