# encoding: utf8
import pyximport
import numpy
pyximport.install(setup_args={'include_dirs':[numpy.get_include()]}, inplace=True)


import  GaussianProcess

class GPMD:
    def __init__(self, dim):
        self.__dim = dim
        self.__gp = [ GaussianProcess.GP() for d in range(self.__dim) ]

    def learn(self,x, y ):
        y = numpy.array(y)
        for d in range(self.__dim):
            if self.__dim==1:
                self.__gp[d].learn( x, y )
            else:
                if len(y)!=0:

                    self.__gp[d].learn( x, y[:,d] )

                else:
                    self.__gp[d].learn( x, [] )

    def learn2(self,x, y ):
        y = numpy.array(y)
        for d in range(self.__dim):
            if self.__dim==1:
                self.__gp[d].learn( x, y )
            else:
                if len(y)!=0:
                    self.__gp[d].learn( x, y[:,d] )
                else:
                    self.__gp[d].learn( x, [] )

    def calc_lik(self, x, y , max_len):
        lik = 0.0
        y = numpy.array(y)

        for d in range(self.__dim):
            if self.__dim==1:
                lik += self.__gp[d].calc_lik( numpy.array(x,dtype=numpy.float) , y, max_len)

            else:
                lik += self.__gp[d].calc_lik( numpy.array(x,dtype=numpy.float) , y[:,d], max_len)

        return lik

    def plot(self, rng ):
        for d in range(self.__dim):
            pylab.subplot( self.__dim, 1, d+1 )
            self.__gp[d].plot(rng)

    def draw(self,dim,rng):
        self.__gp[dim].plot(rng)

    def generate(self, rng):
        gendata = numpy.zeros( (len(rng), self.__dim) )
        for d in range(self.__dim):
            for i,y in enumerate(self.__gp[d].generate(rng)):
                gendata[i,d] = y

        return gendata


    def generate2(self, rng):        
        gendata_mi = numpy.zeros( (len(rng), self.__dim) )
        gendata_lo = numpy.zeros( (len(rng), self.__dim) )
        gendata_hi = numpy.zeros( (len(rng), self.__dim) )
        gendata_si = numpy.zeros( (len(rng), self.__dim) )

        for d in range(self.__dim):
            y_lo, y_mi, y_hi,y_si =self.__gp[d].generate2(rng)
            for i in range(len(y_lo)):
                gendata_lo[i,d] = y_lo[i]
                gendata_mi[i,d] = y_mi[i]
                gendata_hi[i,d] = y_hi[i]
                gendata_si[i,d] = y_si[i]

        return gendata_lo,gendata_mi,gendata_hi,gendata_si

    def generatesigma(self, rng):
        gendata_sigma = numpy.zeros( (len(rng), self.__dim) )
        print("make")
        y_lo, y_mi, y_hi,y_si =self.__gp.generate2(rng)
        gendata_sigma = y_si
        return gendata_sigma

def main():
    gp = GPMD( 2 )
    gp.learn([],[])

    data = [[1.0,1.0],[1.0,1.0],[1.0,1.0]]

    gp.plot( range(0,3) )
    pylab.show()
    print gp.calc_lik( range(3), data )


if __name__ == '__main__':
    main()