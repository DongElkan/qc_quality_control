# -*- coding: utf-8 -*-
"""
SirQR baseline correction

SirQR is a baseline correction algorithm based on iteratively reweighted quantile
regression which is robust, computation efficient, flexible and effective. This
program can be used as a standalone command line for future modifications and
reuse in new project.
Since The coefficient matrix "B" is the concatenation of an identity matrix and
adjusted matrix to transform the smoothing series z into differences of neighboring
elements, the latter of which is a bidiagonal matrix, the multiplication of the
coefficient matrix and other matrices or vectors is equivalent to concatenate
these matrices or vectors and the differences of them (i.e. diff(X)). As matrix
multiplication by numpy (via "dot" function) is very slow, it is much faster to
use numberic computation directly, for example "diff" and "concatenate" functions,
in quantile regression. Further, because the input y for quantile regression has
been extended by a zero vector, we omit the dot product operation involving this.
To retain the original codes in MATLAB, codes for matrix multiplications are
provided in the annotations.

Reference:
Liu, X. B.; Zhang, Z. M.; Sousa, P. F. M.; Chen, C.; Ouyang, M. L.; Wei, Y. C.;
Liang, Y. Z.; Chen, Y.; Zhang, C. P. Selective iteratively reweighted quantile
regression for baseline correction. Anal Bioanal Chem. 2014, 406, 1985-1998.

LICENCE

All Copyrights to Naiping Dong (np.dong572@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from numpy import dot, diff, ones, zeros, concatenate, exp, amin, sum, abs, max, array
from multiprocessing import Pool, cpu_count


#def bs_check(X):
#    """ Sudden drop baseline check to eleminate the influence of negative baseline
#    drift """
#    minx = min(X)
#    if minx>=-5.0: return X
#    
#    mins = min(X,axis=1)
    

#@profile
def lp_fnm(m,c,theta,w,penlambda,
           p      = 0.9995,
           ebs    = 1e-5,
           maxiter= 50):
    """
    Sovlving linear programming problem for quantile regression using interior
    point method with Mehrotra corrector[1]. The problem can be characterized as
            min(c'*x), s.t. A*x = b and 0<x<1
    All quantile regression codes are translated from MATLAB codes obtained from
    http://www.econ.uiuc.edu/~roger/research/rq/rq.html
    The detail of the algorithm can be found in his text book of quantile
    regression by Koenker, R [2].
    
    NOTE: [1] In solving linear programming problems, majority computation times
              are spent inversing the AQ.T to solve linear equations AQ*dy = rhs.
              Fortunately, as coefficient matrix AQ or A is special, i.e., a 
              concatenation of an identity matrix and the column difference of the
              identity matrix, AQ'*AQ is a tridiagonal matrix. Thus, for equation
              AQ*dy = rhs, a transformation can be obtained as:
                         AQ'*AQ*dy = AQ'*rhs
              Notice that AQ here is the row concatenation of two matrices since
              in the original MATLAB code the input A is the transformed coefficient
              matrix. The new linear equation system therefore has a special
              coefficient matrix, and efficient algorithm named Thomas algorithm
              can be applied to solve this. This leads current implementation a
              linear time complexity.
          [2] To further speed up the computation, the equation solving procedure
              is imbedded in the linear programming implementation. This can speed
              up about 15%. Using pythonic operation in iteration, the speed is
              improved even for 4 times. The disadvantage is that the code is not
              so clearly readable.
    
    
    ---------------------------------------------------------------------------
    Inputs:
        m:          size of the input raw data;
        c:          Coefficient vector in object function c'*x;
        theta:      Quantile of interest;
        w:          Weights, if no, set them to 1.0;
        penlambda:  Penalty value for baseline correction;
        p:          Adjustment parameter, suggested to be 0.9995;
        ebs:        Minimum gap between primal and dual function, default is 1e-5;
        maxiter:    Max iteration, default is 50;
    
    Output:
        y:          The x vector in linear programming problem as well as the
                    coefficient vector in quantile regression if we suppose that
                    the coefficient matrix A mentioned above is the input variables
                    for regression. Namely, Ax = b.
    
    Reference:
    [1] Mehrotra, S. On the Implementation of a Primal-Dual Interior Point Method. 
        SIAM Journal on Optimization. 1992, 2, 575-601.
    [2] Koenker, R. Quantile Regression, Econometric Society Monograph Series,
        Cambridge University Press, 2005.
    """
    
    # This is column number of coefficient matrix * 2, for line 248, i.e.
    # mu = mu*(g/mu)**3/col
    col= 4*m-2
    l  = penlambda**2
    c  = c*-1.0
    # paramters in solving tridiagonal linear equation system
    # ta is the vector for i,i+1 and i+1,i; tb is the diagonal vector in tridiagonal
    # coefficient matrix A; and td is y vector for A*x = y; t is a temporary vector
    t  = zeros(m-1)
    ta = -ones((m-1,1))*l
    tb = ones((m,1))+l
    tb[1:m-1] += l
    td = c*1.0
    
    # ...Generate initial feasible point
    s = theta*w
    a = w-s
    # dot(A.T,a)
    b       = a[:m]*1
    b[0]   -= a[m]*penlambda
    b[1:-1]-= penlambda*diff(a[m:])
    b[-1]  += a[-1]*penlambda
    # >>>>> solving tridiagonal linear equation system
    # y = (A'\c')'
    ta[0] /= tb[0]
    td[0] /= tb[0]
    for i in range(1,m-1):
        t0 = tb[i]+ta[i-1]*l
        ta[i] /= t0
        td[i] += l*td[i-1]
        td[i] /= t0
    td[-1] = (td[-1]+l*td[-2])/(tb[-1]+ta[-1]*l)
    for i in range(2,m+1):
        td[-i] -= ta[-i+1]*td[-i+1]
    y = td
    # >>>>> end solving    
    r = concatenate((c-y,-penlambda*diff(y)))                   # c-dot(y,A)
    r[r==0] = 0.001
    u = r*(r>0)
    v = u-r
    gap = dot(c,a[:m]) - dot(y,b) + dot(v,w)
    
    # ...Start iterations
    it = 0
    while gap>ebs and it<maxiter:
        it  += 1
        invs = 1.0/s
        inva = 1.0/a
        
        # compute affine step
        q = 1.0/(u*inva + v*invs)
        # as the following computation involves lambda*diff(A)*q, we pre-multiply
        # lambda by q[m:] here directly
        q[m:] *= penlambda
        r = (u-v)*q
        # >>>>> solving tridiagonal linear equation system
        # Q = spdiags(sqrt(q),0,n,n); AQ = A*Q; rhs = Q*r'; dy = (AQ'\rhs)';
        # ..tridiagonal matrix vectors and y
        tq2= penlambda*q[m:]
        ta = -1.0*tq2                                         # -lambda^2*q[m:]
        tb = q[:m]*1
        tb[0]   += tq2[0]                                       # q[0]+lambda^2*q[m] for b[0]
        tb[1:-1]+= (tq2[:-1]+tq2[1:])                        # q[i]+lambda^2*(q[m+i-1]+q[m+i]) for b[i]
        tb[-1]  += tq2[-1]                                     # q[m]+lambda^2*q[2m-1] for b[m]
        
        td       = r[:m]*1
        td[0]   -= r[m]
        td[1:-1]+= (r[m:-1]-r[m+1:])
        td[-1]  += r[-1]
        # ..compute dy
        ta[0] /= tb[0]
        td[0] /= tb[0]
        for i in range(1,m-1):
            t[i-1] = tb[i]+ta[i-1]*tq2[i-1]
            ta[i] /= t[i-1]
            td[i] += tq2[i-1]*td[i-1]
            td[i] /= t[i-1]
        t[-1] = tb[-1]+ta[-1]*tq2[-1]
        td[-1]= (td[-1]+tq2[-1]*td[-2])/t[-1]
        
        for i in range(2,m+1):
            td[-i] -= ta[-i+1]*td[-i+1]
        dy = td
        # >>>>> solving ended
        da = concatenate((q[:m]*dy-r[:m],q[m:]*diff(dy)-r[m:]/penlambda))    # dot(dy,A)
        ds = -da
        du = -u*(1.0+da*inva)
        dv = -v*(1.0+ds*invs)
        # ...Compute maximum allowable step lengths
        # Here we omit to construct additional function "round" used in the original
        # code to set boundaries for Newton step. Instead, find the minimum of
        # the ratio and 1. Since the 1e20*p is significantly larger than 1, this
        # can be deleted during minimum operation. However, to avoid the error
        # raised due to the empty vector, try-except structure is used.
        try:
            fp = min(p*amin(-a[da<0]/da[da<0]),
                     p*amin(-s[ds<0]/ds[ds<0]),
                     1.0)
        except ValueError:
            fp = 1.0
        
        try:
            fd = min(p*amin(-u[du<0]/du[du<0]),
                     p*amin(-v[dv<0]/dv[dv<0]),
                     1)
        except ValueError:
            fd = 1.0

        # If full step is feasible, take it. Otherwise modify it using Mehrotra
        # corrector.
        if min(fp,fd) < 1:
            
            # Update mu
            mu  = dot(u,a) + dot(v,s)
            g   = dot((u+fd*du),(a+fp*da)) + dot((v+fd*dv),(s+fp*ds))
            mu *= (g/mu)**3/col
            
            # Compute modified step
            dadu= da*du
            dsdv= ds*dv
            xi  = (mu * (inva - invs))
            r  += (dadu - dsdv - xi) * q
            # >>>>> solving tridiagonal linear equation system
            td       = r[:m]*1.0
            td[0]   -= r[m]
            td[1:-1]+= (r[m:-1]-r[m+1:])
            td[-1]  += r[-1]
            
            # ..compute dy
            td[0] /= tb[0]
            for i in range(1,m):
                td[i] += tq2[i-1]*td[i-1]
                td[i] /= t[i-1]
            
            for i in range(2,m+1):
                td[-i]-= ta[-i+1]*td[-i+1]
            dy = td
            # >>>>> solving ended
            da = concatenate((q[:m]*dy-r[:m],q[m:]*diff(dy)-r[m:]/penlambda))  # dot(A,dy)
            ds = -da
            du = mu*inva - u - u*inva*da - dadu
            dv = mu*invs - v - v*invs*ds - dsdv
            
            # Compute maximum allowable step lengths
            # print('a=',a,'\n','dy=',dy,'\n','s=',s,'\n','ds=',ds,'\n')
            try:
                fp = min(p*amin(-a[da<0]/da[da<0]),
                         p*amin(-s[ds<0]/ds[ds<0]),
                         1.0)
            except ValueError:
                fp = 1.0
            try:
                fd = min(p*amin(-u[du<0]/du[du<0]),
                         p*amin(-v[dv<0]/dv[dv<0]),
                         1.0)
            except ValueError:
                fd = 1.0

        # Take the step
        a += fp*da
        s += fp*ds
        y += fd*dy
        v += fd*dv
        u += fd*du
        gap = dot(c,a[:m]) - dot(y,b) + dot(v,w)
        
    return -y


def _sirqr(x,
          penval = 1.25,
          u      = 0.03,
          wlow   = 1e-10,
          d      = 5e-5,
          maxiter= 4):
    """
    Main function of SirQR
    Inputs:
        X:          Raw data with rows are chromatograms and columns are wavelengths;
                    Note:   To simplify the computation and matrix operation, we
                            used X.T here, thus each chromatogram can be extracted
                            simply as X[i], being a r by 0 array. This can also
                            speed up the computation in quantile regression by
                            simple element indexing and dot operation as quantile
                            regression processes a chromatogram in a single run.
        penlambda:  Penalty parameter, default is 1.25;
        u:          Quantile used for quantile regression, default is 0.03;
        wlow:       Penalty value for possible peak signals to ignore the processing
                    of them in next iteration, default is 1e-10;
        d:          Parameter for a better fitting to the original signal dataset,
                    default is 5e-5;
        maxiter:    Maximum iteration, default is 4;
    """
    
    r  = len(x)
    wep= 1e-4
    w  = ones(2*r-1)*wep
    x0 = x*1.0
#    ebs= 5e-005*sum(abs(x))
    m  = 10.0*max(abs(x))
    for j in range(maxiter):
        z    = lp_fnm(r,x0,u,w,penval)
        # If maximum value after correction is larger than maximum of original
        # by 100, break the iteration to avoid value overflow
        if max(abs(z))>=m:
            break
        dx   = x-z
        dxlow= dx<d
        ds   = abs(sum(dx[dxlow]))
#        if ds<=ebs:                       # If criterion reached, stop iteration
#            break
        w[dx>d] = wlow
        w[dxlow]= exp((j+1)*abs(dx[dxlow])/ds)
        x0      = z
        
    return dx.T
    

def poolsirqr(X):
    """
    Pool version of Sirqr.
    """
    
    if len(X.shape) == 1:
        Y = _sirqr(X)
        return Y
    else:
        c   = X.shape[1]
        ncpu= cpu_count()
        if ncpu >= 6:
            pool = Pool(processes=6)
        elif ncpu <= 2:
            Y = list(map(_sirqr,[X[:,i] for i in range(c)]))
            Y = array(Y).T
            return Y
        else:
            pool = Pool(processes=ncpu)
        
        Y = pool.map(_sirqr,[X[:,i] for i in range(c)])
        Y = array(Y).T
        pool.terminate()
        return Y
#if __name__ == "__main__":
#    import scipy.io as scio
#    d = scio.loadmat("F:\\Analysis\\Python\\fpgui\\testdata.mat")["data"]
#    b = _sirqr(d[:,0][:,None])