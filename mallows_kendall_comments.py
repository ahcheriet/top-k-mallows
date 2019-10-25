import numpy as np
import itertools as it
import scipy as sp
#fit MM

# estas dos funciones valen para buscar los thetas para diferentes E[d].
# desde 0<E[d]<1 (theta alto) hasta E_0[d] (theta 0)
def find_phi_n(n, bins): #NO
    ed, phi_ed = [], []
    #expected dist en la uniforma para este n
    ed_uniform = (n*(n-1)/2)/2 # dist max div 2. AKA
    # ed_uniform = np.mean([expected_dist_MM(n,-0.01), expected_dist_MM(n,0.01)])
    # if k is not None : ed_uniform = np.mean([expected_V(n,theta=None, phi=0.99999,k=k).sum(), expected_V(n,theta=None, phi=1.00000001,k=k).sum()])
    # print(ed_uniform, "find")
    for dmin in np.linspace(0,ed_uniform-1,bins):
        ed.append(dmin)
        phi_ed.append(find_phi(n, dmin, dmin+1))
    return ed, phi_ed
def find_phi(n, dmin, dmax): #NO
    imin, imax = np.float64(0),np.float64(1)
    iterat = 0
    while iterat < 500:
        med = imin + (imax-imin)/2
        d = expected_dist_MM(n,phi2theta(med))#mk.phi2theta(med)
        #print(imin, imax, med, d,imin==imax)
        if d < dmax and d > dmin: return med
        elif d < dmin : imin = med
        elif d > dmax : imax = med
        iterat  += 1

def max_dist(n):
    """
        Parameters
        ----------
        n: int
            length of permutations

        Returns
        -------
        int
            Maximum distance between permutations of given n length
    """
    return int(n*(n-1)/2)

def compose(s,p):
    """This function composes two given permutations

    Parameters
    ----------
    s: ndarray
        The first permutation array
    p: ndarray
        The second permutation array

    Returns
    -------
    ndarray
        The composition of the permutations

    """
    return np.array(s[p])
def compose_partial(partial,full):#s is partial
                                  # Test in the list building always true if full not partial ???
                                  # [partial[i] if not np.isnan(partial[i]) else np.nan for i in full] ???
    """This function composes a partial permutation with an other (full)

        Parameters
        ----------
        partial: ndarray
            The partial permutation (should be filled with float)
        full:
            The full permutation (should be filled with integers)

        Returns
        -------
        ndarray
            The composition of the permutations
    """
    return [partial[i] if not np.isnan(i) else np.nan for i in full]
def inverse_partial(sigma):
    """This function computes the inverse of a given partial permutation

        Parameters
        ----------
        sigma: ndarray
            A partial permutation array (filled with float)

        Returns
        -------
        ndarray
            The inverse of given partial permutation
    """
    inv = np.array([np.nan]*len(sigma))
    for i,j in enumerate(sigma):
        if not np.isnan(j):
            inv[int(j)] = i
    return inv
def inverse(s):
    """This function computes the inverse of a given permutation

        Parameters
        ----------
        s: ndarray
            A permutation array

        Returns
        -------
        ndarray
            The inverse of given permutation
    """
    return np.argsort(s)

def alpha2beta(alpha,k):
    """Inverse a partial ordering

        Parameters
        ----------
        alpha: ndarray
            A permutation
        k: int
            The index to which the permutation inverse must be given

        Returns
        -------
        ndarray
            The partial ordering inverse also called partial ranking
    """
    inv = np.array([np.nan]*len(alpha))
    for i,j in enumerate(alpha[:k]):
        #print(i, j)
        inv[int(j)] = i
        #print(inv)
    return inv


def borda(rankings):
    """This function computes an average permutation given several permutations

        Parameters
        ----------
        rankings: ndarray
            Matrix of several permutations

        Returns
        -------
        ndarray
            The 'average' permutation of permutations given
    """
    consensus =  np.argsort( # give the inverse of result --> sigma_0
                            np.argsort( # give the indexes to sort the sum vector --> sigma_0^-1
                                        rankings.sum(axis=0) # sum the indexes of all permutations
                                        )
                            ) #borda
    return consensus

def borda_partial(rankings):
    """

        Parameters
        ----------

        Returns
        -------

    """
    borda = np.argsort(np.argsort(np.nanmean(rankings, axis=0))).astype(float)
    mask = np.isnan(rankings).all(axis=0)
    borda[mask]=np.nan
    return borda


def check_theta_phi(theta, phi):
    """This function automatically converts theta to phi or phi to theta as
    list or float depending on the values and value types given as input

        Parameters
        ----------
        theta: float or list
            Dispersion parameter theta to convert to phi (can be None)
        phi: float or list
            Dispersion parameter phi to convert to theta (can be None)

        Returns
        -------
        tuple
            tuple containing both theta and phi (of list or float type depending on the input type)
    """
    if not ((phi is None) ^ (theta is None)):
        print("KAKA, pon valores")
    if phi is None and type(theta)!=list:
        phi = theta2phi(theta)
    if theta is None and type(phi)!=list:
        theta = phi2theta(phi)
    if phi is None and type(theta)==list:
        phi = [theta2phi(t) for t in theta]
    if theta is None and type(phi)==list:
        theta = [phi2theta(p) for p in phi]
    return theta, phi

def expected_dist_MM(n,theta=None, phi=None):
    """Compute the expected distance, MM under the Kendall's-tau distance

        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Real dispersion parameter (optionnal if phi is given)
        phi: float
            Real dispersion parameter (optionnal if theta is given)

        Returns
        -------
        float
            The expected disance under the MMs
    """
    theta, phi = check_theta_phi(theta, phi)
    expected_dist = n * np.exp(-theta) / (1-np.exp(-theta)) - np.sum([j * np.exp(-j*theta) / (1 - np.exp(-j*theta))  for j in range(1,n+1)])
    return expected_dist
def variance_dist_MM(n,theta=None, phi=None):
    """

        Parameters
        ----------

        Returns
        -------

    """
    theta, phi = check_theta_phi(theta, phi)
    variance = (phi*n)/(1-phi)**2 - np.sum([(pow(phi,i) * i**2)/(1-pow(phi,i))**2  for i in range(1,n+1)])
    return variance
def expected_V(n,theta=None, phi=None,k=None):#txapu integrar
    """This function computes the expected decomposition vector

        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Real dispersion parameter (optionnal if phi is given)
        phi: float
            Real dispersion parameter (optionnal if theta is given)
        k: int
            Index to which the decomposition vector is needed ???

        Returns
        -------
        ndarray
            The expected decomposition vector
    """
    theta, phi = check_theta_phi(theta, phi)
    if k is None: k = n-1
    if type(theta)!=list: theta = [theta]*k
    expected_v = [np.exp(-theta[j]) / (1-np.exp(-theta[j])) - (n-j) * np.exp(-(n-j)*theta[j]) / (1 - np.exp(-(n-j)*theta[j]))  for j in range(k)]
    return np.array(expected_v)
def variance_V(n,theta=None, phi=None,k=None):#txapu integrar es posibe q solo fuciones con MM
    """

        Parameters
        ----------

        Returns
        -------

    """
    theta, phi = check_theta_phi(theta, phi)
    if k is None: k = n-1
    if type(phi)!=list: phi = [phi]*k
    var_v = [phi[j]/(1-phi[j])**2 - (n-j)**2 * phi[j]**(n-j) / (1-phi[j]**(n-j))**2 for j in range(k)]
    return np.array(var_v)

def psiMM(n,theta=None,phi=None):
    """This function computes the normalization constant psi

        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Real dispersion parameter (optionnal if phi is given)
        phi: float
            Real dispersion parameter (optionnal if theta is given)

        Returns
        -------
        float
            The normalization constant psi
    """
    if theta is not None: return np.prod([(1-np.exp(-theta*j))/(1-np.exp(-theta)) for j in range(2,n+1)])
    if phi is not None:  return np.prod([(1-np.power(phi,j))/(1-phi) for j in range(2,n+1)])
    theta, phi = check_theta_phi(theta, phi) #por si acaso
    #np.array([(1 - np.exp(( - n + i )*(theta)))/(1 - np.exp( -theta)) for i in range(n-1)])

def prob_mode(n, theta):
    """This function computes the probability mode

        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Real dispersion parameter

        Returns
        -------
        float
            The probability mode

    """
    #theta as array
    psi = np.array([(1 - np.exp(( - n + i )*(theta[i])))/(1 - np.exp( -theta[i])) for i in range(n-1)])
    return np.prod(1.0/psi)

def prob(n, theta, dist):
    """This function computes the probability of a permutation given a distance to the consensus

        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Dispersion vector
        dist: int
            Distance of the permutation to the consensus permutation

        Returns
        -------
        float
            Probability of the permutation

    """
    psi = np.array([(1 - np.exp(( - n + i )*(theta)))/(1 - np.exp( -theta)) for i in range(n-1)])
    psi = np.prod(psi)
    return np.exp(-theta*dist) / psi

def prob_sample(perms,sigma,theta=None,phi=None):
    """This function computes the probabilities for each permutation of a sample
    of several permutations

        Parameters
        ----------
        perms: ndarray
            The matrix of permutations
        sigma: ndarray
            Permutation mode
        theta: float
            Real dispersion parameter (optionnal if phi is given)
        phi: float
            Real dispersion parameter (optionnal if theta is given)

        Returns
        -------
        ndarray
            Array of probabilities for each permutation given as input

    """
    m,n = perms.shape
    theta, phi = check_theta_phi(theta, phi)
    psi = np.array([(1 - np.exp(( - n + i )*(theta)))/(1 - np.exp( -theta)) for i in range(n-1)])
    psi = np.prod(psi)
    return np.array([np.exp(-theta*kendallTau(perm, sigma)) / psi for perm in perms])

def fit_MM(rankings, s0=None): #returns sigma, phi
    """This function computes the consensus permutation and the MLE for the
    dispersion parameter phi for MM models

        Parameters
        ----------
        rankings: ndarray
            The matrix of permutations
        s0: ndarray, optional
            The consensus permutation (default value is None)

        Returns
        -------
        tuple
            The ndarray corresponding to s0 the consensus permutation and the
            MLE for the dispersion parameter phi

    """
    m , n = rankings.shape
    if s0 is None: s0 = np.argsort(np.argsort(rankings.sum(axis=0))) #borda
    dist_avg = np.mean(np.array([kendallTau(s0, perm) for perm in rankings]))
    try:
        theta = sp.optimize.newton(mle_theta_mm_f, 0.01, fprime=mle_theta_mm_fdev, args=(n, dist_avg), tol=1.48e-08, maxiter=500, fprime2=None)
    except:
        if dist_avg == 0.0: return s0, np.exp(-5)#=phi
        print("error. fit_mm. dist_avg=",dist_avg, dist_avg == 0.0)
        print(rankings)
        print(s0)
        raise
    # theta = - np.log(phi)
    return s0, np.exp(-theta)#=phi

def fit_MM_phi(n, dist_avg): #returns sigma, phi
    """Same as fit_MM but just returns phi ??? Also does not compute dist_avg
    but take it as a parameter

        Parameters
        ----------
        n: int
            Dimension of the permutations
        dist_avg: float
            Average distance of the sample (between the consensus and the
            permutations of the consensus)

        Returns
        -------
        float
            The MLE for the dispersion parameter phi
    """
    try:
        theta = sp.optimize.newton(mle_theta_mm_f, 0.01, fprime=mle_theta_mm_fdev, args=(n, dist_avg), tol=1.48e-08, maxiter=500, fprime2=None)
    except:
        if dist_avg == 0.0: return s0, np.exp(-5)#=phi
        print("error. fit_mm. dist_avg=",dist_avg, dist_avg == 0.0)
        print(rankings)
        print(s0)
        raise
    # theta = - np.log(phi)
    return np.exp(-theta)

def fit_GMM(rankings, s0=None):
    """This function computes the consensus permutation and the MLE for the
    dispersion parameters theta_j for GMM models

        Parameters
        ----------
        rankings: ndarray
            The matrix of permutations
        s0: ndarray, optional
            The consensus permutation (default value is None)

        Returns
        -------
        tuple
            The ndarray corresponding to s0 the consensus permutation and the
            MLE for the dispersion parameters theta

    """
    m , n = rankings.shape
    if s0 is None: s0 = np.argsort(np.argsort(rankings.sum(axis=0))) #borda
    V_avg = v_avg(rankings)
    try:
        theta = []
        for j in range(1,n):
            theta_j = sp.optimize.newton(mle_theta_j_gmm_f, 0.01, fprime=mle_theta_j_gmm_fdev, args=(n, j, V_avg[j-1]), tol=1.48e-08, maxiter=500, fprime2=None)
            theta.append(theta_j)
    except:
        print("Error no convergence ?")
        #if dist_avg == 0.0: return s0, np.exp(-5)#=phi
        # print("error. fit_mm. dist_avg=",dist_avg, dist_avg == 0.0)
        # print(rankings)
        # print(s0)
        raise
    #theta = - np.log(phi)
    return s0, theta




def theta2phi(theta):
    """This functions converts theta dispersion parameter into phi

        Parameters
        ----------
        theta: float
            Real dispersion parameter

        Returns
        -------
        float
            phi real dispersion parameter
    """
    return np.exp(-theta)
def phi2theta(phi):
    """This functions converts phi dispersion parameter into theta

        Parameters
        ----------
        phi: float
            Real dispersion parameter

        Returns
        -------
        float
            theta real dispersion parameter
    """
    return - np.log(phi)

def mle_theta_mm_f(theta, n, dist_avg):
    """Computes the function that equals zero for the MLE for the dispersion
    parameter

        Parameters
        ----------
        theta: float
            The dispersion parameter
        n: int
            Dimension of the permutations
        dist_avg: float
            Average distance of the sample (between the consensus and the
            permutations of the consensus)

        Returns
        -------
        float
            Value of the function for given parameters

    """
    aux = 0
    for j in range(1,n):
        k = n - j + 1
        aux += (k * np.exp(-theta * k))/(1 - np.exp(-theta * k))
    aux2 = (n-1) / (np.exp( theta ) - 1) - dist_avg
    return aux2 - aux

def mle_theta_mm_fdev(theta, n, dist_avg):
    """This function computes the derivative of the function mle_theta_mm_f
    given the dispersion parameter and the average distance

        Parameters
        ----------
        theta: float
            The dispersion parameter
        n: int
            The dimension of the permutations
        dist_avg: float
            Average distance of the sample (between the consensus and the
            permutations of the consensus)

        Returns
        -------
        float
            The value of the derivative of function mle_theta_mm_f for given
            parameters
    """
    aux = 0
    for j in range(1,n):
        k = n - j + 1
        aux += (k * k * np.exp( -theta * k ))/pow((1 - np.exp(-theta * k)) , 2 )
    aux2 = (- n + 1) * np.exp( theta ) / pow ((np.exp( theta ) - 1) , 2 )
    # print(theta)
    return aux2 + aux

def mle_theta_j_gmm_f(theta_j, n, j, v_j_avg):
    """Computes the function that equals zero for the MLE for the dispersion
    parameter theta_j in the GMM

        Parameters
        ----------
        theta: float
            The jth dispersion parameter theta_j
        n: int
            Dimension of the permutations
        j: int
            The position of the theta_j in vector theta of dispersion parameters
        v_j_avg: float
            jth element of the average decomposition vector over the sample

        Returns
        -------
        float
            Value of the function for given parameters

    """
    f_1 = np.exp( -theta_j ) / ( 1 - np.exp( -theta_j ) )
    f_2 = - ( n - j + 1 ) * np.exp( - theta_j * ( n - j + 1 ) ) / ( 1 - np.exp( - theta_j * ( n - j + 1 ) ) )

    return f_1 + f_2 - v_j_avg

def mle_theta_j_gmm_fdev(theta_j, n, j, v_j_avg):
    """This function computes the derivative of the function mle_theta_j_gmm_f
    given the jth element of the dispersion parameter and the jth element of the
    average decomposition vector

        Parameters
        ----------
        theta: float
            The jth dispersion parameter theta_j
        n: int
            Dimension of the permutations
        j: int
            The position of the theta_j in vector theta of dispersion parameters
        v_j_avg: float
            jth element of the average decomposition vector over the sample

        Returns
        -------
        float
            The value of the derivative of function mle_theta_j_gmm_f for given
            parameters
    """
    fdev_1 = - np.exp( - theta_j ) / pow( ( 1 - np.exp( -theta_j ) ), 2 )
    fdev_2 = pow( n - j + 1, 2 ) * np.exp( - theta_j * ( n - j + 1 ) ) / pow( 1 - np.exp( - theta_j * ( n - j + 1 ) ), 2 )
    return fdev_1 + fdev_2

def likelihood_mm(perms, s0, theta):
    """This function computes the log-likelihood for MM model given a matrix of
    permutation, the consensus permutation, and the dispersion parameter

        Parameters
        ----------
        perms: ndarray
            A matrix of permutations
        s0: ndarray
            The consensus permutation
        theta: float
            The dispersion parameter

        Returns
        -------
        float
            Value of log-likelihood for given parameters
    """
    m,n = perms.shape
    psi = 1.0 / np.prod([(1-np.exp(-theta*j))/(1-np.exp(-theta)) for j in range(2,n+1)])
    probs = np.array([np.log(np.exp(-kendallTau(s0, perm)*theta)/psi) for perm in perms])
    # print(probs,m,n)
    return probs.sum()

def samplingMM(m,n,theta=None, phi=None, k=None):
    """This function generates m permutations (rankings) according
    to Mallows Models given a parameter of dispersion (theta or phi).
    It applies the samplingGMM function with dispersion vector filled with
    theta

        Parameters
        ----------
        m: int
            The number of rankings to generate
        n: int
            The dimension of the permutations
        theta: float, optional (if phi given)
            The dispersion parameter theta
        phi: float, optional (if theta given)
            The dispersion parameter phi
        k: int, optional
            ???

        Returns
        -------
        list
            The rankings generated
    """
    # k return partial orderings
    theta, phi = check_theta_phi(theta, phi)
    if k==n:k=None
    return samplingGMM(m,[theta]*(n-1),topk=k)

def samplingGMM(m,theta, topk=None):
    """This function generates m permutations (rankings) according
    to Generalized Mallows Models given a vector of dispersion (theta). It
    first generates the decomposition vectors and computes then the
    corresponding permutation using v2ranking function
    topk ???

        Parameters
        ----------
        m: int
            The number of rankings to generate
        theta: ndarray
            The dispersion vector for GMM
        topk: int
            ???

        Returns
        -------
        list
            The rankings generated

    """
    #  returns RANKINGS!!!!!!!*****
    n = len(theta)+1
    if topk is None or topk == n: k = n-1
    else: k = topk
    psi = [(1 - np.exp(( - n + i )*(theta[ i ])))/(1 - np.exp( -theta[i])) for i in range(k)]
    vprobs = np.zeros((n,n))
    for j in range(k): #range(n-1):
        vprobs[j][0] = 1.0/psi[j]
        for r in range(1,n-j):
            vprobs[j][r] = np.exp( -theta[j] * r ) / psi[j]#vprobs[j][ r - 1 ] + np.exp( -theta[j] * r ) / psi[j]
    sample = []
    vs = []
    for samp in range(m):
        v = [np.random.choice(n,p=vprobs[i,:]) for i in range(k)] # v = [np.random.choice(n,p=vprobs[i,:]/np.sum(vprobs[i,:])) for i in range(k)]
        #vs.append(v)
        #print(v, np.sum(v))
        # print(v, topk)
        if topk is None: v += [0] # la fun discordancesToPermut necesita, len(v)==n
        ranking = v2ranking(v, n)#discordancesToPermut(v,list(range(n)))
        # if topk is not None :
        #     ranking = np.concatenate([ranking, np.array([np.nan]*(n-topk))])
        sample.append(ranking)
    return sample


def v2ranking(v, n): ##len(v)==n, last item must be 0
    """This function computes the corresponding permutation given
    a decomposition vector

        Parameters
        ----------
        v: ndarray
            Decomposition vector, same length as the permutation, last item must be 0
        n: int
            Length of the permutation

        Returns
        -------
        ndarray
            The permutation corresponding to the decomposition vectors
    """
    # n = len(v)
    rem = list(range(n))
    rank = np.array([np.nan]*n)# np.zeros(n,dtype=np.int)
    # print(v,rem,rank)
    for i in range(len(v)):
        # print(i,v[i], rem)
        rank[i] = rem[v[i]]
        rem.pop(v[i])
    return rank#[i+1 for i in permut];

def ranking2v(sigma):
    """This function computes the corresponding decomposition vector given
    a permutation

        Parameters
        ----------
        sigma: ndarray
            A permutation

        Returns
        -------
        ndarray
            The decomposition vector corresponding to the permutation. Will be
            of length n and finish with 0
    """
    n = len(sigma)
    V = []
    for j, sigma_j in enumerate(sigma):
        V_j = 0
        for i in range(j+1,n):
            if sigma_j > sigma[i]:
                V_j += 1
        V.append(V_j)
    return V

def v_avg(rankings):
    """This function return the vector which is the average of all the
    decomposition vectors of a given sample of permutation. The function will
    first converts the corresponding decomposition vector of each permutation
    and will compute the average of these vectors then.

        Parameters
        ----------
        rankings: ndarray
            A sample of permutation

        Returns
        -------
        ndarray
            The average of all the decomposition vectors
    """
    m , n = rankings.shape

    V_sample = np.array([ranking2v(sigma)[:-1] for sigma in rankings]) # Removing last element since not useful

    v_avg = np.mean(V_sample, axis = 0)

    return v_avg


def discordancesToPermut(indCode, refer):
    """

        Parameters
        ----------

        Returns
        -------

    """
    print("warning. discordancesToPermut is deprecated. Use function v2ranking")
    return v2ranking(indCode)
    # returns rNKING
    # n = len(indCode)
    # rem = refer[:] #[i for i in refer]
    # ordering = np.zeros(n,dtype=np.int)
    # for i in range(n):
    #     ordering[i] = rem[indCode[i]]
    #     rem.pop(indCode[i])
    # return ordering#[i+1 for i in permut];

def kendallTau(A, B=None):
    """This function computes the kendall's-tau distance between two permutations.
    If only one permutation is given, the distance will be computed with the
    identity permutation as the second permutation

        Parameters
        ----------
        A: ndarray
            The first permutation
        B: ndarray, optionnal
            The second permutation (default is None)

        Returns
        -------
        int
            The kendall's-tau distances between both permutations
    """
    # if any partial is B
    if B is None : B = list(range(len(A)))
    n = len(A)
    pairs = it.combinations(range(n), 2)
    distance = 0
    # print("IIIIMNNMNNN",list(pairs),len(A))
    for x, y in pairs:
        #if not A[x]!=A[x] and not A[y]!=A[y]:#OJO no se check B
        a = A[x] - A[y]
        try:
            b = B[x] - B[y]# if discordant (different signs)
        except:
            print("ERROR kendallTau, check b",A, B, x, y)
        # print(b,a,b,A, B, x, y,a * b < 0)
        if (a * b < 0):
            distance += 1
    return distance


def distAlpha(alpha, k):
    """Compute the distance of a partial ordering (also called top-k list)
    according to an alternative definition of Kendall's-tau distance. The
    distance is defined as follows: it is the sum of the js in the head larger
    than i for every i.

        Parameters
        ----------
        alpha: ndarray
            The partial ordering
        k: int
            The order ??? of the partial list

        Returns
        -------
        int
            The kendall's-tau distance alternative for alpha
    """
    #an alternative def for kendall is to sum the js in the tail smaller than i, for every i
    #or the js in the head larger than i for every i*. we take this since the head is defined an d the tail is not for alpha in Alpha
    dist = 0
    for j in range(k):
        dist += alpha[j] - np.sum([1 for i in alpha[:j] if i<alpha[j]])
        #print((i) , np.sum([1 for j in alpha[:i] if j<alpha[i]]) )
    return dist

def distBeta(beta, sigma=None): #if sigma is None=> sigma=e, sigma is complete
    # OJO he cambiado la definicion, he quitado la k
    #for beta in Beta we assume that the positions not defined are >k
    """Compute the distance of a partial ranking according to an alternative
    definition of Kendall's-tau distance. The distance is defined as follows:
    missing ranks in beta are filled with a value greater than all the values
    in both rankings (length of the rankings + 1 here). Then the classical
    Kendall's-tau distance is applied to this new vector.

        Parameters
        ----------
        beta: ndarray
            The partial ranking
        sigma: ndarray, optional
            A full permutation to which wew want to compute the distance with
            beta (default None, sigma will be the identity permutation)

        Returns
        -------
        int
            The kendall's-tau distance alternative for beta
    """
    n = len(beta)
    if sigma is None: sigma = list(range(n))
    aux = beta.copy()
    aux = [i if not np.isnan(i) else n+1 for i in aux ]
    return kendallTau(aux, sigma)

def partial_ord2partial_rank(pord,n,k,type="beta"):#NO
    if type=="gamma": val = -1
    if type=="beta": val = k

    # pord is a collection of partial orderings, each of which (1) has len n (2) np.nans for the unspecified places (3) is np.array
    #input partial ordering of the first k items. The first k positions have vals [0,n-1]
    #output partial ranking of the first k ranks. There are k positions have vals [0,k-1]. The rest have val=k (so the kendall dist can be compared)
    prank = []
    # n = len(pord[0])
    # for perm in pord:
    res = np.array([val]*n)
    for i,j in enumerate(pord[~np.isnan(pord)]):
        res[int(j)]=i
    # prank.append(res)
    return np.array(res)

# m'/M segun wolfram -((j - n) e^(j x))/(e^(n x) - e^(j x)) - (j e^x - j - n e^x + n + e^x)/(e^x - 1)
#
def Ewolfram(n,j,x):#NO
    return (-((j - n) * np.exp(j * x))/(np.exp(n* x) - np.exp(j *x)) - (j* np.exp(x) - j - n *np.exp(x) + n + np.exp(x))/(np.exp(x) - 1))
#(E^(x + 2 j x) + E^(x + 2 n x) - E^((j + n) x) (j - n)^2 - E^((2 + j + n) x) (j - n)^2 + 2 E^((1 + j + n) x) (-1 + j^2 - 2 j n + n^2))/((-1 + E^x)^2 (-E^(j x) + E^(n x))^2)
def Vwolfram(n,j,x):#NO
    numer = (np.exp(x + 2* j *x) + np.exp(x + 2* n *x) - np.exp((j + n) *x)*(j - n)**2 - np.exp((2+j+n)* x)* (j - n)**2 + 2 *np.exp((1 + j + n) *x)*(-1 + j**2 - 2* j *n + n**2))
    denom = ((-1 + np.exp(x))**2 *(-np.exp(j *x) + np.exp(n* x))**2)
    return numer/denom


## number of perms at each dist
def num_perms_at_dist(n):
    """This function computes the number of permutations of length 1 to n for
    each possible Kendall's-tau distance d

        Parameters
        ----------
        n: int
            Dimension of the permutations

        Returns
        -------
        ndarray
            ??? ---> to finish
    """
    sk = np.zeros((n+1,int(n*(n-1)/2+1)))
    for i in range(n+1):
        sk[i,0] = 1
    for i in range(1,1+n):
        for j in range(1,int(i*(i-1)/2+1)):
            if j - i >= 0 :
                sk[i,j] = sk[i,j-1]+ sk[i-1,j] - sk[i-1,j-i]
            else:
                sk[i,j] = sk[i,j-1]+ sk[i-1,j]
    return sk.astype(np.uint64)
## random permutations at distance
def random_perm_at_dist(n, dist, sk):
    """

        Parameters
        ----------

        Returns
        -------

    """
    # param sk is the results of the function num_perms_at_dist(n)
    i = 0
    probs = np.zeros(n+1)
    v = np.zeros(n,dtype=int)
    while i<n and dist > 0 :
        rest_max_dist = (n - i - 1 ) * ( n - i - 2 ) / 2
        if rest_max_dist  >= dist:
            probs[0] = sk[n-i-1,dist]
        else:
            probs[0] = 0
        mi = min(dist + 1 , n - i )
        for j in range(1,mi):
            if rest_max_dist + j >= dist: probs[j] = sk[n-i-1, dist-j]
            else: probs[ j ] = 0
        v[i] = np.random.choice(mi,1,p=probs[:mi]/probs[:mi].sum())
        dist -= v[i]
        i += 1
    return v2ranking(v)











# end
