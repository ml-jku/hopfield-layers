import numpy as np
from itertools import product
from tqdm import tqdm

class ClassicalHopfield:
    """class providing the classical hopfield model which can learn patterns and retrieve them

    Attributes
    ----------
    size : int
        dimension d of single pattern vector
    W : np.ndarray, dxd
        weight matrix of the hopfield network
    b : np.ndarray, dx1
        bias vector, in our case just set to 0
    num_pat : int
        number of learned patterns
    
    Methods
    -------
    learn(patterns) : None
        learn given list of patterns
    retrieve(test_pat, reps=1) : np.ndarray
        retrieves a memorized pattern from the provided one
    energy(pattern) : float
        calculates energy of the given pattern for the current state of the Hopfield network
    energy_landscape() : None
        prints out energy for all possible input vectors {-1,1}^d
    """

    def __init__(self, pat_size):
        """Constructor of classical Hopfield network

        Args:
            pat_size (int): the dimension d of a single pattern vector
        """
        self.size = pat_size
        self.W = np.zeros((self.size,self.size))
        self.b = np.zeros((self.size,1))

    def learn(self, patterns):
        """learns the weight matrix by applying hebbian learning rule 
        to the provided patterns, without self reference


        Args:
            patterns ([np.ndarray]):  list of dx1 numpy arrays, column-wise
        """
        self.num_pat = len(patterns)
        assert all(type(x) is np.ndarray for x in patterns), 'not all input patterns are numpy arrays'
        assert all(len(x.shape) == 2 for x in patterns), 'not all input patterns have dimension 2'
        assert all(1 == x.shape[1] for x in patterns) , 'not all input patterns have shape (%d,1)'%self.size
        
        n = len(patterns)
        W = np.zeros(self.W.shape)
        
        for pat in tqdm(patterns):
            W += np.outer(pat, pat.T)# w_ij += pat_i * pat_j
                
        W /= n
        np.fill_diagonal(W,0)
        self.W = W
        return

    def retrieve(self, test_pat, reps=1):
        """[summary]

        Args:
            test_pat (np.ndarray): the partial/masked test pattern which should be retrieved
            reps (int, optional): number of times the retrieval update should be applied . Defaults to 1.

        Returns:
            np.ndarray: the retrieved pattern of shape dx1
        """
        assert type(test_pat) is np.ndarray, 'input pattern is not a numpy array'
        assert test_pat.shape == (self.size, 1), 'input pattern has wrong shape %s' % test_pat.shape

        for idx_rep in tqdm(range(reps)):
            # synchronous update, aka use old state for all updates 
            reconst = self.W@test_pat # reconst_i = \sum_j w_ij * partial_pattern_j
            test_pat = np.where(reconst>self.b, 1, -1)

        return test_pat
    
    def energy(self, pattern):
        """calculates energy for a pattern according to hopfield model

        Args:
            pattern (np.ndarray): the pattern for which the energy should be calculated

        Returns:
            float: the calculated energy
        """
        assert type(pattern) is np.ndarray
        return -0.5 * pattern.T @ self.W @ pattern + self.b.T @pattern 

    def energy_landscape(self):
        """print out all vectors of the input space with their energy
        dont use this on larger input space!
        """
        for pat in product([1,-1], repeat=self.size):
            print("energy(%r)=%.3f"%(pat, self.energy(np.array(pat))))
        return

class DenseHopfield:
    def __init__(self, pat_size, beta=1, normalization_option=1):
        self.size = pat_size
        self.beta = beta
        self.max_norm = np.sqrt(self.size)
        if normalization_option == 0:
            self.energy = self.energy_unnormalized
        elif normalization_option == 1: # normalize dot product of patterns by 1/sqrt(pattern_size)
            self.energy = self.energy_normalized
        elif normalization_option == 2: # normalize dot product of patterns by shifting and clamping low exponentials
            self.energy = self.energy_normalized2
        else:
            raise ValueError('unkown option for normalization: %d'% normalization_option)

        return

    def learn(self, patterns):
        """expects patterns as numpy arrays and stores them col-wise in pattern matrix 
        """
        self.num_pat = len(patterns)
        assert(all(type(x) is np.ndarray for x in patterns)), 'not all input patterns are numpy arrays'
        assert(all(len(x.shape) == 2 for x in patterns)), 'not all input patterns have dimension 2'
        assert(all(1 == x.shape[1] for x in patterns)), 'not all input patterns have shape (-1,1) '
        self.patterns = np.array(patterns).squeeze(axis=-1).T # save patterns col-wise
        # without squeeze axis would result in problem with one pattern
        self.max_pat_norm = max(np.linalg.norm(x) for x in patterns)

    def retrieve(self, partial_pattern, max_iter=np.inf, thresh=0.5):
        # partial patterns have to be provided with None/0 at empty spots
        if partial_pattern.size != self.size:
            raise ValueError("Input pattern %r does not match state size: %d vs %d" 
                %(partial_pattern, len(partial_pattern), self.size))
        
        if None in partial_pattern:
            raise NotImplementedError("How should we deal with empty spots? Fill randomly or set to 0?")
            # fill with 0 seems to make most sense

        assert type(partial_pattern) == np.ndarray, 'test pattern was no numpy array'
        assert len(partial_pattern.shape) <=2 and 1 == partial_pattern.shape[1], 'test pattern with shape %r is not a col-vector' %(partial_pattern.shape,)

        pat_old = partial_pattern.copy()
        iters = 0

        for iters in tqdm(range(max_iter)):
            pat_new = np.zeros(partial_pattern.shape)
            # jj = np.random.randint(self.size)
            for jj in range(self.size):
                # simple variant:
                E = 0
                temp = pat_old[jj].copy()
                pat_old[jj] = +1
                E -= self.energy(pat_old)
                pat_old[jj] = -1
                E += self.energy(pat_old)

                pat_old[jj] = temp
                pat_new[jj] = np.where(E >0 , 1, -1)
            
            if np.count_nonzero(pat_old != pat_new)<= thresh:
                break
            else:
                pat_old = pat_new
            
        return pat_new

    @staticmethod
    def _lse(z, beta):
        return 1/beta * np.log(np.sum(np.exp(beta*z)))

    def energy_unnormalized(self, pattern):
        # return -1*np.exp(self._lse(self.patterns.T @pattern, beta=self.beta))
        # this is equal, but faster
        return -1*np.sum(np.exp(self.patterns.T @pattern ))
    
    def energy_normalized(self, pattern):
        # normalize dot product of patterns by 1/sqrt(pattern_size)
        return -1*np.sum(np.exp((self.patterns.T @pattern)/self.max_norm ))
    
    def energy_normalized2(self, pattern):
        # normalize dot product of patterns by shifting by -sqrt(pattern_size)
        # also clamp exponential for exponents smaller then -73 to 0 
        exponents = self.patterns.T @pattern
        norm_exponents = exponents - self.max_pat_norm
        norm_exponents[norm_exponents < -73] = -np.inf

        return -1*np.sum(np.exp(norm_exponents))

    
    def energy_landscape(self):
        for pat in product([1,-1], repeat=self.size):
            pat = np.array(pat)
            print("energy(%r)=%.3f"%(pat, self.energy(pat)))


class ContinuousHopfield:
    def __init__(self, pat_size, beta=1, do_normalization=True):
        self.size = pat_size # size of individual pattern
        self.beta = beta
        self.max_norm = np.sqrt(self.size)
        if do_normalization:
            self.softmax = self.softmax_normalized
            self.energy = self.energy_normalized
        else:
            self.softmax = self.softmax_unnormalized
            self.energy = self.energy_unnormalized
        
        return

    def learn(self, patterns):
        """expects patterns as numpy arrays and stores them col-wise in pattern matrix 
        """
        self.num_pat = len(patterns)
        assert(all(type(x) is np.ndarray for x in patterns)), 'not all input patterns are numpy arrays'
        assert(all(len(x.shape) == 2 for x in patterns)), 'not all input patterns have dimension 2'
        assert(all(1 == x.shape[1] for x in patterns)), 'not all input patterns have shape (-1,1) '
        self.patterns = np.array(patterns).squeeze(axis=-1).T # save patterns col-wise
        # without squeeze axis would result in problem with one pattern
        # return -1*np.sum(np.exp([(self.patterns[:,ii].T @pattern)/self.max_norm for ii in range(self.patterns.shape[1])]))
        self.M = max(np.linalg.norm(vec) for vec in patterns)# maximal norm of actually stored patterns
        return

    def retrieve(self, partial_pattern, max_iter=np.inf, thresh=0.5):
        # partial patterns have to be provided with None/0 at empty spots
        if partial_pattern.size != self.size:
            raise ValueError("Input pattern %r does not match state size: %d vs %d" 
                %(partial_pattern, len(partial_pattern), self.size))
        
        if None in partial_pattern:
            raise NotImplementedError("How should we deal with empty spots? Fill randomly or set to 0?")
            # fill with 0 seems to make most sense

        assert type(partial_pattern) == np.ndarray, 'test pattern was no numpy array'
        assert len(partial_pattern.shape) ==2 and 1 == partial_pattern.shape[1], 'test pattern with shape %r is not a col-vector' %(partial_pattern.shape,)

        pat_old = partial_pattern.copy()
        iters = 0

        while(iters < max_iter):
            pat_new = self.patterns @ self.softmax(self.beta*self.patterns.T @ pat_old)

            if np.count_nonzero(pat_old != pat_new)<= thresh: # converged
                break
            else:
                pat_old = pat_new
            iters += 1
            
        return pat_new

    @staticmethod
    def softmax_unnormalized(z):
        numerators = np.exp(z) # top
        denominator = np.sum(numerators) # bottom
        return numerators/denominator

    def softmax_normalized(self,z):
        numerators = np.exp(z/self.max_norm) # top
        denominator = np.sum(numerators) # bottom
        return numerators/denominator

    @staticmethod
    def _lse(z, beta):
        return 1/beta * np.log(np.sum(np.exp(beta*z)))

    def energy_unnormalized(self, pattern):
        return -1*self._lse(self.patterns.T @pattern, 1) + 0.5 * pattern.T @ pattern\
            + 1/self.beta*np.log(self.num_pat)\
            + 0.5*self.M**2
    
    def energy_normalized(self, pattern):
        # normalize dot product of patterns by 1/sqrt(pattern_size)
        return -1*self._lse((self.patterns.T @pattern )/self.max_norm, 1) + 0.5 * pattern.T @ pattern\
            + 1/self.beta*np.log(self.num_pat)\
            + 0.5*self.M**2

    def energy_landscape(self):
        for pat in product([1,-1], repeat=self.size):
            pat = np.array(pat)
            print("energy(%r)=%.3f"%(pat, self.energy(pat)))


if __name__ == "__main__":
    # performs small test of Hopfield classes
    patterns = [
                (1,1,1),
                (-1,1,-1),
                (-1,-1,-1)
                ]
    print('input patterns:')
    print(patterns)
    patterns = [np.array(x).reshape(-1,1) for x in patterns]

    d = len(patterns[0])
    for net in [ClassicalHopfield(d), DenseHopfield(d), ContinuousHopfield(d)]:
        net = DenseHopfield(len(patterns[0]))

        # train
        net.learn(patterns)

        # recall
        test_pattern = [(1,1,1), (-1,1,-1), (1,-1,1), (1,1,-1), (-1,1,1), (-1,-1,-1)]
        test_pattern = [np.array(x).reshape(-1,1) for x in test_pattern]

        print("recalls")
        for tpat in test_pattern:
            rec = net.retrieve(tpat, max_iter=5)
            print("test=%r, recall=%r, energy=%.3f "%(tpat.squeeze().tolist(),rec.squeeze().tolist(), net.energy(rec)))


        print("energy landscape:")
        net.energy_landscape()