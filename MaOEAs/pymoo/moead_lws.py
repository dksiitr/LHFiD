import numpy as np
from scipy.spatial.distance import cdist
import math
import warnings
from numpy.linalg import LinAlgError
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.factory import get_decomposition
from pymoo.model.individual import Individual
from pymoo.model.population import Population,pop_from_array_or_individual
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none
from pymoo.util.normalization import normalize
from pymoo.util.function_loader import load_function


# =========================================================================================================
# Implementation
# =========================================================================================================

class MOEAD_LWS(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.8,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        MOEA/D-LWS Algorithm.

        Parameters
        ----------
        ref_dirs
        display
        kwargs
        """

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        super().__init__(display=display, **kwargs)

        # initialized when problem is known
        self.ref_dirs = ref_dirs

    def _initialize(self):

        super()._initialize()

        self.all_F = []
        self.all_CV = []

        self.ideal_point = np.min(self.pop.get("F"), axis=0)
        self.nadir_point = np.max(self.pop.get("F"), axis=0)

        # Assign each weight with a randomly selected solution
        self.assoc_index = np.random.permutation(self.pop_size)

        # define the weights for the weighted sum
        mat = np.array([[1/(pt+1e-6) for pt in row] for row in self.ref_dirs])
        self.ws_weights = np.array([row/sum(row) for row in mat])

        # Find the 'T' neighboring weights
        T = self.n_neighbors
        matrix = [[calc_angle(a,b) for a in self.ref_dirs] for b in self.ref_dirs]
        self.neighbor_weights = [np.argsort(row)[1:T+1] for row in matrix]

        # find the apex angle of each weight
        M = self.problem.n_obj
        self.apex_angles = np.array([np.mean(np.sort(row)[1:M+1]) for row in matrix])

    def _next(self):
        repair, crossover, mutation = self.repair, self.mating.crossover, self.mating.mutation

        #set a temporary solution set
        Sc = []

        for i in range(self.pop_size):

            if np.random.random() < self.prob_neighbor_mating:
                # find the neighboring solutions 
                Q = [j for j in range(self.pop_size) if self.assoc_index[j] in self.neighbor_weights[i]]
            else:
                Q = [j for j in range(self.pop_size)]
            
            # generate a new soution
            parents = np.random.permutation(Q)[:crossover.n_parents]
            off = crossover.do(self.problem, self.pop, parents[None, :])
            off = mutation.do(self.problem, off)
            Sc.append(off[0])

        # evaluate Sc and merge the population 
        off_x = np.array([ind.X for ind in Sc])
        offs = Population(n_individuals=self.pop_size)
        offs = offs.new("X", off_x)

        self.evaluator.eval(self.problem, offs, algorithm=self)
        JointS = Population.merge(self.pop, offs)

        #update the ideal and nadir points and normalize the pop
        F = JointS.get("F")
        self.ideal_point = np.min(F, axis=0)
        self.nadir_point = np.max(F, axis=0)
        F_norm = normalize(F, x_min=self.ideal_point, x_max=self.nadir_point)

        # compute the theta-(i,j) matrix
        theta = np.array([[calc_angle(f,w) for w in self.ref_dirs] for f in F_norm])

        # compute the matrix C (2N X N)
        C = np.zeros(theta.shape)
        for i in range(len(F_norm)):
            for j in range(self.pop_size):
                if theta[i,j] < self.apex_angles[j]:
                    C[i,j] = get_ws(F_norm[i],self.ws_weights[j])
                else:
                    C[i,j] = math.inf

        # Set the selection set empty
        S = []

        all_CV = np.array([ind.CV[0] for ind in JointS])
        for i in range(len(self.ref_dirs)):
            index = 0
            compute = C[:,i]
            order = np.argsort(compute)

            if self.problem.n_constr>0:
                # constraint handling
                order = np.array([val for val in order if compute[val]<math.inf]) # solutions to be considered during the update
                if len(order)>0:
                    if self.pop[i].CV>0 and min(all_CV[order])==0:
                        temp = [val for val in order if all_CV[val]==0] #only the feasible solutions
                        self.pop[i] = JointS[temp[0]] # the feasible solution with minimum WS

                    elif self.pop[i].CV>0 and min(all_CV[order])>0:
                        if min(all_CV[order])<self.pop[i].CV:
                            self.pop[i] = JointS[order[np.argmin(all_CV[order])]]

                    elif self.pop[i].CV==0 and min(all_CV[order])==0:
                        temp = [val for val in order if all_CV[val]==0]
                        self.pop[i] = JointS[temp[0]]
            else:
                if len(order)>0:
                    self.pop[i] = JointS[order[0]]
            
        self.assoc_index = [i for i in range(self.pop_size)]
        self.all_F = (self.pop.get("F"))
        self.all_CV = (self.pop.get("CV"))
        
def get_ws(f, ref):
    val = np.dot(f, ref)
    return val

def calc_angle(a,b):
    val = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    if val>1:
        val = 1
    elif val<0:
        val = 0
    return math.acos(val)