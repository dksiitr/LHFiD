import numpy as np
from scipy.spatial.distance import cdist
import warnings
from numpy.linalg import LinAlgError
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.factory import get_decomposition
from pymoo.model.individual import Individual
from pymoo.model.population import Population, pop_from_array_or_individual
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

class LHFID(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        HFIDEAD Algorithm.

        Parameters
        ----------
        ref_dirs
        display
        kwargs

        For constraint-handling:
        add tournament selection
        divide joint pop between feasible and infeasible
        determine nadir point from feasible solutions only
        """

        self.extreme_points = None
        self.F_check = None
        self.term_gen = 1

        """For overall termination"""
        self.termination_suggestion = None
        self.termination_pop = None

        """Archives for stabilization tracking algorithm"""
        self.mu_D = []
        self.D_t = []
        self.S_t = []
        self.Q = None

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=0.9, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=0.1, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        super().__init__(display=display, **kwargs)

        # initialized when problem is known
        self.ref_dirs = ref_dirs
    
    def _initialize(self):

        super()._initialize()

        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        pop.set("n_gen", self.n_gen)
        self.pop = pop
        self.evaluator.eval(self.problem, self.pop, algorithm=self)

        """Compute the ideal point from the initialized population"""
        self.ideal_point = np.min(self.pop.get("F"), axis=0)

        """Set the nadir point as empty set"""
        self.nadir_point = None 

    def _next(self):

        """Do the mating to produce the offspring Q_t"""
        self.off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        self.off.set("n_gen", self.n_gen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        """Update the ideal point"""
        temp = np.min(self.off.get("F"), axis=0)
        self.ideal_point = np.min([temp, self.ideal_point], axis=0)
        self.Q = self.pop.get("F")

        """Merge P_t and Q_t into U_t"""
        self.pop = Population.merge(self.pop, self.off)

        """Execute the environmental selection"""
        self.pop = survival_selection(self)

        """Update the archives required for stabilization tracking algorithm"""
        update_termination_archives(self, self.Q)

        """If mild stabilization is achieved, update the nadir point"""
        check_for_nadir_update(self,2,20) # {2, 20} are the parameters used for nadir-point update

        """Check if LHFiD needs to be terminated"""        
        if self.termination_suggestion is None:
            term_cond = check_for_termination(self,3,50) # {3, 50} are the parameters used for LHFiD termination
            if term_cond:
                self.termination_suggestion = self.n_gen
                self.termination_pop = self.pop

        """Terminate LHFiD, if suggested by the stabilization tracking algorithm"""
        if self.termination_suggestion is not None:
            self.termination.force_termination = True
            return

def survival_selection(self):

    F = self.pop.get("F")
    feas_pop = self.pop

    """If nadir point is not empty, normalize, else, translate the population"""
    if self.nadir_point is not None:
        F_norm = normalize(F, x_min=self.ideal_point, x_max=self.nadir_point)
    else:
        F_norm = np.array([(pt-self.ideal_point) for pt in F])

    """Calculate the perpendicular distance of each solution from each RV, and associate"""
    dist_matrix = load_function('calc_perpendicular_distance')(F_norm, self.ref_dirs)
    assoc_index = np.array([np.argmin(row) for row in dist_matrix])
    assoc_value = np.array([min(row) for row in dist_matrix])

    """Set the survived population as empty"""
    survivors = []
    selected_vectors = []

    """Selection of solutions on the non-empty RVs"""
    for i in range (len(self.ref_dirs)):
        
        """Cluster the associated solutions"""
        cluster_indices = [p for p in range(len(assoc_index)) if assoc_index[p]==i]
        cluster_perp_values = [assoc_value[p] for p in range(len(assoc_index)) if assoc_index[p]==i]
        
        if len(cluster_indices)==1:
            survivors.append(cluster_indices[0])
            selected_vectors.append(i)

        elif len(cluster_indices)>1:
            
            is_pareto_non_dom = []
            for ind1 in cluster_indices:
                count = 0
                for ind2 in cluster_indices:
                    if ind1 != ind2:    
                        n_b = sum(F_norm[ind1]>F_norm[ind2])
                        n_w = sum(F_norm[ind1]<F_norm[ind2])
                        if n_b>=1 and n_w==0:
                            count+=1
                if count==0:
                    is_pareto_non_dom.append(True)
                else:
                    is_pareto_non_dom.append(False)

            """Remove the Pareto-dominated solutions from the cluster"""
            cluster_indices = [cluster_indices[p] for p in range(len(cluster_indices)) if is_pareto_non_dom[p] is True]
            cluster_perp_values = [cluster_perp_values[p] for p in range(len(cluster_perp_values)) if is_pareto_non_dom[p] is True]

            """Identify the alpha-solution"""
            alpha_solution = cluster_indices[np.argmin(cluster_perp_values)]

            """Find the solutions that hf-dominate the alpha solution"""
            if max(self.ref_dirs[i])<1: # For the non-axis reference-vectors
                candidates = []
                candidates_perp_d = []
                for p in range(len(cluster_indices)):
                    ind = cluster_indices[p]
                    if ind != alpha_solution:
                        temp_alpha, temp_ind = F_norm[alpha_solution], F_norm[ind]
                        for q in range(self.problem.n_obj):
                            if abs(temp_alpha[q]-temp_ind[q])<1e-06:
                                temp_alpha[q] = temp_ind[q]
                        n_b = sum(temp_alpha>temp_ind)
                        n_w = sum(temp_alpha<temp_ind)
                        if n_b >= n_w:
                            decide = tie_breaker(F_norm[ind], F_norm[alpha_solution], self.ref_dirs[i])
                            if decide<0:
                                candidates.append(ind)
                                candidates_perp_d.append(cluster_perp_values[p])
                
                """If one or more beta solutions are identified"""
                if len(candidates)>=1:
                    alpha_solution = candidates[np.argmin(candidates_perp_d)]
            
            """Append the selected solution in the survived solutions"""
            survivors.append(alpha_solution)
            selected_vectors.append(i)

    """Identify the solutions and RVs not selected yet"""
    left_solutions = [i for i in range(len(feas_pop)) if i not in survivors]
    left_vectors = [i for i in range(len(self.ref_dirs)) if i not in selected_vectors]

    """Choose the solutions closest to the unselected RVs"""
    dist_matrix = load_function('calc_perpendicular_distance')(F_norm[left_solutions], self.ref_dirs[left_vectors])
    for i in range(len(left_vectors)):
        dis = dist_matrix[:,i]
        survivors.append(left_solutions[np.argmin(dis)])
    return feas_pop[survivors]

"""To check the second condition of hf-dominance"""
def tie_breaker(f, alpha, rd):
    M = len(rd)
    sum_val = 0
    for i in range(M):
        if rd[i]==0:
            temp = 1e-6
        else:
            temp = rd[i]
        sum_val += (f[i] - alpha[i])/temp
    return sum_val

"""To update the archives related to the stabilization tracking algorithm"""
def update_termination_archives(self,Q):
    P = self.pop.get("F")
    if len(P)>0 and len(Q)>0:
        if self.nadir_point is not None:
            P_norm = normalize(P, x_min=self.ideal_point, x_max=self.nadir_point)
            Q_norm = normalize(Q, x_min=self.ideal_point, x_max=self.nadir_point)
        else:
            P_norm = np.array([(pt-self.ideal_point) for pt in P])
            Q_norm = np.array([(pt-self.ideal_point) for pt in Q])
        dist_matrix = load_function('calc_perpendicular_distance')(P_norm, self.ref_dirs)
        assoc_P = np.array([np.argmin(row) for row in dist_matrix])
        dist_matrix = load_function('calc_perpendicular_distance')(Q_norm, self.ref_dirs)
        assoc_Q = np.array([np.argmin(row) for row in dist_matrix])
        mu_D = 0
        count = 0
        for i in range (len(self.ref_dirs)):
            cluster_P = [p for p in range(len(assoc_P)) if assoc_P[p]==i]
            cluster_Q = [p for p in range(len(assoc_Q)) if assoc_Q[p]==i]
            if len(cluster_P)>0 and len(cluster_Q)>0:
                p = np.mean(P_norm[cluster_P], axis=0)
                q = np.mean(Q_norm[cluster_Q], axis=0)
                D = abs(np.matmul(self.ref_dirs[i],p)-np.matmul(self.ref_dirs[i],q))
                if D!=0:
                    D = D/max(np.matmul(self.ref_dirs[i],p),np.matmul(self.ref_dirs[i],q))
                else:
                    D = 0
                mu_D += D
                count += 1
            elif  len(cluster_P)==0 and len(cluster_Q)>0:
                mu_D += 1
                count += 1
            elif  len(cluster_P)>0 and len(cluster_Q)==0:
                mu_D += 1
                count += 1
        mu_D = mu_D/count
    else:
        mu_D = 1
    self.mu_D.append(mu_D)
    self.D_t.append(np.mean(self.mu_D))
    self.S_t.append(np.std(self.mu_D))

"""Check the conditions for nadir point update and update it required"""
def check_for_nadir_update(self,n_p,ns):
    self.term_gen += 1
    if self.term_gen>ns:
        D_t = self.D_t[-ns:]
        S_t = self.S_t[-ns:]
        D_t = [round(val,n_p) for val in D_t]
        S_t = [round(val,n_p) for val in S_t]
        if len(np.unique(D_t))==1 and len(np.unique(S_t))==1:
            self.nadir_point = get_nadir_point(self)

"""Check the conditions for LHFiD termination"""
def check_for_termination(self,n_p,ns):
    to_terminate = False
    if self.n_gen>ns:
        D_t = self.D_t[-ns:]
        S_t = self.S_t[-ns:]
        D_t = [round(val,n_p) for val in D_t]
        S_t = [round(val,n_p) for val in S_t]
        if len(np.unique(D_t))==1 and len(np.unique(S_t))==1:
            to_terminate = True
        return to_terminate

"""Compute the nadir point"""
def get_nadir_point(self):
    ideal_point = self.ideal_point
    F = self.pop.get("F")
    is_pareto_non_dom = []
    for ind1 in F:
        count = 0
        for ind2 in F:    
            n_b = sum(ind1>ind2)
            n_w = sum(ind1<ind2)
            if n_b>=1 and n_w==0:
                count+=1
        if count==0:
            is_pareto_non_dom.append(True)
        else:
            is_pareto_non_dom.append(False)
    F = np.array([F[i] for i in range(len(F)) if is_pareto_non_dom[i]==True])
    if len(F)==0:
        return None
    else:
        self.extreme_points = get_extreme_points_c(F,ideal_point,self.extreme_points)
        extreme_points = self.extreme_points
        try:
            M = extreme_points - ideal_point
            b = np.ones(extreme_points.shape[1])
            plane = np.linalg.solve(M, b)
            warnings.simplefilter("ignore")
            intercepts = 1 / plane
            nadir_point = ideal_point + intercepts
            if not np.allclose(np.dot(M, plane), b) or np.any(intercepts <= 1e-6):
                raise LinAlgError()
            else:
                self.term_gen = 1
        except LinAlgError:
            nadir_point = self.nadir_point
        return nadir_point

"""Compute the extreme points (for nadir point computation)"""
def get_extreme_points_c(F, ideal_point, extreme_points):
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0
    F_asf = np.max(__F * weights[:, None, :], axis=2)
    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]
    return extreme_points