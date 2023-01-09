import copy
import numpy as np

class Selection:
    @staticmethod
    def eval_fitness(creatures):
        fitness = []
        for cr in creatures:
            start = np.array(cr.start_position)
            finish = np.array(cr.last_position)
            dist = np.array(cr.get_distance())
            n_joint = len(cr.get_expanded_links())

            reward_x  = 1 + finish[0] - start[0] / 100
            penalty_n = 1 + int(n_joint / 2)
            # to prevent negative value of x distance 'finish[0] - start[0]'
            fit = np.maximum(0, dist * reward_x / penalty_n)  
            fit = np.nan_to_num(fit, nan = 0)
            fitness.append(fit)

        return fitness

    @staticmethod
    def select_parent_indices(fits):
        fits  = np.nan_to_num(np.array(fits), nan = 0.0)
        probs = fits / np.sum(fits)
        if np.mean(probs == 0.0) == 1:
            probs = np.ones(len(fits)) / len(fits)
        return np.random.choice(range(len(fits)), 2, False, probs)

    @staticmethod
    def select_parents(creatures, fits = None):
        if fits == None:
            fits = Selection.eval_fitness(creatures)
        id1, id2 = Selection.select_parent_indices(fits)
        return creatures[id1], creatures[id2]

class Crossover:
    @staticmethod
    def crossover_dna(parent_dna_1, parent_dna_2, max_length_limit = 15, max_growth_rate = 1.2):
        growth_limit = int(np.maximum(len(parent_dna_1), len(parent_dna_2)) * max_growth_rate)
        length_limit = np.minimum(growth_limit, max_length_limit)
        ind_1 = np.random.randint(1, len(parent_dna_1)+1)
        ind_2 = np.random.randint(0, len(parent_dna_2))
        child = np.concatenate((parent_dna_1[:ind_1], parent_dna_2[ind_2:]))
        
        return child[:length_limit]

class Mutation:
    @staticmethod
    def point_mutate(dna, mutation_rate = .05, mutation_amount = .05):
        cp_dna = copy.copy(dna).flatten()
        mutated = np.random.choice((True, False), size = len(cp_dna), replace = True, p = (mutation_rate, 1-mutation_rate))
        cp_dna[mutated] = np.minimum(
            0.0001,
            np.maximum(
                0.999,
                cp_dna[mutated] + np.random.random() * mutation_amount * (-1) ** np.random.randint(1, 3)
            )
        )
        return np.reshape(cp_dna, dna.shape)

    @staticmethod
    def shrink_mutate(dna, mutation_rate = .05):
        cp_dna = copy.copy(dna)
        if len(cp_dna) <= 2:
            return cp_dna
        mutated = np.random.choice((True, False), size = len(cp_dna), replace = True, p = (mutation_rate, 1-mutation_rate))
        return np.delete(cp_dna, mutated, axis = 0)

    @staticmethod
    def grow_mutate(dna, mutation_rate = .05):
        cp_dna = copy.copy(dna)
        mutated = np.random.choice((True, False), size = len(cp_dna), replace = True, p = (mutation_rate, 1-mutation_rate))
        return np.append(cp_dna, cp_dna[mutated], axis = 0)

class NewGeneration:
    @staticmethod
    def generate_child_dna(parent_dna_1, parent_dna_2):
        child_dna = Crossover.crossover_dna(parent_dna_1, parent_dna_2)
        child_dna = Mutation.point_mutate(child_dna)
        child_dna = Mutation.shrink_mutate(child_dna)
        child_dna = Mutation.grow_mutate(child_dna) 
        return child_dna
