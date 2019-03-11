import unittest
import array
import numpy as np
import cobra
import random
import os
import multiprocessing
from itertools import repeat
from itertools import product
from deap import base
from deap import creator
from deap import tools
from deap import algorithms



from deap.benchmarks.tools import diversity, convergence, hypervolume
#from jamieexcel_to_json import cobrajsonfromexcel

import matplotlib.pyplot as plt
import csv

class TestFBAEvolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._evolver = CobraFBA('E:\\Chrome Download\\FBA\\FBA\\iJO1366.json')

    def test_biomass_amount(self):
        solution = self._evolver.solution
        self.assertAlmostEqual(solution, 0.9865144469529762)

    def test_reaction_values(self):
        model = self._evolver.model
        self.assertEqual(len(model.reactions), 2583)
        self.assertEqual(model.reactions[10].lower_bound, -1000)
        self.assertEqual(model.reactions[10].upper_bound, 1000)
        self.assertEqual(model.reactions[10].id, '12PPDStpp')

    def test_reaction_values(self):
        init_values = self._evolver.initial_reaction_bounds
        self.assertEqual(len(init_values), 2255)

    def test_Ex_removed_from_reactions(self):
        init_values = self._evolver.initial_reaction_bounds
        for reaction_id, reaction_bounds in init_values.items():
            self.assertFalse('EX' in reaction_id)

    def test_set_reactions(self):
        zero_individual = np.zeros(len(self._evolver.initial_reaction_bounds))
        self._evolver.set_reaction_bounds(zero_individual)

        for reaction in self._evolver.model.reactions:
            if 'EX' in reaction.id:
                pass
            else:
                self.assertEqual(reaction.lower_bound, 0 )
                self.assertEqual(reaction.upper_bound, 0)

        ones_individual = np.ones(len(self._evolver.initial_reaction_bounds))
        self._evolver.set_reaction_bounds(ones_individual)
        for reaction in self._evolver.model.reactions:
            if 'EX' in reaction.id:
                pass
            else:
                self.assertEqual(reaction.lower_bound, self._evolver.initial_reaction_bounds[reaction.id][0])
                self.assertEqual(reaction.upper_bound, self._evolver.initial_reaction_bounds[reaction.id][1])



class CobraFBA():
    def __init__(self, input, knockout= False):

        # import iaf1260
        self.model = cobra.io.load_json_model(input)
        print(self.model.objective)
        self.knockout = knockout
        #self.model.objective = "BiomassEcoli"
        if knockout:
            print('Knocked Out: ' + str(knockout))
            self.model.reactions.get_by_id(knockout).knock_out()
        self.non_essential_reactions, self.essential_reactions, self.exchanges, self.exchange_bounds, self.modifiable_exchanges = self.evaluate_essential_reactions(self.model)
        self.reaction_names, self.initial_reaction_bounds  = self.find_reaction_intial_bounds(self.non_essential_reactions)
        self.essential_reaction_names, essential_reaction_bounds = self.find_reaction_intial_bounds(self.essential_reactions)



    def find_reaction_intial_bounds(self, reactions):
        reaction_names = list()
        exchange_names = list()
        exchange_bounds = list()
        reaction_bounds = list()
        for reaction in reactions:
            if 'EX' in reaction.id[:2]:
                exchange_names.append(reaction.id)
                exchange_bounds.append(reaction.upper_bound)

            else:
                reaction_names.append(reaction.id)
                reaction_bounds.append([reaction.lower_bound, reaction.upper_bound])


        return reaction_names, dict(zip(reaction_names, reaction_bounds))

    def set_reaction_bounds(self, individual):

        for reaction_active, reaction in zip(individual, self.reaction_names):
            if reaction_active:
                bounds = self.initial_reaction_bounds[reaction]
                self.model.reactions.get_by_id(reaction).lower_bound = bounds[0]
                self.model.reactions.get_by_id(reaction).upper_bound = bounds[1]
            else:
                self.model.reactions.get_by_id(reaction).lower_bound = 0
                self.model.reactions.get_by_id(reaction).upper_bound = 0

    def set_modifiable_exchanges(self, individual):
        for exchange_active, exchange in zip(individual, self.modifiable_exchanges):
            if exchange_active:
                self.model.reactions.get_by_id(exchange.id).lower_bound = -1000
            else:
                self.model.reactions.get_by_id(exchange.id).lower_bound = 0

    def run_fba(self, queue = False):
        growth = self.model.slim_optimize()
        if self.knockout:
            self.model.reactions.get_by_id(self.knockout).knock_out()
        if queue:
            queue.put(growth)
        return growth

    def evaluate_essential_reactions(self,model):
        reactions_list = list()
        reaction_bounds = list()
        exchange_list = list()
        modifiable_exchange_list = list()
        exchange_bounds = list()


        for reaction in model.reactions:
            if 'EX' in reaction.id[:2]:
                exchange_list.append(reaction)
                exchange_bounds.append(reaction.lower_bound)
                #print('Exchange Bounds:' + str(reaction.id) + ' Lower: ' + str(reaction.lower_bound) + ' Upper Bound: ' + str(reaction.upper_bound))
                if reaction.lower_bound != 0:
                    pass
                else:

                    modifiable_exchange_list.append(reaction)

            else:
                #print('Reaction Bounds: '+ str(reaction.id) +' Lower: ' + str(reaction.lower_bound) + ' Upper Bound: ' + str(
                 #   reaction.upper_bound))
                if np.abs(reaction.lower_bound) +reaction.upper_bound != 0:
                    reactions_list.append(reaction)
                    reaction_bounds.append([reaction.lower_bound, reaction.upper_bound])
                else:
                    print(reaction.id)
        print(len(reactions_list))

        non_essential_reactions = list()
        essential_reactions = list()

        for reaction, reaction_bound in zip(reactions_list, reaction_bounds):
            wt_growth = model.slim_optimize()
            if 'PPC' in reaction.id:
                print('Reaction id: ' + str(reaction.id) + ' Upper Bound: ' + str(model.reactions.get_by_id(reaction.id).upper_bound)
                      + ' Lower Bound: ' + str(model.reactions.get_by_id(reaction.id).upper_bound))
                print('Reaction id: ' + str(reaction.id) + ' Wild Type Growth: ' + str(wt_growth))

            model.reactions.get_by_id(reaction.id).lower_bound = 0
            model.reactions.get_by_id(reaction.id).upper_bound = 0
            knockout_growth = model.slim_optimize()
            if 'PPC' in reaction.id:
                print('Reaction id: ' + str(reaction.id) + ' Upper Bound: ' + str(
                    model.reactions.get_by_id(reaction.id).upper_bound)
                      + ' Lower Bound: ' + str(model.reactions.get_by_id(reaction.id).upper_bound))
                print('Reaction id: ' + str(reaction.id) + ' Knockout Growth: ' + str(knockout_growth))

            if knockout_growth > 0.001:
                non_essential_reactions.append(reaction)
            else:
                essential_reactions.append(reaction)

            model.reactions.get_by_id(reaction.id).lower_bound = reaction_bound[0]
            model.reactions.get_by_id(reaction.id).upper_bound = reaction_bound[1]


        print('Number of essential reactions: ' + str(len(essential_reactions)))
        print('Number of non essential reactions: ' + str(len(non_essential_reactions)))
        print('Number of unset reactions: ' + str(len(modifiable_exchange_list)))

        return non_essential_reactions, essential_reactions, exchange_list, dict(zip(exchange_list,exchange_bounds)),modifiable_exchange_list


class CobraFBABase():
    def __init__(self):
        self.fba = None

    def fitness_function_no_queue(self, individual):
        self.fba.set_reaction_bounds(individual)
        growth = self.fba.run_fba()

        if growth > 0.001:
            reactions = sum(individual)
        else:
            reactions = len(self.fba.non_essential_reactions)
        return growth, reactions

    def fitness_function(self,individual,queue):

        # with open('current_ind.csv', 'w') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',')
        #     writer.writerow(individual)
        self.fba.set_reaction_bounds(individual)

        p = multiprocessing.Process(target=self.fba.run_fba, args=(queue,))
        p.start()
        p.join(3)
        if p.is_alive():
            print('Killing Thread')
            p.terminate()
            p.join()
            growth = 0
        else:
            growth = queue.get()


        if growth > 0.001:
            reactions = sum(individual)
        else:
            reactions = len(self.fba.non_essential_reactions)
        return growth, reactions

    def exchange_fitness_function(self,individual):

        with open('current_ind.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(individual)

        self.fba.set_modifiable_exchanges(individual)

        growth = self.fba.run_fba()
        if growth > 0.001:
            reactions = sum(individual)
        else:
            reactions = len(self.fba.modifiable_exchanges)
        return growth, reactions



class CobraFBAProcessor(CobraFBABase):
    def __init__(self, input):
        super().__init__()
        self.fba = CobraFBA(input)


class CobraFBAEvolver(CobraFBABase):
    def __init__(self, input,results_folder,knockout=False):
        super().__init__()
        self.population_file = 'false'
        self.results_folder = results_folder
        self.fba = CobraFBA(input,knockout=knockout)
        self.create_evo(len(self.fba.non_essential_reactions))



    def run_nsga2evo(self, num_run, gen_record, seed =None, population =None,population_file=None):
        random.seed(seed)

        NGEN = 51
        MU = 100
        self.population_file = population_file


        stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register("avg", numpy.mean, axis=0)
        # stats.register("std", numpy.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        if population == None:
            try:
                pop = self.toolbox.population_restart()
            except FileNotFoundError:
                pop = self.toolbox.population(n=MU)
        else:
            pop = population


        # Evaluate the individuals with an invalid fitness
        queue = multiprocessing.Queue()
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind, repeat(queue))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = self.toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)

        # Begin the generational process
        for gen in range(1, NGEN):
            # Vary the population

            print('Cloning Population')
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            print('Cloning Completed')

            print('Mutating Offspring')
            for ind1 in offspring:
                self.toolbox.mutate(ind1)
                del ind1.fitness.values
            print('Mutation Completed')

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            print('Evaluating Fitness')
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind,repeat(queue))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            print("Fitness Evaluation Completed")

            # Select the next generation population
            print('Selecting Population')
            pop = self.toolbox.select(pop + offspring, MU)
            print('Selection Completed')
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)


        print('Evolution Completed')
        filename = str(num_run) + '_' + str(gen_record) + '.csv'
        filepath = os.path.join(self.results_folder, filename)
        with open(filepath, 'w') as csvfile:
            print('Writing to ' + filename)
            writer = csv.writer(csvfile, delimiter=',')
            for p in pop:
                writer.writerow(p)
        #print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

        return pop, logbook

    def run_evo(self, iteration):
        random.seed()
        min_fits = list()


        pop = self.toolbox.population(n=50)

        #pop = [np.ones(len(self.fba.reaction_names)) for p in pop]

        CXPB, MUTPB, NGEN = 0, 1, 2000
        MU = 50
        LAMBDA = 100

        print("Start of evolution")

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(pop))
        filename = str(iteration) + '.csv'
        final_filename = str(iteration) + '_final.csv'
        # Begin the evolution

        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        algorithms.eaMuPlusLambda(pop, self.toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                                  halloffame=hof)


    def initInd(self, icls,content):
        return icls(content)
    def initPop(self,pcls,ind_init, filename):
        with open(filename,'r') as csv_file:
            initial_list = list()
            reader = csv.reader(csv_file, delimiter=',')
            for ind in reader:
                if ind:
                    initial_list.append([int(i) for i in ind])
            return pcls(ind_init(ind) for ind in initial_list)

    def create_evo(self, num_reactions):
        creator.create("FitnessMin", base.Fitness, weights=(1,-1))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        # Attribute generator
        self.toolbox.register("attr_bool", random.randint, 1, 1)
        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                         self.toolbox.attr_bool, num_reactions)
        # pool = multiprocessing.Pool()
        # self.toolbox.register("map", pool.starmap)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("individual_restart", self.initInd, creator.Individual)
        self.toolbox.register("population_restart", self.initPop, list, self.toolbox.individual_restart, self.population_file)
        self.toolbox.register("evaluate", self.fitness_function)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.0005)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("HallOfFame", tools.HallOfFame)


if __name__ == "__main__":

    directory = os.getcwd()

    num_runs = range(1, 11)
    gen_record = range(0, 2500, 50)

    print('Beginning FBA Evolution for ' + str(num_runs[-1]) + ' runs of ' + str(gen_record[-1]) + ' generations')

    for num_run in num_runs:
        pop = None
        for gens in gen_record:
            evolver = CobraFBAEvolver(os.path.join(directory, 'iJR904.json'),'./iJR904',)
            filename = str(num_run) + '_1000.csv'
            population_file = os.path.join('./PDH_Knockout_Blood',filename)
            pop, log = evolver.run_nsga2evo(num_run, gens, population=pop)


