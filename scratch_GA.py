import math
from typing import Dict,List
import random

class GeneticOptimizer:
    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        @staticmethod
        def get_default_config() -> Dict:
             return {
                "population_size": 50,
                "num_generations": 1000,
                "crossover_rate": 0.6,
                "mutation_rate": 0.001}
        self.population_size = self.config["population_size"]
        self.num_generations = self.config["num_generations"]
        self.crossover_rate = self.config["crossover_rate"]
        self.mutation_rate = self.config["mutation_rate"]
        self.initial_alt = self.config["initial_alt"]
        self.target_alt = self.config["target_alt"]
        self.time_delta = self.config["time_delta"]
        self.D = [self.initial_alt, self.target_alt, self.time_delta]
        

    def fitness_function(self, D: Dict) -> float:
        alt_dev = (D[self.target_alt] - D[self.current_alt])**2
        return alt_dev

    def generate_initial_population(self) -> List[Dict]:
        population = []
        for i in range(self.population_size):
            population = self.generate_individual()
            population.append(population)
            return population
    
    def generate_individual(self) -> Dict:
        individual = {}
        individual[self.initial_alt] = random.uniform(self.D[0], self.D[1])
        individual[self.target_alt] = random.uniform(self.D[0], self.D[1])
        individual[self.time_delta] = random.uniform(self.D[0], self.D[1])
        return individual

    

        



        


            
        

