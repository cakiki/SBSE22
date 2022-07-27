from collections import defaultdict
from copy import deepcopy
import inspect
import random
import time
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Dict, List, Set, Tuple

from fuzzingbook.Coverage import Coverage, branch_coverage


##### EXAMPLE CODE TO GET STARTED


def get_methods(class_ref):
    methods = [
        method
        for method in dir(class_ref)
        if not method.startswith("__") or "__init__" in method
    ]
    method_dict = {}
    for method in methods:
        method_dict[method] = {"params": {}}
        method_ref = getattr(class_ref, method)
        m_src = inspect.getsource(method_ref).split("\n")
        method_dict[method]["src"] = m_src
        method_params = inspect.signature(method_ref).parameters
        for param, p_obj in method_params.items():
            if param != "self":
                m_type = p_obj.annotation
                method_dict[method]["params"][param] = m_type
        method_dict[method]["n_params"] = len(method_dict[method]["params"])
    return method_dict


class MethodCall:
    FAILED = "FAILED"

    def __init__(self, method_name: str, method_metadata: Dict, params: Dict) -> None:
        self.coverage = None
        self.execution_time = None
        self.method_metadata = method_metadata
        self.method_name = method_name
        self.params = params
        self.result = None
        # for parameter_name, parameter_type in self.method_metadata["params"].items():
        #     # TODO: handle types here
        #     print("Processing", parameter_type, "parameter")
        #     self.params[parameter_name] = 5

    # def is_init(self) -> bool:
    #     return self.method_name == "__init__"

    def run_constructor(self, constructor) -> Any:
        """
        super cursed
        """
        args = {k: v for k, v in self.params.items()}

        with Coverage() as coverage:
            try:
                start_time = time.time()
                result = constructor(**args)
            except:
                result = MethodCall.FAILED

        self.execution_time = time.time() - start_time
        self.result = result
        del coverage.original_trace_function
        self.coverage = coverage
        return self.result

    def run_method(self, class_instance) -> Tuple[float, Coverage, Any]:
        """
        cursed
        """
        args = {k: v for k, v in self.params.items()}

        with Coverage() as coverage:
            try:
                start_time = time.time()
                result = getattr(class_instance, self.method_name)(**args)
            except:
                result = MethodCall.FAILED

        self.execution_time = time.time() - start_time
        self.result = result
        del coverage.original_trace_function
        self.coverage = coverage
        return self.coverage, self.result

    def run_raw(self, function_pointer):
        args = {k: v.get() for k, v in self.params.items()}
        if self.is_init():
            # obj is Class
            result = function_pointer(**args)
        else:
            result = getattr(function_pointer, self.method_name)(**args)
        return result

    def branch_coverage_fitness(self) -> int:
        trace = [
            (method, line)
            for method, line in self.coverage.trace()
            #  if method not in ignore_methods
        ]
        pairs = branch_coverage(trace)
        return len(pairs)

    # def line_coverage_fitness(self, ignore_methods=None) -> int:
    #     lines = [
    #         (method, line)
    #         for method, line in self.coverage.coverage()
    #         if method not in ignore_methods
    #     ]
    #     return len(lines)

    def execution_time_fitness(self) -> float:
        return self.execution_time

    def get_line(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        s = "{}({})".format(
            self.method_name,
            ", ".join(
                [
                    "{} = {}".format(p_name, p_val)
                    for p_name, p_val in self.params.items()
                ]
            ),
        )
        return s


class AbstractCandidate(ABC):
    @abstractmethod
    def get_cost(self) -> float:
        """
        returns cost in execution time
        """
        return NotImplemented

    @abstractmethod
    def get_coverage(self) -> int:
        """
        returns coverage as a number
        """
        return NotImplemented

    @abstractmethod
    def run(self):
        """
        runs test and throws no error
        """
        return NotImplemented

    @abstractmethod
    def get_lines_of_code(self) -> str:
        """
        returns string that contains the lines of the test
        """
        return NotImplemented


class Candidate(AbstractCandidate):
    def __init__(self, target_class: type, method_dict: Dict):
        self.target_class = target_class
        self.method_dict = method_dict
        self.all_params = {}
        self.genome_vec = None
        self.method_calls = None
        self.total_execution_time = None
        self.total_coverage = None
        self.total_code = None
        self.rank = None
        self.sparsity = 0

    def _crossover(self, vec_a: List, vec_b: List) -> Tuple[List, List]:
        point = random.randint(1, len(vec_a) - 1)

        child_vec_a = vec_a[:point]
        child_vec_a.extend(vec_b[point:])

        child_vec_b = vec_b[:point]
        child_vec_b.extend(vec_a[point:])

        return child_vec_a, child_vec_b

    def _mutate(self, probability: float) -> None:
        for i in range(len(self.genome_vec)):
            if random.uniform(0, 1) < probability:
                gen_type = type(self.genome_vec[i])
                if gen_type is int:
                    self.genome_vec[i] = random.randint(-10, 10)
                if gen_type is float:
                    self.genome_vec[i] = random.uniform(-10, 10)

    def breed(self, partner, mutation_rate) -> Tuple:
        child_a = Candidate(self.target_class, self.method_dict)
        child_b = Candidate(self.target_class, self.method_dict)

        child_a.genome_vec, child_b.genome_vec = self._crossover(
            self._vectorize(), partner._vectorize()
        )

        child_a._mutate(mutation_rate)
        child_b._mutate(mutation_rate)

        child_a._devectorize()
        child_b._devectorize()

        return child_a, child_b

    def _vectorize(self) -> List:
        """
        Converts method_args (the genome) into a vector to easily mutate
        """
        list_of_list = [i.values() for i in self.all_params.values()]
        return list(chain(*list_of_list))

    def _devectorize(self):
        """
        Converts a vector into a human readable representation thats also usable for testing.
        """
        i = 0

        for method_name, method_metadata in self.method_dict.items():
            params = {}

            for param_name in method_metadata["params"].keys():
                params[param_name] = self.genome_vec[i]
                i += 1

            self.all_params[method_name] = params

    def gen_random(self):
        for method_name, method_metadata in self.method_dict.items():
            params = {}

            for parameter_name, parameter_type in method_metadata["params"].items():
                if parameter_type is int:
                    params[parameter_name] = random.randint(-10, 10)
                if parameter_type is float:
                    params[parameter_name] = random.uniform(-10, 10)

            self.all_params[method_name] = params

    def get_cost(self) -> float:
        """
        returns cost in execution time
        """
        return self.total_execution_time

    def get_coverage(self) -> int:
        """
        returns coverage as a number
        """
        return self.total_coverage

    def run(self):
        """
        runs test and throws no error
        """
        self.total_execution_time = 0
        self.total_coverage = 0
        self.total_code = []

        constructor_call = None
        method_calls = []

        for method_name, method_metadata in self.method_dict.items():
            if method_name == "__init__":
                constructor_call = MethodCall(
                    method_name, method_metadata, self.all_params[method_name]
                )
            else:
                method_calls.append(
                    MethodCall(
                        method_name, method_metadata, self.all_params[method_name]
                    )
                )

        class_instance = constructor_call.run_constructor(self.target_class)
        self.total_execution_time += constructor_call.execution_time_fitness()
        self.total_coverage += constructor_call.branch_coverage_fitness()
        self.total_code.append(constructor_call.get_line())

        for method_call in method_calls:
            method_call.run_method(class_instance)
            self.total_execution_time += method_call.execution_time_fitness()
            self.total_coverage += method_call.branch_coverage_fitness()
            self.total_code.append(method_call.get_line())

    def get_lines_of_code(self) -> str:
        """
        returns string that contains the lines of the test
        """
        return "\n\n".join(self.total_code)

    def get_fitness(self) -> Tuple:
        return {
            "execution time": self.total_execution_time,
            "coverage": self.total_coverage,
        }


def generate_random_pop(target_class: type, method_dict: Dict, n: int):
    candidates = set(Candidate(target_class, method_dict) for i in range(n))
    for candidate in candidates:
        candidate.gen_random()
        candidate.run()
    return candidates


def dominates(candidate_1: Candidate, candidate_2: Candidate) -> bool:
    """
    Pareto Domination binary domination

    returns wether or not candidate_1 dominates candidate_2
    """
    if (
        candidate_1.get_cost() == candidate_2.get_cost()
        and candidate_1.get_coverage() == candidate_2.get_coverage()
    ):
        return False
    elif (
        candidate_1.get_cost() <= candidate_2.get_cost()
        and candidate_1.get_coverage() >= candidate_2.get_coverage()
    ):
        return True
    else:
        return False


def get_pareto_front(population: Set[Candidate]) -> Set[Candidate]:
    front = set()
    for challenged in population:
        front = front | {challenged}
        for challengers in front:
            if dominates(challengers, challenged):
                front = front - {challenged}
                break
            elif dominates(challenged, challengers):
                front = front - {challengers}
    return front


def calculate_pareto_front_ranks(pop: Set[Candidate]) -> Set[Candidate]:
    """
    Calculates the pareto fron rank for all
    candidates in population

    Keyword arguments:
    population -- list holding all candidates
    """

    remaining_pop = deepcopy(pop)
    ranked_pop = set()
    current_rank = 1
    while remaining_pop:
        pareto_front = get_pareto_front(remaining_pop)
        for candidate in pareto_front:
            candidate.rank = current_rank
        remaining_pop = remaining_pop - pareto_front
        ranked_pop = ranked_pop | pareto_front
        current_rank += 1
    return ranked_pop


def calculate_sparsity(pop: Set[Candidate]) -> Set[Candidate]:       
    by_coverage = sorted(pop, key=lambda x: x.get_coverage())

    by_coverage_max = max(candidate.get_coverage() for candidate in pop)
    by_coverage_min = min(candidate.get_coverage() for candidate in pop)
    by_coverage_range = by_coverage_max - by_coverage_min

    by_coverage[0].sparsity = float("inf")
    by_coverage[-1].sparsity = float("inf")

    for i, candidate in enumerate(by_coverage[1:-1], start=1):
        candidate.sparsity += (
            by_coverage[i + 1].get_coverage() - by_coverage[i - 1].get_coverage()
        ) / by_coverage_range

    by_cost = sorted(pop, key=lambda x: x.get_cost(), reverse=True)

    by_cost_max = max(candidate.get_cost() for candidate in pop)
    by_cost_min = min(candidate.get_cost() for candidate in pop)
    by_cost_range = by_cost_max - by_cost_min

    by_cost[0].sparsity = float("inf")
    by_cost[-1].sparsity = float("inf")

    for i, candidate in enumerate(by_cost[1:-1], start=1):
        candidate.sparsity += (
            by_coverage[i + 1].get_cost() - by_coverage[i - 1].get_cost()
        ) / by_cost_range

    return pop


def calculate_sparsity_by_front(
    pop: Set[Candidate],
) -> Set[Candidate]:
    by_front = defaultdict(set)
    for candidate in pop:
        by_front[candidate.rank].add(candidate)

    return_pop = set()
    for front in by_front.values():
        return_pop |= calculate_sparsity(front)

    return return_pop


def select_to_breed(archive: List[Candidate]) -> Tuple[Candidate, Candidate]:
    potential_parent_a = random.sample(archive, 2)
    potential_parent_b = random.sample(archive, 2)

    ranked_parents_a = sorted(potential_parent_a, key=lambda x: (x.rank, -x.sparsity))
    ranked_parents_b = sorted(potential_parent_b, key=lambda x: (x.rank, -x.sparsity))

    return ranked_parents_a[0], ranked_parents_b[0]


def breed_archive(breed_size: int, mutation_rate: float, archive: List[Candidate]) -> Set[Candidate]:
    children = set()

    while len(children) <= breed_size:
        partner_a, partner_b = select_to_breed(archive)
        children |= set(partner_a.breed(partner_b, mutation_rate))

    return set(list(children)[:breed_size])


class AbstractGenerator(ABC):
    def generate(self, target_class: type) -> List[AbstractCandidate]:
        """
        Runs MOO and returns the final solution set containing the pareto front
        Always uses doesitwork.py in the same directory. This file will be swapped out for evaluation.
        """
        return NotImplemented


class Generator(AbstractGenerator):
    def __init__(self):
        pass

    def generate(self, target_class: type) -> List[AbstractCandidate]:
        ###################################
        ### IMPLEMENT YOUR SOLUTION HERE ##
        ###################################
        POP_SIZE = 40
        ARCHIVE_PERCENTAGE = 0.2
        MUTATION_RATE = 0.2

        archive_size = int(POP_SIZE * ARCHIVE_PERCENTAGE)
        breed_size = POP_SIZE - archive_size

        method_dict = get_methods(target_class)
        
        pop = generate_random_pop(target_class, method_dict, POP_SIZE)
        archive = []
        generation = 1
        while True:
            for candidate in pop:
                candidate.rank = None
                candidate.sparsity = 0

            # select, breed, mutate
            pop = calculate_pareto_front_ranks(pop)
            pop = calculate_sparsity_by_front(pop)
            archive = sorted(pop, key=lambda x: (x.rank, -x.sparsity))[0:archive_size]

            # print("\nCandidate:")
            # for entry in pop:
            #     print(f"My rank is {entry.rank} and my sparsity is {entry.sparsity}")

            # print("\nArchive:")
            # for entry in archive:
            #     print(f"My rank is {entry.rank} and my sparsity is {entry.sparsity}")

            children = breed_archive(breed_size, MUTATION_RATE, archive)
            for child in children:
                child.run()

            pop = set(archive) | children

            # yield to act like a iterable for interface as used in provided test cases
            yield list(pop)
            generation += 1
