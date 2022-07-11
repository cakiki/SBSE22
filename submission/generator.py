import inspect
from abc import ABC, abstractmethod
from fuzzingbook.Coverage import Coverage, branch_coverage


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


class AbstractGenerator(ABC):
    def generate(self, target_class: type):
        """
        Runs MOO and returns the final solution set containing the pareto front
        Always uses doesitwork.py in the same directory. This file will be swapped out for evaluation.
        """
        return NotImplemented


class Generator(AbstractGenerator):
    def __init__(self):
        pass

    def generate(self, target_class: type):
        ###################################
        ### IMPLEMENT YOUR SOLUTION HERE ##
        ###################################
        generation = 1
        while True:
            # select, breed, mutate
            front = ["ConcreteCandidate1", "ConcreteCandidate2", "ConcreteCandidate3", ]
            # yield to act like a iterable for interface as used in provided test cases
            yield front
            generation += 1


##### EXAMPLE CODE TO GET STARTED

def get_methods(class_ref):
    methods = [method for method in dir(class_ref) if not method.startswith("__") or "__init__" in method]
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

    def __init__(self, method_name, method_metadata, ercs, lower, upper):
        self.cov = None
        self.m_dict = method_metadata
        self.m_name = method_name
        self.params = {}
        self.lower = lower
        self.upper = upper
        self.result = None
        for param, p_type in self.m_dict["params"].items():
            # TODO: handle types here
            print("Processing", p_type, "parameter")
            self.params[param] = 5

    def is_init(self):
        return self.m_name == "__init__"

    def run(self, obj):
        args = {k: v for k, v in self.params.items()}
        if self.is_init():
            # obj is Class
            with Coverage() as cov:
                try:
                    result = obj(**args)
                except:
                    result = MethodCall.FAILED
        else:
            with Coverage() as cov:
                try:
                    result = getattr(obj, self.m_name)(**args)
                except:
                    result = MethodCall.FAILED
        self.result = result
        del cov.original_trace_function
        self.cov = cov
        return cov, result

    def run_raw(self, obj):
        args = {k: v.get() for k, v in self.params.items()}
        if self.is_init():
            # obj is Class
            result = obj(**args)
        else:
            result = getattr(obj, self.m_name)(**args)
        return result

    def branch_coverage_fitness(self, ignore_methods=None):
        trace = [(method, line) for method, line in self.cov.trace() if method not in ignore_methods]
        pairs = branch_coverage(trace)
        return len(pairs), pairs

    def line_coverage_fitness(self, ignore_methods=None):
        lines = [(method, line) for method, line in self.cov.coverage() if method not in ignore_methods]
        return len(lines), lines

    def get_line(self):
        return self.__repr__()

    def __repr__(self):
        s = "{}({})".format(self.m_name,
                            ", ".join(["{} = {}".format(p_name, p_val) for p_name, p_val in self.params.items()]))
        return s
