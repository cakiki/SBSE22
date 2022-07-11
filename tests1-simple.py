import unittest
from submission.generator import Generator, AbstractCandidate
from submission import doesitwork

unittest.TestLoader.sortTestMethodsUsing = None


class GenerationTest(unittest.TestCase):

    def test_generator_is_callable(self):
        candidate = run_single_generation(doesitwork.DoesItWork)

    def test_get_candidate(self):
        candidate = get_ind_from_pop()
        self.assertIsInstance(candidate, AbstractCandidate)

    def test_get_candidate_cost(self):
        candidate = get_ind_from_pop()
        cost = candidate.get_cost()
        self.assertTrue(cost >= 0, "Cost should be positive!")

    def test_get_candidate_coverage(self):
        candidate = get_ind_from_pop()
        cov = candidate.get_coverage()
        self.assertTrue(cov >= 0, "Cost should be positive!")

    def test_get_candidate_code(self):
        candidate = get_ind_from_pop()
        code = candidate.get_lines_of_code()
        self.assertIsInstance(code, str)
        self.assertTrue(len(code) >= 0, "Test should not be empty")

    def test_run_candidate(self):
        candidate = get_ind_from_pop()
        candidate.run()


def run_single_generation(target_class: type, ):
    solution_iterator = Generator().generate(target_class)
    result = next(solution_iterator)
    return result


def get_ind_from_pop():
    target_class = doesitwork.DoesItWork
    result = run_single_generation(target_class)
    candidate: AbstractCandidate = result[0]
    return candidate


if __name__ == '__main__':
    unittest.main()
