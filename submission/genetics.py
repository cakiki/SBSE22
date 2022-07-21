import itertools
from tkinter import N
from submission.generator import AbstractCandidate


def dominates(
    candidate_1: AbstractCandidate, candidate_2: AbstractCandidate
) -> bool:
    if (
        candidate_1.get_cost() == candidate_2.get_cost()
        and candidate_1.get_coverage() == candidate_2.get_coverage()
    ):
        return False

    dominates_cost = candidate_1.get_cost() <= candidate_2.get_cost()
    dominates_coverage = candidate_1.get_coverage() >= candidate_2.get_coverage()

    return dominates_cost and dominates_coverage

def calculate_pareto_ranks(candidates: List[AbstractCandidate]) -> None:
    for candidate in candidates:
        candidate.pareto_rank = None
    
    n = 1
    remaining_candidates = candidates

    while remaining_candidates:
        is_dominated = []
        for challenger in remaining_candidates:
            for challenged in remaining_candidates:
                if dominates(challenger, challenged):
                    is_dominated.append(challenged)
        
        remaining_candidates = []
        for candidate in candidates:
            if candidate not in is_dominated and candidate.pareto_front is None:
                candidate.pareto_rank = n
            else:
                remaining_candidates.append((candidate))
            
        n += 1
