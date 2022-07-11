import unittest
import time
from submission.generator import Generator
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from submission import doesitwork

unittest.TestLoader.sortTestMethodsUsing = None


class GenerationTestExtended(unittest.TestCase):

    def test_generation_extended(self):
        timeout = 100
        target_class = doesitwork.DoesItWork
        run_with_time_limit(target_class, timeout=timeout, plots=True)


def run_with_time_limit(target_class: type, timeout: float = 30, plots=False):
    start = time.time()
    solution_iterator = Generator().generate(target_class)
    result = None
    history = []
    while time.time() - start < timeout:
        last_result = result
        # if last_result:
        result = next(solution_iterator)
        history.append(result)
    result = last_result
    history = history[:-1]
    if plots: plot_history(history)
    return result


def plot_history(history):
    first_pop = history[0]
    objectives = list(first_pop[0].fitness)
    plot_tuples = []
    for generation_number, archive in enumerate(history):
        tuple_list = [(generation_number, *list(ind.fitness.values())) for ind in archive]
        plot_tuples.extend(tuple_list)
    cols = ["generation", *objectives]
    df = pd.DataFrame(plot_tuples, columns=cols)
    df = df.sort_values(by=["generation"])
    print(df)
    sns.scatterplot(data=df[df["generation"] != df["generation"].max()], x=objectives[0], y=objectives[1], hue="generation", palette="Blues_r")
    sns.scatterplot(data=df[df["generation"] == df["generation"].max()], x=objectives[0], y=objectives[1], hue="generation", palette="bright")
    plt.show()


if __name__ == '__main__':
    unittest.main()
