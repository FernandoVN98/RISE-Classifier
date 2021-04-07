import os
from argparse import ArgumentParser

from preprocess.read_dataset import Preprocessing
from risealgorithm.rise import rise_algorithm
from ruleswriter.rule_writer import rule_writer
from rulesinterpreter.rule_interpreter import rule_interpreter
import time

def main():
    parser = ArgumentParser(description="Work 1 - SEL")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["glass", "Montgomery-dataset", "studentInfo"],
        help="Select dataset to use",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["arff", "csv"],
        help="Type of dataset",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file",
        required=True,
    )
    parser.add_argument(
        "-q",
        "--q",
        type=int,
        default=1,
        help="Q value, optional to set, default = 1",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--s",
        type=int,
        default=1,
        help="S value, optional to set, default = 1. It defines the distance.",
        required=False,
    )

    args = parser.parse_args()

    print(
        "Creating execution with: 'dataset={}'".format(
            args.dataset
        )
    )

    rise = rise_algorithm(s=args.s, q=args.q)
    preproc = Preprocessing(args.dataset, os.path.dirname(os.path.abspath(__file__)))
    X, Y, X_val, Y_val, X_test, Y_test = preproc.get_dataset(flag=args.type)
    start = time.time()
    my_rise_algorithm = rise.extract_rules(X.to_numpy(), Y, X.columns.tolist())
    end = time.time()
    print("The extraction of rules lasts: "+str(end - start))
    start = time.time()
    print("Validation Accuracy: " + str(my_rise_algorithm.validation(X_val.to_numpy(), Y_val)))
    end = time.time()
    print("The validation of rules lasts: " + str(end - start))
    rule_writ = rule_writer(args.output, my_rise_algorithm.rs)
    rule_writ.write()
    rule_inter = rule_interpreter(args.output, svdm=my_rise_algorithm.svdm, s=args.s, q=args.q)
    rule_inter.extract_rules()
    start = time.time()
    print("Test Accuracy: " +str(rule_inter.evaluate_rules(X_test.to_numpy(), Y_test, max=my_rise_algorithm.max_values, min=my_rise_algorithm.min_values)))
    end = time.time()
    print("The test of rules lasts: " + str(end - start))


if __name__ == "__main__":
    main()