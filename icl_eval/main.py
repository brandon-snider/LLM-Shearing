import argparse
import json
import logging
import os
from lm_eval import tasks, evaluator, utils
from lm_eval.tasks import TaskManager
from lm_eval.models.utils import MultiChoice

logging.getLogger("openai").setLevel(logging.WARNING)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            # Fallback for any non-serializable object
            return str(obj)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    task_manager = TaskManager()

    parser.add_argument("--tasks", default=None, choices=MultiChoice(task_manager.all_tasks))
    # parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    # parser.add_argument("--data_sampling", type=float, default=None)
    # parser.add_argument("--no_cache", action="store_true")
    # parser.add_argument("--decontamination_ngrams_path", default=None)
    # parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    task_manager = TaskManager()

    if args.tasks is None:
        task_names = task_manager.all_tasks
    else:
        task_names = utils.pattern_match(args.tasks.split(","), task_manager.all_tasks)

    print(f"Selected Tasks: {task_names}")

    # description_dict = {}
    # if args.description_dict_path:
    #     with open(args.description_dict_path, "r") as f:
    #         description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        # no_cache=args.no_cache,
        limit=args.limit,
        # description_dict=description_dict,
        # decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        # output_base_path=args.output_base_path,
    )

    dumped = json.dumps(results, indent=2, cls=CustomJSONEncoder)

    # dumped = json.dumps(results, indent=2)
    # print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    # print(
    #     f"{args.model} ({args.model_args}), limit: {args.limit}, "
    #     f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    # )
    print(utils.make_table(results))

    if args.output_base_path:
        output_file_path = os.path.join(args.output_base_path, "results_table.txt")
        with open(output_file_path, "w") as file:
            file.write(utils.make_table(results))

if __name__ == "__main__":
    main()