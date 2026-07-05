from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

from heca.helper.descriptions import TASKS_V1, TASKS_V2

ALPHABET = ["A", "B", "C"]
choice_template = "{choice}: {state}"
prompt_template = "{question}\n{choices}\nAnswer with exactly one of the letters {letters} corresponding to the correct answer."


def get_choices_description(classes: list[str]) -> str:
    return "\n".join([f"{choice}: {state}" for choice, state in zip(ALPHABET, classes)])


def get_choices(classes: list[str]) -> str:
    return f"[{', '.join(ALPHABET[:len(classes)])}]"


# def get_prompt(entity: str, data: dict) -> str:
#     question = TASKS_V1[entity]["question"]
#     classes = TASKS_V1[entity]["classes"]
#     choices_with_text = get_choices_description(classes)
#     letters = get_letters(len(classes))

#     return prompt_template.format(
#         question=question, choices=choices_with_text, letters=letters
#     )


def get_properties_description(properties: dict[str, str]) -> str:
    return "\n".join([f"- {key}: {value}" for key, value in properties.items()])


def get_prompt(entity: str, data: dict) -> str:
    properties = get_properties_description(data["properties"])
    missing_property = data["missing_property"]
    question = f"Given the following properties of the {entity}:\n{properties}\n What is the {missing_property} of the {entity}?"
    task = f"You can choose from the following options for the {missing_property}:\n{get_choices_description(data['classes'])}\nAnswer with exactly one of the letters {get_choices(data['classes'])} corresponding to the correct answer."
    return f"{question}\n{task}"


def get_letters(states: int) -> set[str]:
    return set(ALPHABET[:states])


def get_data() -> (
    tuple[list[Image.Image], list[str], list[str], list[str], dict[str, set[str]]]
):
    images = []
    prompts = []
    ground_truth = []
    task_names = []
    letters_per_task = {}

    # Build dataset
    for entity, data in TASKS_V2.items():
        prompt = get_prompt(entity, data)
        print(f"Prompt for {entity}:\n{prompt}\n")
        for state in data["states"]:
            for i in range(10):
                dir = "../data/samples/front"
                image = Image.open(f"{dir}/{entity}_{state}_{i}.png").convert("RGB")
                images.append(image)
                prompts.append(prompt)
                ground_truth.append(state)
                task_names.append(entity)
                letters_per_task[entity] = set(ALPHABET[: len(data["states"])])
    return images, prompts, ground_truth, task_names, letters_per_task


def plot_predictions(
    predicted_letters: list[str], ground_truth: list[str], task_names: list[str]
):
    # Evaluate
    correct = 0
    correct_per_task = {entity: 0 for entity in TASKS_V1}
    wrong_per_task = {entity: 0 for entity in TASKS_V1}
    for i in range(len(predicted_letters)):
        entity = task_names[i]
        states = TASKS_V1[entity]["states"]
        letter_to_state = {ALPHABET[j]: state for j, state in enumerate(states)}
        predicted_state = letter_to_state.get(predicted_letters[i])

        is_ok = predicted_state == ground_truth[i]
        correct += int(is_ok)
        correct_per_task[entity] += int(is_ok)
        wrong_per_task[entity] += int(not is_ok)

        print(
            f"TASK={entity:<10} "
            f"GT={ground_truth[i]:<10} "
            f"PRED={predicted_state:<10} "
            f"LETTER={predicted_letters[i]} "
        )

    acc = correct / max(len(predicted_letters), 1)
    print(f"\nAccuracy: {acc:.3f} ({correct}/{len(predicted_letters)})")
    for entity in TASKS_V1:
        task_acc = correct_per_task[entity] / max(
            correct_per_task[entity] + wrong_per_task[entity], 1
        )
        print(
            f"TASK={entity:<10} Accuracy: {task_acc:.3f} ({correct_per_task[entity]}/{correct_per_task[entity] + wrong_per_task[entity]})"
        )


def plot_probabilities(predictions, task_names, ground_truth, probs_predicted):
    grouped = defaultdict(list)

    for entity, gt, probs in zip(task_names, ground_truth, probs_predicted):
        grouped[entity].append((gt, probs))

    for entity, samples in grouped.items():
        labels = list(samples[0][1].keys())  # ["A", "B", "C"]
        x = np.arange(len(samples))
        bottoms = np.zeros(len(samples))
        fig, ax = plt.subplots(figsize=(12, 5))
        for label in labels:
            values = [probs[label] for _, probs in samples]
            ax.bar(
                x,
                values,
                bottom=bottoms,
                label=label,
            )

            bottoms += np.array(values)

        # x-axis labels = ground truth labels
        gt_labels = [gt for gt, _ in samples]

        ax.set_xticks(x)
        ax.set_xticklabels(gt_labels)

        ax.set_ylim(0, 1)

        ax.set_title(entity)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Samples (GT label)")
        ax.legend()

        plt.tight_layout()
        plt.show()


get_data()
