import json
import os
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

def read_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def evaluate_attributions(dir1, json_file, dir2):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    all_preds, all_gts = [], []

    y_true_x, y_pred_x = [], []
    
    for q_id, _ in data.items():
        # if not q_id.startswith("wiki"):
        #     continue
        # Example path: ./output/chartpathagent_run/wiki00689_6/agent_response.json
        agent_response_path = os.path.join(dir1, q_id, "agent_response.json")
        
        if not os.path.exists(agent_response_path):
            # print("X")
            # print(agent_response_path)
            continue
        
        agent_response = read_json(agent_response_path)
        list_attributions = agent_response.get("attributed_nodes", "")
        list_attributions = list_attributions.replace(" ", "").split(",")

        
        
        json2_path = os.path.join(dir2, data.get(q_id).get("key"), "seg2annotations.json")
        
        if not os.path.exists(json2_path):
            print("Y")
            continue
        
        json2 = read_json(json2_path)
        list_attributions = [json2[x] if x in json2 else None for x in list_attributions]
        list_attributions = [x for x in list_attributions if x is not None]

        if len(list_attributions) == 0:
            continue

        
        
        ground_truth = data.get(q_id).get("attribution")

        print(q_id, list_attributions, ground_truth)
        
        all_preds.append(set(list_attributions))
        all_gts.append(set(ground_truth))

        y_true_x.extend([1 if key in ground_truth else 0 for key in list_attributions])
        y_pred_x.extend([1] * len(list_attributions))
    
    # Convert to binary format for precision_recall_fscore_support
    all_labels = list(set().union(*all_preds, *all_gts))

    y_true = [[1 if lbl in gt else 0 for lbl in all_labels] for gt in all_gts]
    y_pred = [[1 if lbl in pred else 0 for lbl in all_labels] for pred in all_preds]

    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro'
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )

    print(len(all_gts))
    
    print(f"Micro Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")

    micro = precision_recall_fscore_support(y_true_x, y_pred_x, average="micro")
    macro = precision_recall_fscore_support(y_true_x, y_pred_x, average="macro")

    print(micro)
    print(macro)
    
    return {
        "micro": {"precision": precision_micro, "recall": recall_micro, "f1": f1_micro},
        "macro": {"precision": precision_macro, "recall": recall_macro, "f1": f1_macro}
    }

# Example usage (uncomment to run):
# results = evaluate_attributions("./output/chartpathagent_run", "./data/dataset.json", "./data/images")
