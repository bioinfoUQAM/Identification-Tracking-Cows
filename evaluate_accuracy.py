import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / (boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0

def normalize_name(name):
    return str(name).strip().lower().replace("_", " ")

def fix_gt_column_order_safe(gt_df: pd.DataFrame) -> pd.DataFrame:
    # Corrige se vier como x1, x2, y1, y2
    cols = gt_df.columns.tolist()
    coord_cols = ["x1", "x2", "y1", "y2"]
    if all(col in cols for col in coord_cols):
        if cols.index("x2") < cols.index("y1"):
            # Swap as colunas
            gt_df = gt_df.rename(columns={"x2": "y1", "y1": "x2"})[gt_df.columns]
    return gt_df

def evaluate_with_iou(pred_csv_path, gt_csv_path, iou_threshold=0.5, output_csv=None):
    df_pred = pd.read_csv(pred_csv_path)
    df_gt = pd.read_csv(gt_csv_path)

    df_pred.columns = df_pred.columns.str.strip().str.replace('\ufeff', '', regex=False)
    df_gt.columns = df_gt.columns.str.strip().str.replace('\ufeff', '', regex=False)

    if "pred_name" in df_pred.columns:
        df_pred.rename(columns={"pred_name": "predicted"}, inplace=True)
    if "name" in df_gt.columns:
        df_gt.rename(columns={"name": "true"}, inplace=True)

    df_pred["predicted"] = df_pred["predicted"].map(normalize_name)
    df_gt["true"] = df_gt["true"].map(normalize_name)

    # 🔧 Corrige possíveis erros de ordem de coordenadas no GT
    df_gt = fix_gt_column_order_safe(df_gt)

    results = []
    grouped_gt = df_gt.groupby("frame")
    grouped_pred = df_pred.groupby("frame")
    #all_frames = sorted(set(df_gt["frame"]).union(set(df_pred["frame"]))) #considera todos os frames de ambos, criando falsos negativos em frames sem detecção.
    #all_frames = sorted(set(df_gt["frame"]).intersection(set(df_pred["frame"]))) #compara somente os frames que estão nos dois CSVs → onde houve predição e ground truth.
    # 🧠 Considerar apenas frames com pelo menos um GT com 'true' válido
    valid_gt_frames = df_gt[df_gt["true"].notnull() & (df_gt["true"] != "?")]["frame"].unique()
    all_frames = sorted(set(valid_gt_frames).intersection(set(df_pred["frame"].unique())))
    

    for frame in all_frames:
        gts = grouped_gt.get_group(frame) if frame in grouped_gt.groups else pd.DataFrame()
        preds = grouped_pred.get_group(frame) if frame in grouped_pred.groups else pd.DataFrame()
        gt_used = set()
        for _, pred_row in preds.iterrows():
            pred_box = [pred_row["x1"], pred_row["y1"], pred_row["x2"], pred_row["y2"]]
            best_iou = 0
            best_gt_idx = None
            best_name_match = False
            for gt_idx, gt_row in gts.iterrows():
                if gt_idx in gt_used:
                    continue
                gt_box = [gt_row["x1"], gt_row["y1"], gt_row["x2"], gt_row["y2"]]
                iou = compute_iou(pred_box, gt_box)
                name_match = pred_row["predicted"] == gt_row["true"]
                if name_match and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_name_match = True
                    break
                elif not best_name_match and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx is not None and best_iou >= iou_threshold:
                matched_gt = gts.loc[best_gt_idx]
                correct = pred_row["predicted"] == matched_gt["true"]
                gt_used.add(best_gt_idx)
                results.append({
                    "frame": frame,
                    "true": matched_gt["true"],
                    "predicted": pred_row["predicted"],
                    "iou": best_iou,
                    "correct": correct
                })
            else:
                results.append({
                    "frame": frame,
                    "true": None,
                    "predicted": pred_row["predicted"],
                    "iou": best_iou,
                    "correct": False
                })

    result_df = pd.DataFrame(results)
    #eval_df = result_df[result_df["true"].notnull()]
    eval_df = result_df[result_df["true"].notnull() & (result_df["true"].astype(str).str.strip() != "")]

    total_accuracy = eval_df["correct"].mean()
    accuracy_per_class = eval_df.groupby("true")["correct"].mean().sort_values(ascending=False)

    y_true = eval_df["true"]
    y_pred = eval_df["predicted"]
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"✅ Acurácia total (IoU ≥ {iou_threshold}): {total_accuracy:.4f}")
    print(f"📈 Precisão: {precision:.4f}")
    print(f"📈 Recall:   {recall:.4f}")
    print(f"📈 F1-Score: {f1:.4f}\n")
    print("📊 Acurácia por vaca:")
    print(accuracy_per_class)

    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"📄 Resultados detalhados salvos em: {output_csv}")

    return total_accuracy, accuracy_per_class, precision, recall, f1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Avalia acurácia priorizando IoU + nome (com nomes normalizados)")
    parser.add_argument('--pred', required=True, help="CSV com predições")
    parser.add_argument('--gt', required=True, help="CSV com ground truth")
    parser.add_argument('--save', help="Caminho para salvar o CSV detalhado")
    parser.add_argument('--iou-thresh', type=float, default=0.5, help="IoU mínimo para considerar uma correspondência")
    args = parser.parse_args()

    evaluate_with_iou(args.pred, args.gt, args.iou_thresh, args.save)
#memlhor:

