import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve

def evaluate_hits(evaluator, pos_pred, neg_pred, k_list):
    results = {}
    for K in k_list:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']
        # test_hits = evaluator.eval({
        #     'y_pred_pos': pos_test_pred,
        #     'y_pred_neg': neg_test_pred,
        # })[f'hits@{K}']

        hits = round(hits, 4)
        # test_hits = round(test_hits, 4)

        results[f'Hits@{K}'] = hits

    return results
        


def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred):
    
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    # neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    
    mrr_output =  eval_mrr(pos_val_pred, neg_val_pred)

    valid_mrr =mrr_output['mrr_list'].mean().item()
    valid_mrr_hit1 = mrr_output['hits@1_list'].mean().item()
    valid_mrr_hit3 = mrr_output['hits@3_list'].mean().item()
    valid_mrr_hit10 = mrr_output['hits@10_list'].mean().item()

    valid_mrr_hit20 = mrr_output['hits@20_list'].mean().item()
    valid_mrr_hit50 = mrr_output['hits@50_list'].mean().item()
    valid_mrr_hit100 = mrr_output['hits@100_list'].mean().item()

    valid_mrr = round(valid_mrr, 4)
    # test_mrr = round(test_mrr, 4)
    valid_mrr_hit1 = round(valid_mrr_hit1, 4)
    valid_mrr_hit3 = round(valid_mrr_hit3, 4)
    valid_mrr_hit10 = round(valid_mrr_hit10, 4)

    valid_mrr_hit20 = round(valid_mrr_hit20, 4)
    valid_mrr_hit50 = round(valid_mrr_hit50, 4)
    valid_mrr_hit100 = round(valid_mrr_hit100, 4)
    
    results = {}
    results['mrr_hit1'] = valid_mrr_hit1
    results['mrr_hit3'] = valid_mrr_hit3
    results['mrr_hit10'] = valid_mrr_hit10

    results['MRR'] = valid_mrr

    results['mrr_hit20'] = valid_mrr_hit20
    results['mrr_hit50'] = valid_mrr_hit50
    results['mrr_hit100'] = valid_mrr_hit100

    return results




def evaluate_auc(pred, true):
    results = {}

    auc = roc_auc_score(true, pred)
    auc = round(auc, 4)

    ap = average_precision_score(true, pred)    
    ap = round(ap, 4)

    results['AUC'] = auc
    results['AP'] = ap

    return results


def evaluate_precision_recall(pred, true):
    """
    Evaluate Precision and Recall.
    Ensure correct data types and handle NaN issues.
    """

    results = {}

    true = true.cpu().numpy().astype(int)
    pred = pred.cpu().numpy().astype(float)

    precision, recall, thresholds = precision_recall_curve(true, pred)

    if len(precision) == 0 or len(recall) == 0:
        return {'Precision': 0.0, 'Recall': 0.0}

    f1_scores = np.nan_to_num(2 * (precision * recall) / (precision + recall + 1e-8))

    best_idx = np.argmax(f1_scores)
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]

    results['Precision'] = round(best_precision, 4)
    results['Recall'] = round(best_recall, 4)

    return results


def evaluate_f1(pred, true):
    """
    Evaluate F1 score.
    """

    pred = (pred > 0.5).int()
    true = true.int()

    f1 = f1_score(true.cpu().numpy(), pred.cpu().numpy())
    results = {'F1': round(f1, 4)}

    return results


def eval_mrr(y_pred_pos, y_pred_neg):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''


    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)

    hits20_list = (ranking_list <= 20).to(torch.float)
    hits50_list = (ranking_list <= 50).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    hits100_list = (ranking_list <= 100).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)

    return { 'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@20_list': hits20_list,
                'hits@50_list': hits50_list,
                'hits@10_list': hits10_list,
                'hits@100_list': hits100_list,
                'mrr_list': mrr_list}




def eval_hard_negs(pos_pred, neg_pred, k_list):
    """
    Eval on hard negatives
    """
    neg_pred = neg_pred.squeeze(-1)

    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (neg_pred >= pos_pred).sum(dim=-1)

    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (neg_pred > pos_pred).sum(dim=-1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    results = {}
    for k in k_list:
        mean_score = (ranking_list <= k).to(torch.float).mean().item()
        results[f'Hits@{k}'] = round(mean_score, 4)

    mean_mrr = 1./ranking_list.to(torch.float)
    results['MRR'] = round(mean_mrr.mean().item(), 4)

    return results