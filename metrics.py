import pandas as pd

def get_precision_recall_f1_score(k_max, threshold_rating, user_pred_actual):
  threshold = threshold_rating
  precision = dict()
  recall = dict()
  f1_score = dict()
  avg_precision = dict()
  avg_recall = dict()
  avg_f1_score = dict()

  for k in range(1,k_max+1):
      for user_id, user_ratings in user_pred_actual.items():

          user_ratings.sort(key=lambda x: x[0], reverse=True)

          count_relevant = sum((actual_rating >= threshold) for (predicted_rating, actual_rating) in user_ratings)

          count_recommended_k = sum((predicted_rating >= threshold) for (predicted_rating, actual_rating) in user_ratings[:k])

          count_relevant_and_recommended_k = sum(
              ((actual_rating >= threshold) and (predicted_rating >= threshold))
              for (predicted_rating, actual_rating) in user_ratings[:k]
          )

          precision[user_id] = count_relevant_and_recommended_k /count_recommended_k if count_recommended_k != 0 else 0
          recall[user_id] = count_relevant_and_recommended_k /count_relevant if count_relevant != 0 else 0
          f1_score[user_id] = 2 * ((precision[user_id] * recall[user_id])/(precision[user_id]+recall[user_id])) if (precision[user_id]+recall[user_id]) != 0 else 0

      avg_precision[k] = sum(prec for prec in precision.values()) / len(precision)
      avg_recall[k] = sum(rec for rec in recall.values()) / len(recall)
      avg_f1_score[k] = sum(f1 for f1 in f1_score.values()) / len(f1_score)

  avg_recall_df = pd.DataFrame(list(avg_recall.items()),columns = ['k','avg_recall'])
  avg_precision_df = pd.DataFrame(list(avg_precision.items()),columns = ['k','avg_precision'])
  avg_f1_score_df = pd.DataFrame(list(avg_f1_score.items()),columns = ['k','avg_f1_score'])

  return avg_precision_df, avg_recall_df, avg_f1_score_df