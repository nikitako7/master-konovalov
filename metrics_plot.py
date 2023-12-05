import matplotlib.pyplot as plt
import seaborn as sns

def precision_recall_f1_curves_(precision_df, recall_df, f1_score_df, x='k', y_pr='avg_precision', y_recall='avg_recall', y_f1='avg_f1_score'):
  fig = plt.figure(figsize=(8, 6))
  sns.set(style='darkgrid')
  sns.lineplot(x=x, y=y_pr, data=precision_df, color='red', label='precision@k')
  sns.lineplot(x=x, y=y_recall, data=recall_df, color='blue', label='recall@k')
  sns.lineplot(x=x, y=y_f1, data=f1_score_df, color='green', label='f1_score@k')
  plt.ylim(0, 1)
  plt.legend()
  plt.title("Precision@k, Recall@k , F1_Score@k Curves", fontsize=14)
  plt.show()