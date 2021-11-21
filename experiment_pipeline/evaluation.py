import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    mean_absolute_error, 
    r2_score, max_error
)

def is_crowded(stop_ids, passenger_counts, stop_stats, num_stds):
    crowded = []
    for (stop_id, passenger_count) in zip(stop_ids, passenger_counts):
      threshold = stop_stats[('passenger_count', 'mean')].loc[stop_id] + num_stds * stop_stats[('passenger_count', 'std')].loc[stop_id]
      if passenger_count > threshold:
        crowded.append(1)
      else:
        crowded.append(0)
    return crowded

class Evaluation:
  def __init__(self, global_feature_set=None, train=None, val=None, test=None, stop_dict=None):
    self.global_feature_set = global_feature_set
    self.train = train
    self.val = val
    self.test = test
    self.stop_dict = stop_dict

    # self.stop_stats = global_feature_set[
    #     ['next_stop_id', 'passenger_count']
    # ].groupby('next_stop_id').agg({'passenger_count':['mean', 'std']})
  
  def basic_eval(self, data, dir=None, segment=None):
    if data == 'train':
      df = self.train.copy()
    elif data == 'val':
      df = self.val.copy()
    elif data == 'test':
      df = self.test.copy()

    if dir:
      df = df[df['direction'] == dir]

    if segment:
      if type(segment) == str:
        next_stop_id = segment
        assert next_stop_id in set(self.stop_dict[dir])
        df = df[df['next_stop_id'] == next_stop_id]
      elif type(segment) == int:
        next_stop_id = self.stop_dict[dir][segment]
        df = df[df['next_stop_id'] == next_stop_id]

    gt = df['passenger_count']
    gt_mean = np.zeros_like(gt) + gt.mean()
    pred = df['passenger_count_pred']

    mae_pred = mean_absolute_error(gt, pred)
    max_error_pred = max_error(gt, pred)
    r2_pred = r2_score(gt, pred)

    mae_mean = mean_absolute_error(gt, gt_mean)
    max_error_mean = max_error(gt, gt_mean)
    r2_mean = r2_score(gt, gt_mean)

    print('Performance: Model Prediction')
    print(f'MAE: {mae_pred:.1f}')
    print(f'ME : {max_error_pred:.1f}')
    print(f'R^2: {r2_pred:.2f}')
    print('\n')
    print('Performance: Mean Prediction')
    print(f'MAE: {mae_mean:.1f}')
    print(f'ME : {max_error_mean:.1f}')
    print(f'R^2: {r2_mean:.2f}')
    
  def plot_passenger_count_by_time_of_day(self, data, dir=None, segment=None, agg='sum'):
    if data == 'train':
      df = self.train.copy()
    elif data == 'val':
      df = self.val.copy()
    elif data == 'test':
      df = self.test.copy()

    if dir:
      df = df[df['direction'] == dir]

    if segment:
      if type(segment) == str:
        next_stop_id = segment
        assert next_stop_id in set(self.stop_dict[dir])
        df = df[df['next_stop_id'] == next_stop_id]
      elif type(segment) == int:
        next_stop_id = self.stop_dict[dir][segment]
        df = df[df['next_stop_id'] == next_stop_id]

    hours = list(range(24))
    df['day_type'] = df['timestamp'].apply(lambda x: 'weekday' if x.dayofweek < 5 else 'weekend')

    if agg == 'sum':
      gt = df.groupby([df['timestamp'].dt.hour, 'day_type'])['passenger_count'].sum().unstack()
      pred = df.groupby([df['timestamp'].dt.hour, 'day_type'])['passenger_count_pred'].sum().unstack()

      if ('weekday' in set(gt.columns)) and ('weekday' in set(pred.columns)):
        gt_weekday = gt['weekday']
        pred_weekday = pred['weekday']

        plt.figure(figsize=(20, 10))
        plt.plot(gt.index, gt_weekday, label='Ground Truth', color='darkorange')
        plt.plot(pred.index, pred_weekday, label='Prediction', color='navy')
        plt.xticks(hours)
        plt.xlabel('Time of Day')
        plt.ylabel('Passenger Count')
        plt.title('Weekday')
        plt.legend()
        plt.show()

      if ('weekend' in set(gt.columns)) and ('weekend' in set(pred.columns)):  
        gt_weekend = gt['weekend']
        pred_weekend = pred['weekend']

        plt.figure(figsize=(20, 10))
        plt.plot(gt.index, gt_weekend, label='Ground Truth', color='darkorange')
        plt.plot(pred.index, pred_weekend, label='Prediction', color='navy')
        plt.xticks(hours)
        plt.xlabel('Time of Day')
        plt.ylabel('Passenger Count')
        plt.title('Weekend')
        plt.legend()
        plt.show()
      
    elif agg == 'mean':
      gt_avg = df.groupby([df['timestamp'].dt.hour, 'day_type'])['passenger_count'].mean().unstack()
      gt_std = df.groupby([df['timestamp'].dt.hour, 'day_type'])['passenger_count'].std().unstack()
      pred_avg = df.groupby([df['timestamp'].dt.hour, 'day_type'])['passenger_count_pred'].mean().unstack()
      pred_std = df.groupby([df['timestamp'].dt.hour, 'day_type'])['passenger_count_pred'].std().unstack()

      if (('weekday' in set(gt_avg.columns)) and ('weekday' in set(gt_std.columns))) and (('weekday' in set(pred_avg.columns)) and ('weekday' in set(pred_std.columns))):
        gt_weekday_avg = gt_avg['weekday']
        gt_weekday_std = gt_std['weekday']
        pred_weekday_avg = pred_avg['weekday']
        pred_weekday_std = pred_std['weekday']
        plt.figure(figsize=(20, 10))
        plt.plot(gt_avg.index, gt_weekday_avg, label='Ground Truth', color='darkorange')
        plt.fill_between(gt_avg.index, gt_weekday_avg - gt_weekday_std, gt_weekday_avg + gt_weekday_std, alpha=0.2, color='darkorange', lw=2)
        plt.plot(pred_avg.index, pred_weekday_avg, label='Prediction', color='navy')
        plt.fill_between(pred_avg.index, pred_weekday_avg - pred_weekday_std, pred_weekday_avg + pred_weekday_std, alpha=0.2, color='navy', lw=2)
        plt.xticks(hours)
        plt.xlabel('Time of Day')
        plt.ylabel('Passenger Count')
        plt.title('Weekday')
        plt.legend()
        plt.show()

      if (('weekend' in set(gt_avg.columns)) and ('weekend' in set(gt_std.columns))) and (('weekend' in set(pred_avg.columns)) and ('weekend' in set(pred_std.columns))):  
        gt_weekend_avg = gt_avg['weekend']
        gt_weekend_std = gt_std['weekend']
        pred_weekend_avg = pred_avg['weekend']
        pred_weekend_std = pred_std['weekend']
        plt.figure(figsize=(20, 10))
        plt.plot(gt_avg.index, gt_weekend_avg, label='Ground Truth', color='darkorange')
        plt.fill_between(gt_avg.index, gt_weekend_avg - gt_weekend_std, gt_weekend_avg + gt_weekend_std, alpha=0.2, color='darkorange', lw=2)
        plt.plot(pred_avg.index, pred_weekend_avg, label='Prediction', color='navy')
        plt.fill_between(pred_avg.index, pred_weekend_avg - pred_weekend_std, pred_weekend_avg + pred_weekend_std, alpha=0.2, color='navy', lw=2)
        plt.xticks(hours)
        plt.xlabel('Time of Day')
        plt.ylabel('Passenger Count')
        plt.title('Weekend')
        plt.legend()
        plt.show()

  def gt_pred_scatter(self, data, dir=None, eps=0.1, s=2, lower=None, upper=None):
    if data == 'train':
      df = self.train.copy()
    elif data == 'val':
      df = self.val.copy()
    elif data == 'test':
      df = self.test.copy()

    df = df[df['direction'] == dir]
    stop_ids_route = self.stop_dict[dir]
    stop_pos_route = list(range(len(stop_ids_route)))

    df['pred_to_gt_ratio'] = (df['passenger_count_pred'] + eps) / (df['passenger_count'] + eps)

    plt.figure(figsize=(20, 10))

    # model predictions too high (plot gt markers on top of pred markers)
    over_est_df = df[df['pred_to_gt_ratio'] >= 1]
    over_est_stop_pos_obs = over_est_df['next_stop_id_pos']
    over_est_gt = over_est_df['passenger_count']
    over_est_ratios = over_est_df['pred_to_gt_ratio']
    over_est_ratios = over_est_ratios.clip(lower=lower, upper=upper)
    over_est_ss = [s * ratio ** 2 for ratio in over_est_ratios]
    plt.scatter(over_est_stop_pos_obs, over_est_gt, s=over_est_ss, marker='o', label='Prediction', color='navy')
    plt.scatter(over_est_stop_pos_obs, over_est_gt, s=s, marker='o', label='Ground Truth', color='darkorange')

    # model predictions too low (plot pred markers on top of gt markers)
    under_est_df = df[df['pred_to_gt_ratio'] < 1]
    under_est_stop_pos_obs = under_est_df['next_stop_id_pos']
    under_est_gt = under_est_df['passenger_count']
    under_est_ratios = under_est_df['pred_to_gt_ratio']
    under_est_ratios = under_est_ratios.clip(lower=lower, upper=upper)
    under_est_ss = [s * ratio ** 2 for ratio in under_est_ratios]
    plt.scatter(under_est_stop_pos_obs, under_est_gt, s=s, marker='o', color='darkorange')
    plt.scatter(under_est_stop_pos_obs, under_est_gt, s=under_est_ss, marker='o', color='navy')

    plt.xticks(stop_pos_route, stop_ids_route, rotation=90)
    plt.xlabel('Stop')
    plt.ylabel('Ground Truth Passenger Count')
    legend = plt.legend()
    for handle in legend.legendHandles:
      handle.set_sizes([s])
    plt.show()


  # def print_classification_metrics(self, data, dir=None, segment=None, num_stds=1):
  #   if data == 'train':
  #     df = self.train.copy()
  #   elif data == 'val':
  #     df = self.val.copy()
  #   elif data == 'test':
  #     df = self.test.copy()

  #   if dir:
  #     df = df[df['direction'] == dir]

  #   if segment:
  #     if type(segment) == str:
  #       next_stop_id = segment
  #       assert next_stop_id in set(self.stop_dict[dir])
  #       df = df[df['next_stop_id'] == next_stop_id]
  #     elif type(segment) == int:
  #       next_stop_id = self.stop_dict[dir][segment]
  #       df = df[df['next_stop_id'] == next_stop_id]
    
  #   gt_crowded = is_crowded(df['next_stop_id'], df['passenger_count'], self.stop_stats, num_stds)
  #   pred_crowded = is_crowded(df['next_stop_id'], df['passenger_count_pred'], self.stop_stats, num_stds)

  #   print('Classification Report:')
  #   print(classification_report(gt_crowded, pred_crowded))
  #   print('\n')
  #   print('Confusion Matrix (0 = not crowded | 1 = crowded):')
  #   print(confusion_matrix(gt_crowded, pred_crowded))