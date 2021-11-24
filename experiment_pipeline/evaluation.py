import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    mean_absolute_error, 
    r2_score, 
    max_error,
    balanced_accuracy_score
)
  
def is_crowded(stop_id_pos, passenger_counts, stop_stats, method='mean', num_classes=2, spread_multiple=1):
    crowded = []
    for (stop_id, passenger_count) in zip(stop_id_pos, passenger_counts):
      if num_classes == 2:
        if method == 'mean':
          threshold = stop_stats[('passenger_count', 'mean')].loc[stop_id]
        elif method == 'q50':
          threshold = stop_stats[('passenger_count', 'q50')].loc[stop_id]
        elif method == 'q25q75':
          threshold = stop_stats[('passenger_count', 'q75')].loc[stop_id]
        elif method == 'std':
          std = stop_stats[('passenger_count', 'std')].loc[stop_id]
          threshold = stop_stats[('passenger_count', 'mean')].loc[stop_id] + spread_multiple * std
        elif method == 'iqr':
          iqr = stop_stats[('passenger_count', 'q75')].loc[stop_id] - stop_stats[('passenger_count', 'q25')].loc[stop_id]
          threshold = stop_stats[('passenger_count', 'q75')].loc[stop_id] + spread_multiple * iqr
        
        if passenger_count > threshold:
          crowded.append(1)
        else:
          crowded.append(0)
      
      elif num_classes == 3:
        if method == 'q25q75':
          upper_threshold = stop_stats[('passenger_count', 'q75')].loc[stop_id]
          lower_threshold = stop_stats[('passenger_count', 'q25')].loc[stop_id]
        elif method == 'std':
          std = stop_stats[('passenger_count', 'std')].loc[stop_id]
          upper_threshold = stop_stats[('passenger_count', 'mean')].loc[stop_id] + spread_multiple * std
          lower_threshold = stop_stats[('passenger_count', 'mean')].loc[stop_id] - spread_multiple * std
        elif method == 'iqr':
          iqr = stop_stats[('passenger_count', 'q75')].loc[stop_id] - stop_stats[('passenger_count', 'q25')].loc[stop_id]
          upper_threshold = stop_stats[('passenger_count', 'q75')].loc[stop_id] + spread_multiple * iqr
          lower_threshold = stop_stats[('passenger_count', 'q25')].loc[stop_id] - spread_multiple * iqr
        
        if passenger_count > upper_threshold:
          crowded.append(1)
        elif passenger_count < lower_threshold:
          crowded.append(-1)
        else:
          crowded.append(0)
      
    return crowded

class Evaluation:
  def __init__(self, global_feature_set=None, train=None, val=None, test=None, stop_id_ls=None, stop_stats=None):
    self.global_feature_set = global_feature_set
    self.train = train
    self.val = val
    self.test = test
    self.stop_id_ls = stop_id_ls
    self.stop_pos_ls = [i for (i, _) in enumerate(self.stop_id_ls)]
    self.stop_id2stop_pos = {stop_id : stop_pos for (stop_id, stop_pos) in zip(self.stop_id_ls, self.stop_pos_ls)}
    self.stop_pos2stop_id = {stop_pos : stop_id for (stop_id, stop_pos) in zip(self.stop_id_ls, self.stop_pos_ls)}
    self.stop_stats = stop_stats
    
  
  def basic_eval(self, data, segment=None, pretty_print=True):
    if data == 'train':
      df = self.train.copy()
      df_train = self.train.copy()
    elif data == 'val':
      df = self.val.copy()
      df_train = self.train.copy()

    elif data == 'test':
      df = self.test.copy()
      df_train = self.train.copy()

    if segment:
      df = df[df['next_stop_id_pos'] == segment]

    gt = df['passenger_count']
    gt_train = df_train['passenger_count']
    gt_train_mean = np.zeros_like(gt) + gt_train.mean()
    pred = df['passenger_count_pred']

    mae_pred = mean_absolute_error(gt, pred)
    max_error_pred = max_error(gt, pred)
    r2_pred = r2_score(gt, pred)

    mae_mean = mean_absolute_error(gt, gt_train_mean)
    max_error_mean = max_error(gt, gt_train_mean)
    r2_mean = r2_score(gt, gt_train_mean)

    model_pred_eval = (mae_pred, max_error_pred, r2_pred)
    mean_pred_eval = (mae_mean, max_error_mean, r2_mean)

    if pretty_print:
      print('Performance: Model Prediction')
      print(f'MAE: {mae_pred:.1f}')
      print(f'ME : {max_error_pred:.1f}')
      print(f'R^2: {r2_pred:.2f}')
      print('\n')
      print('Performance: Mean Prediction')
      print(f'MAE: {mae_mean:.1f}')
      print(f'ME : {max_error_mean:.1f}')
      print(f'R^2: {r2_mean:.2f}')

      return model_pred_eval, mean_pred_eval
    
    else:
      return model_pred_eval, mean_pred_eval
    
    
  def plot_passenger_count_by_time_of_day(self, data, segment=None, agg='sum'):
    if data == 'train':
      df = self.train.copy()
    elif data == 'val':
      df = self.val.copy()
    elif data == 'test':
      df = self.test.copy()

    if segment:
      df = df[df['next_stop_id_pos'] == segment]

    hours = list(range(24))
    df['day_type'] = df['timestamp'].apply(lambda x: 'weekday' if x.dayofweek < 5 else 'weekend')

    if agg == 'sum':
      gt = df.groupby([df['timestamp'].dt.hour, 'day_type'])['passenger_count'].sum().unstack()
      pred = df.groupby([df['timestamp'].dt.hour, 'day_type'])['passenger_count_pred'].sum().unstack()

      if ('weekday' in set(gt.columns)) and ('weekday' in set(pred.columns)):
        gt_weekday = gt['weekday']
        pred_weekday = pred['weekday']

        fig = plt.figure(figsize=(20, 10))
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

        fig = plt.figure(figsize=(20, 10))
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
        fig = plt.figure(figsize=(20, 10))
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
        fig = plt.figure(figsize=(20, 10))
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
    return fig


  def gt_pred_scatter(self, data, plot='basic', errors='all', n=100, s=50):
    if data == 'train':
      df = self.train.copy()
    elif data == 'val':
      df = self.val.copy()
    elif data == 'test':
      df = self.test.copy()

    df['pred_error'] = df['passenger_count_pred'] - df['passenger_count']
    df['pred_abs_error'] = df['pred_error'].abs()
    df['day_type'] = df['timestamp'].apply(lambda x: 'weekday' if x.dayofweek < 5 else 'weekend')

    if errors == 'large':
      df = df.sort_values(by=['pred_abs_error'], ascending=False).iloc[0:n, :]
    
    elif errors == 'small':
      df = df.sort_values(by=['pred_abs_error'], ascending=True).iloc[0:n, :]

    # weekday
    if 'weekday' in set(df['day_type']):
        if plot == 'basic':
          fig = plt.figure(figsize=(20, 20))
          gt = df[df['day_type'] == 'weekday']['passenger_count']
          pred = df[df['day_type'] == 'weekday']['passenger_count_pred']
          plt.scatter(pred, gt, s=s, marker='o', color='navy', alpha=0.25)
          plt.xlim([min(gt.min(), pred.min()), max(gt.max(), pred.max())])
          plt.ylim([min(gt.min(), pred.min()), max(gt.max(), pred.max())])
          plt.plot(plt.xlim(), plt.xlim(), color='darkorange', scalex=False, scaley=False)
          plt.xlabel('Predicted Passenger Count')
          plt.ylabel('Ground Truth Passenger Count')
          plt.title('Weekday')

        if plot == 'stop':
          fig = plt.figure(figsize=(20, 10))
          # model predictions too high (plot gt markers on top of pred markers)
          over_est_df = df[(df['pred_error'] >= 0) & (df['day_type'] == 'weekday')]
          over_est_stop_pos_obs = over_est_df['next_stop_id_pos']
          over_est_gt = over_est_df['passenger_count']
          over_est_errors = over_est_df['pred_abs_error']
          over_est_ss = [s * max(1, error) for error in over_est_errors]
          plt.scatter(over_est_stop_pos_obs, over_est_gt, s=over_est_ss, marker='o', label='Prediction', color='navy')
          plt.scatter(over_est_stop_pos_obs, over_est_gt, s=s, marker='o', label='Ground Truth', color='darkorange')

          # model predictions too low (plot pred markers on top of gt markers)
          under_est_df = df[(df['pred_error'] < 0) & (df['day_type'] == 'weekday')]
          under_est_stop_pos_obs = under_est_df['next_stop_id_pos']
          under_est_gt = under_est_df['passenger_count']
          under_est_errors = under_est_df['pred_abs_error']
          under_est_ss = [s * min(1, 1 / error) for error in under_est_errors]
          plt.scatter(under_est_stop_pos_obs, under_est_gt, s=s, marker='o', color='darkorange')
          plt.scatter(under_est_stop_pos_obs, under_est_gt, s=under_est_ss, marker='o', color='navy')

          plt.xticks(self.stop_pos_ls, self.stop_id_ls, rotation=90)
          plt.xlabel('Stop')
          plt.ylabel('Ground Truth Passenger Count')
          plt.title('Weekday')
          legend = plt.legend()
          for handle in legend.legendHandles:
            handle.set_sizes([s])
          plt.show()

        if plot == 'hour':
          fig = plt.figure(figsize=(20, 10))
          # model predictions too high (plot gt markers on top of pred markers)
          over_est_df = df[(df['pred_error'] >= 0) & (df['day_type'] == 'weekday')]
          over_est_hour_obs = over_est_df['hour']
          over_est_gt = over_est_df['passenger_count']
          over_est_errors = over_est_df['pred_abs_error']
          over_est_ss = [s * max(1, error) for error in over_est_errors]
          plt.scatter(over_est_hour_obs, over_est_gt, s=over_est_ss, marker='o', label='Prediction', color='navy')
          plt.scatter(over_est_hour_obs, over_est_gt, s=s, marker='o', label='Ground Truth', color='darkorange')

          # model predictions too low (plot pred markers on top of gt markers)
          under_est_df = df[(df['pred_error'] < 0) & (df['day_type'] == 'weekday')]
          under_est_hour_obs = under_est_df['hour']
          under_est_gt = under_est_df['passenger_count']
          under_est_errors = under_est_df['pred_abs_error']
          under_est_ss = [s * min(1, 1 / error) for error in under_est_errors]
          plt.scatter(under_est_hour_obs, under_est_gt, s=s, marker='o', color='darkorange')
          plt.scatter(under_est_hour_obs, under_est_gt, s=under_est_ss, marker='o', color='navy')

          hours = list(range(24))
          plt.xticks(hours)
          plt.xlabel('Time of Day')
          plt.ylabel('Ground Truth Passenger Count')
          plt.title('Weekday')
          legend = plt.legend()
          for handle in legend.legendHandles:
            handle.set_sizes([s])
          plt.show()

        if plot =='datetime':

          def find_hot_indices(datetime_array):
            indices = []
            for i in range(len(datetime_array)):
                if datetime_array[i].weekday() >= 5:
                    indices.append(i)
            return indices

          def find_rain_indices(datetime_array):
              indices = []
              for i in range(len(datetime_array)):
                  if datetime_array[i].weekday() < 5:
                      if datetime_array[i].hour >= 7 and datetime_array[i].hour <= 19:
                          indices.append(i)
              return indices

          fig = plt.figure(figsize=(20, 10))
          # model predictions too high (plot gt markers on top of pred markers)
          over_est_df = df[(df['pred_error'] >= 0) & (df['day_type'] == 'weekday')]
          over_est_timestamp_obs = over_est_df['timestamp']
          over_est_gt = over_est_df['passenger_count']
          over_est_errors = over_est_df['pred_abs_error']
          over_est_ss = [s * max(1, error) for error in over_est_errors]
          plt.scatter(over_est_timestamp_obs, over_est_gt, s=over_est_ss, marker='o', label='Prediction', color='navy')
          plt.scatter(over_est_timestamp_obs, over_est_gt, s=s, marker='o', label='Ground Truth', color='darkorange')

          # model predictions too low (plot pred markers on top of gt markers)
          under_est_df = df[(df['pred_error'] < 0) & (df['day_type'] == 'weekday')]
          under_est_timestamp_obs = under_est_df['timestamp']
          under_est_gt = under_est_df['passenger_count']
          under_est_errors = under_est_df['pred_abs_error']
          under_est_ss = [s * min(1, 1 / error) for error in under_est_errors]
          plt.scatter(under_est_timestamp_obs, under_est_gt, s=s, marker='o', color='darkorange')
          plt.scatter(under_est_timestamp_obs, under_est_gt, s=under_est_ss, marker='o', color='navy')

          start_date = df['timestamp'].dt.min()
          end_date = df['timestamp'].dt.max()
          plt.xticks(hours)
          plt.xlabel('Date')
          plt.ylabel('Ground Truth Passenger Count')
          plt.title('Weekday')
          legend = plt.legend()
          for handle in legend.legendHandles:
            handle.set_sizes([s])
          plt.show()

    # weekend
    if 'weekend' in set(df['day_type']):
        fig = plt.figure(figsize=(20, 10))
        if plot == 'basic':
          fig = plt.figure(figsize=(20, 20))
          gt = df[df['day_type'] == 'weekend']['passenger_count']
          pred = df[df['day_type'] == 'weekend']['passenger_count_pred']
          plt.scatter(pred, gt, s=s, marker='o', color='navy', alpha=0.25)
          plt.xlim([min(gt.min(), pred.min()), max(gt.max(), pred.max())])
          plt.ylim([min(gt.min(), pred.min()), max(gt.max(), pred.max())])
          plt.plot(plt.xlim(), plt.xlim(), color='darkorange', scalex=False, scaley=False)
          plt.xlabel('Predicted Passenger Count')
          plt.ylabel('Ground Truth Passenger Count')
          plt.title('Weekend')

        if plot == 'stop':
          # model predictions too high (plot gt markers on top of pred markers)
          over_est_df = df[(df['pred_error'] >= 0) & (df['day_type'] == 'weekend')]
          over_est_stop_pos_obs = over_est_df['next_stop_id_pos']
          over_est_gt = over_est_df['passenger_count']
          over_est_errors = over_est_df['pred_abs_error']
          over_est_ss = [s * max(1, error) for error in over_est_errors]
          plt.scatter(over_est_stop_pos_obs, over_est_gt, s=over_est_ss, marker='o', label='Prediction', color='navy')
          plt.scatter(over_est_stop_pos_obs, over_est_gt, s=s, marker='o', label='Ground Truth', color='darkorange')

          # model predictions too low (plot pred markers on top of gt markers)
          under_est_df = df[(df['pred_error'] < 0) & (df['day_type'] == 'weekend')]
          under_est_stop_pos_obs = under_est_df['next_stop_id_pos']
          under_est_gt = under_est_df['passenger_count']
          under_est_errors = under_est_df['pred_abs_error']
          under_est_ss = [s * min(1, 1 / error) for error in under_est_errors]
          plt.scatter(under_est_stop_pos_obs, under_est_gt, s=s, marker='o', color='darkorange')
          plt.scatter(under_est_stop_pos_obs, under_est_gt, s=under_est_ss, marker='o', color='navy')

          plt.xticks(self.stop_pos_ls, self.stop_id_ls, rotation=90)
          plt.xlabel('Stop')
          plt.ylabel('Ground Truth Passenger Count')
          plt.title('Weekend')
          legend = plt.legend()
          for handle in legend.legendHandles:
            handle.set_sizes([s])
          plt.show() 

        if plot == 'hour':
          fig = plt.figure(figsize=(20, 10))
          # model predictions too high (plot gt markers on top of pred markers)
          over_est_df = df[(df['pred_error'] >= 0) & (df['day_type'] == 'weekend')]
          over_est_hour_obs = over_est_df['hour']
          over_est_gt = over_est_df['passenger_count']
          over_est_errors = over_est_df['pred_abs_error']
          over_est_ss = [s * max(1, error) for error in over_est_errors]
          plt.scatter(over_est_hour_obs, over_est_gt, s=over_est_ss, marker='o', label='Prediction', color='navy')
          plt.scatter(over_est_hour_obs, over_est_gt, s=s, marker='o', label='Ground Truth', color='darkorange')

          # model predictions too low (plot pred markers on top of gt markers)
          under_est_df = df[(df['pred_error'] < 0) & (df['day_type'] == 'weekday')]
          under_est_hour_obs = under_est_df['hour']
          under_est_gt = under_est_df['passenger_count']
          under_est_errors = under_est_df['pred_abs_error']
          under_est_ss = [s * min(1, 1 / error) for error in under_est_errors]
          plt.scatter(under_est_hour_obs, under_est_gt, s=s, marker='o', color='darkorange')
          plt.scatter(under_est_hour_obs, under_est_gt, s=under_est_ss, marker='o', color='navy')

          hours = list(range(24))
          plt.xticks(hours)
          plt.xlabel('Time of Day')
          plt.ylabel('Ground Truth Passenger Count')
          plt.title('Weekday')
          legend = plt.legend()
          for handle in legend.legendHandles:
            handle.set_sizes([s])
          plt.show() 
    return fig


  def print_classification_metrics(self, data, segment=None, method='mean', num_classes=2, spread_multiple=1, pretty_print=True):

    if data == 'train':
      df = self.train.copy()
    elif data == 'val':
      df = self.val.copy()
    elif data == 'test':
      df = self.test.copy()

    if segment:
      df = df[df['next_stop_id_pos'] == segment]
    
    gt_crowded = is_crowded(df['next_stop_id_pos'], df['passenger_count'], self.stop_stats, method, num_classes, spread_multiple)
    pred_crowded = is_crowded(df['next_stop_id_pos'], df['passenger_count_pred'], self.stop_stats, method, num_classes, spread_multiple)
    
    bal_acc = balanced_accuracy_score(gt_crowded, pred_crowded)
    cr_str = classification_report(gt_crowded, pred_crowded)
    cr_dict = classification_report(gt_crowded, pred_crowded, output_dict=True)
    cm = confusion_matrix(gt_crowded, pred_crowded)

    if pretty_print:
      if num_classes == 2:
        print('Labels: 0 = not crowded | 1 = crowded')
      if num_classes == 3:
        print('Labels: -1 = sparse | 0 = normal | 1 = crowded')
      print('\n')
      print(f'Balanced Accuracy: {bal_acc}')
      print('\n')
      print('Classification Report:')
      print(cr_str)
      print('\n')
      print('Confusion Matrix:')
      print(cm)
    
      return bal_acc, cr_dict, cm

    else:
      return bal_acc, cr_dict, cm


