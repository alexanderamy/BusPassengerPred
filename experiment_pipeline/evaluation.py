import datetime
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

    fig_weekday = None
    fig_weekend = None

    if agg == 'sum':
      gt = df.groupby([df['timestamp'].dt.hour, 'day_type'])['passenger_count'].sum().unstack()
      pred = df.groupby([df['timestamp'].dt.hour, 'day_type'])['passenger_count_pred'].sum().unstack()

      if ('weekday' in set(gt.columns)) and ('weekday' in set(pred.columns)):
        gt_weekday = gt['weekday']
        pred_weekday = pred['weekday']

        fig_weekday = plt.figure(figsize=(20, 10))

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

        fig_weekend = plt.figure(figsize=(20, 10))

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
        fig_weekday = plt.figure(figsize=(20, 10))
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
        fig_weekend = plt.figure(figsize=(20, 10))
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
    return fig_weekday, fig_weekend

  def gt_pred_scatter(self, data, plot='basic', errors='all', n=100, s=50):

    if data == 'train':
      df = self.train.copy()
    elif data == 'val':
      df = self.val.copy()
    elif data == 'test':
      df = self.test.copy()

    df['pred_error'] = df['passenger_count_pred'] - df['passenger_count']
    df['pred_abs_error'] = df['pred_error'].abs()
    df['day_type'] = df['timestamp'].apply(lambda x: 'Weekday' if x.dayofweek < 5 else 'Weekend')
    day_types = ['Weekday', 'Weekend']

    # weather
    group_df = df.copy()[['timestamp', 'hour', 'Precipitation', 'Heat Index']]
    group_df['Precipitation'] = group_df['Precipitation'].apply(lambda x: 1 if x > 0 else 0)
    group_df['Heat Index'] = group_df['Heat Index'].apply(lambda x: 1 if x >= 90 else 0)
    group_df = group_df.groupby(by=[group_df['timestamp'].dt.date, 'hour']).max()
    precip_dts = [datetime.datetime(dt.year, dt.month, dt.day, hour, 0) for (dt, hour) in group_df[group_df['Precipitation'] == 1].index]
    heat_dts = [datetime.datetime(dt.year, dt.month, dt.day, hour, 0) for (dt, hour) in group_df[group_df['Heat Index'] == 1].index]

    if errors == 'large':
      df = df.sort_values(by=['pred_abs_error'], ascending=False).iloc[0:n, :]
    
    elif errors == 'small':
      df = df.sort_values(by=['pred_abs_error'], ascending=True).iloc[0:n, :]

    fig_weekday = None
    fig_weekend = None
    fig_datetime = None
    figs_dict = {'Weekday':fig_weekday, 'Weekend':fig_weekend, 'DateTime':fig_datetime}

    if plot == 'basic':
      for day_type in day_types:
        if day_type in set(df['day_type']):
          fig, ax = plt.subplots(figsize=(20, 20))
          gt = df[df['day_type'] == day_type]['passenger_count']
          pred = df[df['day_type'] == day_type]['passenger_count_pred']
          ax.scatter(pred, gt, s=s, marker='o', color='navy', alpha=0.25)
          ax.set_xlim([min(gt.min(), pred.min()), max(gt.max(), pred.max())])
          ax.set_ylim([min(gt.min(), pred.min()), max(gt.max(), pred.max())])
          ax.plot(ax.get_xlim(), ax.get_xlim(), color='darkorange', scalex=False, scaley=False)
          ax.set_xlabel('Predicted Passenger Count')
          ax.set_ylabel('Ground Truth Passenger Count')
          ax.set_title(day_type)
          fig.tight_layout()
          figs_dict[day_type] = fig
          plt.show()
    
    elif (plot == 'stop') or (plot == 'hour'):
      if plot == 'stop':
        col = 'next_stop_id_pos'
      else:
        col = 'hour'
      for day_type in day_types:
        if day_type in set(df['day_type']):
          fig, ax = plt.subplots(figsize=(20, 10))
          # model predictions too high (plot gt markers on top of pred markers)
          over_est_df = df[(df['pred_error'] >= 0) & (df['day_type'] == day_type)]
          over_est_col_obs = over_est_df[col]
          over_est_gt = over_est_df['passenger_count']
          over_est_errors = over_est_df['pred_abs_error']
          over_est_ss = [s * max(1, error) for error in over_est_errors]
          ax.scatter(over_est_col_obs, over_est_gt, s=over_est_ss, marker='o', label='Prediction', color='navy')
          ax.scatter(over_est_col_obs, over_est_gt, s=s, marker='o', label='Ground Truth', color='darkorange')
          # model predictions too low (plot pred markers on top of gt markers)
          under_est_df = df[(df['pred_error'] < 0) & (df['day_type'] == day_type)]
          under_est_col_obs = under_est_df[col]
          under_est_gt = under_est_df['passenger_count']
          under_est_errors = under_est_df['pred_abs_error']
          under_est_ss = [s * min(1, 1 / error) for error in under_est_errors]
          ax.scatter(under_est_col_obs, under_est_gt, s=s, marker='o', color='darkorange')
          ax.scatter(under_est_col_obs, under_est_gt, s=under_est_ss, marker='o', color='navy')
          if plot == 'stop':
            ax.set_xticks(self.stop_pos_ls)
            ax.set_xticklabels(self.stop_id_ls, rotation=90)
          else:
            hours = list(range(24))
            ax.set_xticks(hours)
          ax.set_xlabel(plot.capitalize())
          ax.set_ylabel('Ground Truth Passenger Count')
          ax.set_title(day_type)
          legend = plt.legend()
          for handle in legend.legendHandles:
            handle.set_sizes([s])
          fig.tight_layout()
          figs_dict[day_type] = fig
          plt.show()
      
    elif plot == 'datetime':
      fig, ax = plt.subplots(figsize=(20, 10))
      for i in range(len(precip_dts)):
        if i < len(precip_dts) - 1:
          ax.axvspan(precip_dts[i], (precip_dts[i] + datetime.timedelta(hours=1)), facecolor='blue', edgecolor='none', alpha=0.5)
        else:
          ax.axvspan(precip_dts[i], (precip_dts[i] + datetime.timedelta(hours=1)), label='Rain', facecolor='blue', edgecolor='none', alpha=0.5)
      for i in range(len(heat_dts)):
        if i < len(heat_dts) - 1:
          ax.axvspan(heat_dts[i], (heat_dts[i] + datetime.timedelta(hours=1)), facecolor='red', edgecolor='none', alpha=0.5)
        else:
          ax.axvspan(heat_dts[i], (heat_dts[i] + datetime.timedelta(hours=1)), label='Heat', facecolor='red', edgecolor='none', alpha=0.5)
      # model predictions too high (plot gt markers on top of pred markers)
      over_est_df = df[(df['pred_error'] >= 0)]
      over_est_timestamp_obs = over_est_df['timestamp']
      over_est_gt = over_est_df['passenger_count']
      over_est_errors = over_est_df['pred_abs_error']
      over_est_ss = [s * max(1, error) for error in over_est_errors]
      ax.scatter(over_est_timestamp_obs, over_est_gt, s=over_est_ss, marker='o', label='Prediction', color='navy')
      ax.scatter(over_est_timestamp_obs, over_est_gt, s=s, marker='o', label='Ground Truth', color='darkorange')
      # model predictions too low (plot pred markers on top of gt markers)
      under_est_df = df[(df['pred_error'] < 0)]
      under_est_timestamp_obs = under_est_df['timestamp']
      under_est_gt = under_est_df['passenger_count']
      under_est_errors = under_est_df['pred_abs_error']
      under_est_ss = [s * min(1, 1 / error) for error in under_est_errors]
      ax.scatter(under_est_timestamp_obs, under_est_gt, s=s, marker='o', color='darkorange')
      ax.scatter(under_est_timestamp_obs, under_est_gt, s=under_est_ss, marker='o', color='navy')
      ax.set_xlim(xmin=df['timestamp'].min().replace(microsecond=0, second=0, minute=0) - datetime.timedelta(hours=12), xmax=df['timestamp'].max().replace(microsecond=0, second=0, minute=0) + datetime.timedelta(hours=12))
      ax.set_xlabel('DateTime')
      ax.set_ylabel('Ground Truth Passenger Count')
      legend = ax.legend()
      for handle in legend.legendHandles:
        if handle.__class__.__name__ == 'PathCollection':
          handle.set_sizes([s])
      fig.tight_layout()
      figs_dict['DateTime'] = fig
      plt.show()

    return figs_dict['Weekday'], figs_dict['Weekend'], figs_dict['DateTime'] 


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


